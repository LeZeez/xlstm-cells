# xlstm-cells

Fast, pip-installable mLSTM and sLSTM cells with pure PyTorch foundation and an `nn.LSTM`-compatible interface, implementing the xLSTM architecture (Beck et al., 2024).

> **Warning: Experimental**
> This repository is experimental and under active development. Certain edge configurations, extreme scaling regimes, or custom hardware setups may exhibit instability. For previous release classes and historical checkpoints, checkout the `old-classes-v1` branch (`pip install git+https://github.com/LeZeez/xlstm-cells.git@old-classes-v1`).

```bash
pip install git+https://github.com/LeZeez/xlstm-cells.git
```

---

## Why?
As much as I found `NX-AI/xlstm` a strong and robust repo, as much I as I found it rigid and confusing, at least for me. I found `nn.LSTM`-like classes much easier to build with and add custom features in.

## Core Philosophy

1. **Pure PyTorch Base:** Every module works out-of-the-box with standard PyTorch on CPU and GPU without requiring external compilers or C++ toolchains.
2. **Optional Official Backends:** Integrates optional high-performance backends taken directly from the official NX-AI repositories:
   - **mLSTM:** Optional Triton chunkwise kernels (`xl_chunk` and `limit_chunk`) via `mlstm_kernels`.
   - **sLSTM:** Optional custom CUDA C++ extension with JIT compilation.

---

## Classes

| Class | Signature | Description |
|---|---|---|
| `mLSTMCell` | `mLSTMCell(input_size, hidden_size, num_heads=4)` | Single-step bare matrix memory cell |
| `sLSTMCell` | `sLSTMCell(input_size, hidden_size, num_heads=4)` | Single-step bare scalar memory cell with block-diagonal recurrence |
| `mLSTM` | `mLSTM(input_size, hidden_size, num_layers=1, num_heads=4, ...)` | Bare multi-layer sequence mLSTM (Triton & native chunked scan) |
| `sLSTM` | `sLSTM(input_size, hidden_size, num_layers=1, num_heads=4, ...)` | Bare multi-layer sequence sLSTM (compiled vanilla scan & CUDA backend) |
| `mLSTMBlock` | `mLSTMBlock(d_model, expand_factor=2, num_heads=4, ...)` | Official Figure 11 residual block with pre up-projection |
| `sLSTMBlock` | `sLSTMBlock(d_model, num_heads=4, mlp_factor=4/3, ...)` | Official Figure 10 residual block with post up-projection GeGLU MLP |
| `mLSTMState` | Dataclass with `.C`, `.n`, `.m` fields | Matrix memory state |
| `sLSTMState` | Dataclass with `.c`, `.n`, `.m`, `.h` fields | Scalar memory state |

---

## Functions

| Function | Usage |
|---|---|
| `detach_states` | `detach_states(states)` -- recursively detach all state tensors in nested structures |
| `zero_rows` | `zero_rows(states, mask)` -- in-place zero selected batch rows across states |

---

## Quick Examples

### 1. Cell-level (single step)

```python
import torch
from xlstm_cells import mLSTMCell, sLSTMCell

# Create bare cells
m_cell = mLSTMCell(input_size=128, hidden_size=256, num_heads=4)
s_cell = sLSTMCell(input_size=128, hidden_size=256, num_heads=4)

# Single step, like nn.LSTMCell
x_t = torch.randn(8, 128)
h_m, state_m = m_cell(x_t, m_cell.init_state(8))
h_s, state_s = s_cell(x_t, s_cell.init_state(8))

# Step repeatedly, carrying state
state = m_cell.init_state(8)
for t in range(10):
    h_m, state = m_cell(x_t, state)
```

### 2. Full Sequence Layers

```python
import torch
from xlstm_cells import mLSTM, sLSTM

# 2-layer bidirectional mLSTM
layer = mLSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
x = torch.randn(8, 50, 128)
out, states = layer(x)
```

### 3. Paper-Compliant Residual Blocks

```python
import torch
from xlstm_cells import mLSTMBlock, sLSTMBlock, detach_states

blocks = torch.nn.ModuleList([
    mLSTMBlock(d_model=512, expand_factor=2, num_heads=8),
    sLSTMBlock(d_model=512, num_heads=8),
    mLSTMBlock(d_model=512, expand_factor=2, num_heads=8),
])

x = torch.randn(4, 64, 512)

# Stateful loop across blocks
states = {}
for idx, blk in enumerate(blocks):
    s = states.get(idx)
    x, s = blk(x, s)
    states[idx] = s

states = detach_states(states)
```

---

## Performance & Backends

### mLSTM: Triton Acceleration (`mlstm_kernels`)
Install `mlstm_kernels` for hardware-accelerated chunkwise mLSTM recurrence on NVIDIA GPUs:

```bash
pip install mlstm_kernels
```

```python
mLSTM(128, 256, use_triton_kernels=True, chunkwise_kernel="xl_chunk", chunk_size=128, eps=1e-6)
mLSTMBlock(512, use_triton_kernels=True, chunkwise_kernel="xl_chunk", chunk_size=128, eps=1e-6)
```

If Triton is unavailable (CPU input, sequence length non-divisible by chunk size, or under `torch.compile`), it falls back automatically to the native chunked-parallel scan.

### sLSTM: Compiled Vanilla Scan & CUDA Kernel
* **`backend="vanilla"` (Default):** Pure PyTorch sequential scan. Set `fast_mode=True` to compile the scan chunk-by-chunk via `torch.compile(dynamic=False)`:
  ```python
  sLSTM(128, 256, backend="vanilla", fast_mode=True, fast_chunk_size=32)
  sLSTMBlock(512, backend="vanilla", fast_mode=True, fast_chunk_size=32)
  ```
* **`backend="cuda"` (Optional):** Compiles the official C++/CUDA extension from source on the fly. Note: `fast_mode` must be False when using the CUDA backend.

### Activation Checkpointing
Reduces peak activation memory by recomputing forward steps during backward:

```python
mLSTMBlock(512, use_checkpoint=True)
sLSTMBlock(512, use_checkpoint=True)
```

### Packed Sequences Without Padding (`boundaries`)
Both `mLSTMBlock` and `sLSTMBlock` support document packing via the `boundaries` boolean mask:

```python
boundaries = torch.zeros(B, T, dtype=torch.bool, device=x.device)
boundaries[:, boundary_positions] = True  # True at the FIRST token of every packed document

out, state = block(x, state, boundaries=boundaries)
```
At boundary positions, the forget pre-activation is set to `-1000.0`, unconditionally resetting the recurrent memory without cross-document pollution.

---

## Development & Testing

```bash
git clone https://github.com/LeZeez/xlstm-cells.git
cd xlstm-cells
pip install -e ".[dev]"
pytest
```

---

## Reference

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M.,
Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended
Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.
