# xlstm-cells

Fast, pip-installable mLSTM and sLSTM cells with pure PyTorch foundation and an `nn.LSTM`-compatible interface, implementing the xLSTM architecture (Beck et al., 2024).

> **Warning: Experimental**
> This repository is experimental and under active development. Certain edge configurations, extreme scaling regimes, or custom hardware setups may exhibit instability.

```bash
pip install git+https://github.com/LeZeez/xlstm-cells.git
```

---

## Why?
As much as I admire [NX-AI/xlstm](https://github.com/NX-AI/xlstm) as a strong and robust repo, I found it rigid and confusing, at least for me. I found `nn.LSTM`-like classes much easier to build with and add custom features to. This repository targets learners and builders, providing a balanced middle ground, featuring:

1. **Pure PyTorch Base:** Every module works out-of-the-box with standard PyTorch on CPU and GPU without requiring external compilers or C++ toolchains. Good for learners.
2. **Optional Backends:** Integrates optional high-performance backends taken directly from the official NX-AI repositories:
   - **mLSTM:** Optional Triton chunkwise kernels (`xl_chunk` and `limit_chunk`) via [mlstm_kernels](https://github.com/NX-AI/mlstm_kernels).
   - **sLSTM:** Optional custom CUDA C++ extension with JIT compilation, taken from [NX-AI/xlstm](https://github.com/NX-AI/xlstm).
3. **Modern Architectures & Blocks:** Support for paper-compliant residual blocks (`mLSTMBlock`, `sLSTMBlock`) and official Large residual blocks (`xLSTMLargeBlock`) featuring SwiGLU feedforward, RMSNorm/LayerNorm options, asymmetric matrix memory dimensions, gate soft-capping, document packing (`boundaries=`), and TBPTT state management (`detach_states`, `zero_rows`).
4. **Auto-Fallback:** Zero-friction execution across training and generation. If Triton is unaligned with chunk size (e.g., generation step `T=1` or short prefill `T < chunk_size`), running on CPU, or compiling under `torch.compile`, the model automatically and transparently routes through the high-performance native parallel scan (emitting a standard runtime warning on CPU fallback).

---

## Overview of Classes

Base classes in `xlstm-cells` support dual initialization:
1. **Direct keyword arguments:** `mLSTMBlock(d_model=512, num_heads=8)`
2. **Dedicated config objects:** `mLSTMBlock(config=mLSTMBlockConfig(d_model=512, num_heads=8))`

---

### 1. Single-Step Recurrent Cells (nn.LSTMCell compatible)

| Class | Dedicated Config | Description |
|---|---|---|
| `mLSTMCell` | `mLSTMCellConfig` | Single-step bare matrix memory cell with exp-stabilizer m |
| `sLSTMCell` | `sLSTMCellConfig` | Single-step bare scalar memory cell with block-diagonal recurrence |

### 2. Multi-Layer Sequence Models (nn.LSTM compatible)

| Class | Dedicated Config | Description |
|---|---|---|
| `mLSTM` | `mLSTMConfig` | Multi-layer sequence mLSTM (Triton & native chunked parallel scan) |
| `sLSTM` | `sLSTMConfig` | Multi-layer sequence sLSTM (native/compiled vanilla scan & CUDA backend) |

### 3. Residual Blocks (Beck et al., 2024)

| Class | Dedicated Config | Description |
|---|---|---|
| `mLSTMBlock` | `mLSTMBlockConfig` | [Official Figure 11 block](https://arxiv.org/pdf/2405.04517#page=30) (Pre up-projection, conv q/k, SiLU) |
| `sLSTMBlock` | `sLSTMBlockConfig` | [Official Figure 10 block](https://arxiv.org/pdf/2405.04517#page=29) (Post up-projection, conv i/f, GeGLU MLP) |
| `xLSTMLargeBlock` | `xLSTMLargeBlockConfig` | Official Large residual block (Pre RMSNorm, SwiGLU, gate soft-capping) |

### 4. State Data Classes

| Class | Contained Fields | Description |
|---|---|---|
| `mLSTMState` | `.C`, `.n`, `.m` | Matrix memory state (`C` shape: `[B, H, Dh_qk, Dh_v]`, `n`: `[B, H, Dh_qk]`, `m`: `[B, H]`) |
| `sLSTMState` | `.c`, `.n`, `.m`, `.h` | Scalar memory state (`c`, `n`, `m`, `h` shape: `[B, H, Dh]`) |

---

## State & Training Functions

| Function | Usage | Description |
|---|---|---|
| `detach_states` | `detach_states(states)` | recursively detach all state tensors in nested tuples, lists, and dicts |
| `zero_rows` | `zero_rows(states, mask)` | in-place zero selected batch rows across states for continuous batching |

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

## Performance, Backends & Fallback

### mLSTM: Triton Acceleration & Transparent Fallback
When `mlstm_kernels` is installed, mLSTM layers and blocks execute hardware-accelerated Triton chunkwise kernels on NVIDIA GPUs when inputs are aligned with the chunk size.

If the sequence length is unaligned (e.g., single-step autoregressive generation `T=1` or short prefill `T < chunk_size`), running on CPU, or under `torch.compile`, the model automatically falls back to the native parallel chunked scan (emitting a runtime warning when falling back from CUDA Triton to CPU scan).

```bash
pip install mlstm_kernels
```

```python
mLSTM(128, 256, use_triton_kernels=True, chunkwise_kernel="xl_chunk", chunk_size=128, eps=1e-6)
mLSTMBlock(512, use_triton_kernels=True, chunkwise_kernel="xl_chunk", chunk_size=128, eps=1e-6)
```

### sLSTM: Compiled Vanilla Scan & CUDA Kernel
* **`backend="vanilla"` (Default):** Pure PyTorch sequential scan. Set `fast_mode=True` to compile the scan chunk-by-chunk via `torch.compile(dynamic=False)`:
  ```python
  sLSTM(128, 256, backend="vanilla", fast_mode=True, fast_chunk_size=32)
  sLSTMBlock(512, backend="vanilla", fast_mode=True, fast_chunk_size=32)
  ```
* **`backend="cuda"` (Optional):** Compiles the official C++/CUDA extension from source on the fly. 

*Note: When using backend="cuda", the constructor automatically emits a warning and forces fast_mode to False.*

### Activation Checkpointing
Reduces peak activation memory by recomputing forward steps during backward pass:

```python
mLSTMBlock(512, use_checkpoint=True)
sLSTMBlock(512, use_checkpoint=True)
xLSTMLargeBlock(config=xLSTMLargeBlockConfig(embedding_dim=512, use_checkpoint=True))
```

*Note: `use_checkpoint=True` adds latency and is often slower than `use_checkpoint=False`. Only use it when you have no memory for small batch size (e.g., <= 2)*

### Packed Sequences Without Padding (`boundaries`)
All blocks (`mLSTMBlock`, `sLSTMBlock`, `xLSTMLargeBlock`) and sequence models support document packing via the `boundaries` boolean mask:

```python
boundaries = torch.zeros(B, T, dtype=torch.bool, device=x.device)
boundaries[:, boundary_positions] = True  # True at the FIRST token of every packed document (BOS recommended here)

out, state = block(x, state, boundaries=boundaries)
```
At boundary positions, the forget pre-activation is set to `-1000.0`, unconditionally resetting the recurrent memory and eliminating cross-document attention and gradient leakage.

---

## Testing

```bash
git clone https://github.com/LeZeez/xlstm-cells.git
cd xlstm-cells
pytest
```

---

## License & Attribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Portions of this codebase (components, normalization layers, initialization routines, and CUDA kernels) are derived from [NX-AI/xlstm](https://github.com/NX-AI/xlstm), licensed under the Apache License, Version 2.0. See the [NOTICE](NOTICE) file for full copyright and attribution notices.

---

## Reference

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.
