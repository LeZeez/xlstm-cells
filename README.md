# xlstm-cells

Pure-PyTorch mLSTM & sLSTM implementing the xLSTM paper (Beck et al., 2024) with an `nn.LSTM`-compatible interface. Optional `mlstm_kernels` triton backend for accelerated mLSTM recurrence.

```bash
pip install git+https://github.com/LeZeez/xlstm-cells.git
```

## Classes

| Class | Signature |
|---|---|
| `mLSTMCell` | `mLSTMCell(input_size, hidden_size, num_heads=4)` -- single-step recurrence |
| `sLSTMCell` | `sLSTMCell(input_size, hidden_size, num_heads=4)` -- single-step recurrence, block-diagonal per head |
| `mLSTM` | `mLSTM(input_size, hidden_size, num_layers=1, num_heads=4, bidirectional=False, dropout=0, bias=True, batch_first=False, pack_state=True, use_checkpoint=False, use_triton_kernels=True, chunkwise_kernel="limit_chunk", chunk_size=64)` -- full sequence |
| `sLSTM` | `sLSTM(input_size, hidden_size, num_layers=1, num_heads=4, bidirectional=False, dropout=0, bias=True, batch_first=False, pack_state=True, use_checkpoint=False, fast_mode=False, fast_chunk_size=32)` -- full sequence |
| `mLSTMBlock` | `mLSTMBlock(d_model, expand_factor=2, num_heads=4, conv_kernel=4, dropout=0, bias=True, use_checkpoint=False, use_triton_kernels=True, chunkwise_kernel="limit_chunk", chunk_size=64)` -- [Figure 11 residual block](https://arxiv.org/pdf/2405.04517#page=30) |
| `sLSTMBlock` | `sLSTMBlock(d_model, expand_factor=4/3, num_heads=4, conv_kernel=4, dropout=0, bias=False, use_checkpoint=False, fast_mode=False, fast_chunk_size=32)` -- [Figure 10 residual block](https://arxiv.org/pdf/2405.04517#page=29) |
| `mLSTMState` | Dataclass with `.C`, `.n`, `.m` fields |
| `sLSTMState` | Dataclass with `.c`, `.n`, `.m`, `.h` fields |

## Functions

| Function | Usage |
|---|---|
| `detach_states` | `detach_states(states)` -- recursively detach all tensors in nested dict/list/tuple/state |
| `zero_rows` | `zero_rows(states, mask)` -- in-place zero selected batch rows across nested states (`mask` is a bool tensor; `True` = zero that row) |

## Quick example

### Cell-level (single step)

```python
import torch
from xlstm_cells import mLSTMCell, sLSTMCell

# Create cells
m_cell = mLSTMCell(input_size=128, hidden_size=256, num_heads=4)
s_cell = sLSTMCell(input_size=128, hidden_size=256, num_heads=4)

# Single step, like nn.LSTMCell
x_t = torch.randn(8, 128)                    # (batch, input_size)
h_m, state_m = m_cell(x_t, m_cell.init_state(8))
h_s, state_s = s_cell(x_t, s_cell.init_state(8))

# Cell state shapes (no num_directions dimension)
assert state_m.C.shape == (8, 4, 64, 64)     # (B, H, Dh, Dh)
assert state_s.c.shape == (8, 4, 64)         # (B, H, Dh)

# Step repeatedly, carrying state
state = m_cell.init_state(8)
for t in range(10):
    h, state = m_cell(x_t, state)
```

### Layer-level (full sequence)

```python
import torch
from xlstm_cells import mLSTM

lstm = mLSTM(128, 256, num_layers=3, bidirectional=True, batch_first=True)
x = torch.randn(8, 50, 128)                  # (batch, seq, input_size)
output, states = lstm(x)                     # states = tuple of mLSTMState, one per layer

# output: (8, 50, 512)  -- hidden_size * 2 for bidirectional
# states[0].C: (2, 8, 4, 64, 64)  -- (D=2, B=8, H=4, Dh=64, Dh=64)
```

### Residual blocks (paper architecture)

```python
import torch
from xlstm_cells import mLSTMBlock, sLSTMBlock, detach_states

blocks = torch.nn.ModuleList([
    mLSTMBlock(d_model=512, expand_factor=2, num_heads=8),
    sLSTMBlock(d_model=512, num_heads=8),
    mLSTMBlock(d_model=512, expand_factor=2, num_heads=8),
])

x = torch.randn(4, 64, 512)                  # (batch, seq, d_model)

# Stateful loop across blocks and batches
states = {}
for block_idx, blk in enumerate(blocks):
    s = states.get(block_idx)
    x, s = blk(x, s)
    states[block_idx] = s
    # states[block_idx].C shape: (1, B, H, Dh, Dh)

states = detach_states(states)               # detach before next micro-batch
```

### Masking batch rows in state

```python
from xlstm_cells import mLSTMBlock, zero_rows

block = mLSTMBlock(d_model=256)
x = torch.randn(4, 10, 256)
out, state = block(x)                        # state is bare mLSTMState

# Zero out batch row 0 across all state tensors
zero_rows(state, torch.tensor([True, False, False, False]))
```

## Performance

### Triton kernels (mLSTM)

Install `mlstm_kernels` for hardware-accelerated mLSTM recurrence on NVIDIA GPUs:

```bash
pip install mlstm_kernels
```

```python
mLSTM(..., use_triton_kernels=True)       # default when mlstm_kernels is installed
mLSTMBlock(..., use_triton_kernels=True)
```

Requires sequence length divisible by 64. Falls back to the native chunked-parallel scan otherwise (with a warning). To avoid the fallback, pad your sequences to a multiple of 64.

### fast_mode (sLSTM)

Compiles the sLSTM sequential scan with `torch.compile` over fixed-size chunks:

```python
sLSTM(..., fast_mode=True, fast_chunk_size=32)
sLSTMBlock(..., fast_mode=True, fast_chunk_size=32)
```

Compile time is `O(fast_chunk_size)` and is not related to sequence length. Larger chunks fuse more aggressively but take longer to compile the first time. Works in both train and eval. The remainder (if `seq_len` is not a multiple of `fast_chunk_size`) runs eagerly.

### Activation checkpointing

Trades compute for memory -- recomputes activations during backward instead of storing them. Useful for long sequences or deep stacks:

```python
mLSTM(..., use_checkpoint=True)
sLSTM(..., use_checkpoint=True)
mLSTMBlock(..., use_checkpoint=True)
sLSTMBlock(..., use_checkpoint=True)
```

Only active during `model.train()`. Combine with TBPTT for maximum memory efficiency.

### Packed sequences without padding pollution (`boundaries`)

Both `mLSTM`/`mLSTMBlock` and `sLSTM`/`sLSTMBlock` accept an optional `boundaries` keyword:

```python
b = torch.zeros(B, T, dtype=torch.bool, device=x.device)
b[:, boundary_positions] = True       # True at the FIRST token of every packed doc

out, state = block(x, state, boundaries=b)
```

`boundaries` is a `(B, T)` bool tensor, `True` at the FIRST position of every packed document (typically `<|BOS|>` markers you insert between concatenated documents). At every `True` position the raw forget gate is forced to `-1000`, killing the cumulative forgetting factor from that position onward in the chunkwise recurrence — equivalent to resetting the recurrent state so the next packed document starts fresh.

This lets you pack multiple short documents into a single `seq_len` window with **zero** padding waste and without the recurrent-state pollution that PAD tokens cause. The model trains on close to 100% of every window instead of ~50% (the typical no-padding skip-short ratio).

When `boundaries=...` and `use_checkpoint=True` both hold, the inner `mLSTM.forward`/`sLSTM.forward` switches to `use_reentrant=True` automatically — required because the override's autograd edge trips PyTorch's `use_reentrant=False` saved-tensor count check. Override this behaviour with `xlstm_cells.set_packed_boundaries_override_mode(PackedBoundariesMode.DISABLE_CKPT_IN_PACKED)` (force `use_checkpoint=False` for any packed call), useful as a fallback on GPUs where re-entrant ckpt interacts badly with the chunkwise kernel.

The override is an approximation, not bit-exact: at boundary positions the carry into the C-state is bounded by `exp(min(0, i_tilde - m_prev + 1000))`, which is below bf16 epsilon for any realistic m-state.

### Forget-bias clamping for high-LR training stability

At aggressive learning rates the forget-gate bias can drift into saturation (`logsigmoid(b_f) ≈ 0`), which makes the log-normalizer `m` grow unboundedly and can silently defeat the boundary reset. All cells, layers, and blocks expose `clamp_forget_bias()` to keep the forget bias in a safe range:

```python
optimizer.step()
model.clamp_forget_bias()          # clamp forget bias to [-8.0, 8.0] by default
model.clamp_forget_bias(max_val=6.0)  # custom bound
```

Call it after every optimizer step when training at high LR with packed boundaries.

## TBPTT (Truncated Backpropagation Through Time)

All layers and blocks accept and return state, enabling stateful training over chunks for unlimited context windows on limited hardware:

```python
from xlstm_cells import mLSTM, detach_states

model = mLSTM(128, 256, batch_first=True, use_checkpoint=True)
optimizer = torch.optim.Adam(model.parameters())

states = None
for chunk in input_sequence.split(chunk_size, dim=1):
    output, states = model(chunk, states)
    loss = criterion(output, target_chunk)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    states = detach_states(states)  # truncate gradients at chunk boundary
```

Use `zero_rows(states, mask)` to reset state for specific batch rows (e.g., when a sequence in the batch ends mid-chunk).

## States

### mLSTMState

| Field | Cell shape | Layer shape |
|---|---|---|
| `C` | `(B, H, Dh, Dh)` | `(D, B, H, Dh, Dh)` |
| `n` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `m` | `(B, H)` | `(D, B, H)` |

### sLSTMState

| Field | Cell shape | Layer shape |
|---|---|---|
| `c` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `n` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `m` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `h` | `(B, H, Dh)` | `(D, B, H, Dh)` |

`B` = batch size, `H` = num_heads, `Dh` = head_dim (= hidden_size / num_heads),
`D` = num_directions (1 or 2).

- `mLSTMCell.init_state(batch)` and `sLSTMCell.init_state(batch)` return cell shapes (no D dim).
- `mLSTM.init_state(batch)` and `sLSTM.init_state(batch)` return a `tuple` of layer-shape states (one per layer). Set `pack_state=False` with `num_layers=1` to get a bare state.
- Blocks use `pack_state=False` internally -- `block(x)` returns a bare state.
- All fields are plain `torch.Tensor`. Each state has `.detach()`, `.to()`, `.clone()` methods.

## Development

```bash
git clone https://github.com/LeZeez/xlstm-cells.git
cd xlstm-cells
pip install -e ".[dev]"
pytest
```

## `mlstm_kernels` Support

`xlstm-cells` optionally integrates with the [`mlstm_kernels`](https://github.com/NX-AI/mlstm_kernels) package for hardware-accelerated mLSTM recurrence via Triton kernels on NVIDIA GPUs.

```bash
pip install mlstm_kernels
```

### Supported kernels

We support two chunkwise kernels from `mlstm_kernels`. Both use **exponential input gating** (`i_prime = exp(i_tilde - m)`) with a running log-space max state `m` for numerical stability. When triton is unavailable (missing package, CPU input, non-divisible sequence length, or `torch.compile`), the native chunked-parallel scan is used automatically — same exp-gate math, no semantic change.

| Short name | Full internal name | Description |
|---|---|---|
| `limit_chunk` | `chunkwise--triton_limit_chunk` | Standard TFLA chunkwise kernel. Default. |
| `xl_chunk` | `chunkwise--triton_xl_chunk` | TFLA kernel optimized for larger chunk sizes. |

Select a kernel and chunk size when creating an `mLSTM` or `mLSTMBlock`:

```python
from xlstm_cells import mLSTM, mLSTMBlock

# Default: limit_chunk, chunk_size=64
layer = mLSTM(128, 256, use_triton_kernels=True)

# xl_chunk with larger chunks
layer = mLSTM(128, 256, use_triton_kernels=True,
              chunkwise_kernel="xl_chunk", chunk_size=128)

# Same args on blocks
block = mLSTMBlock(512, use_triton_kernels=True,
                   chunkwise_kernel="xl_chunk", chunk_size=128)
```

### Requirements

- NVIDIA GPU with CUDA support
- `mlstm_kernels` installed (`pip install mlstm_kernels`)
- Sequence length must be divisible by `chunk_size` (default 64) for triton acceleration
- `torch.compile` is **not** compatible with `mlstm_kernels` (crashes inside Inductor); the native scan is used as fallback

### Unsupported kernels

The `mlstm_kernels` package also ships `xl_chunk_siging` (sigmoid input gating from the [TFLA paper](https://arxiv.org/abs/2503.14376)). We do not support it because it uses fundamentally different gate math (`sigmoid` vs `exp`) with no semantically equivalent native fallback, no inference/step kernels, and NX-AI does not use it in production.

## Reference

Beck, M., Poppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M.,
Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended
Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.
