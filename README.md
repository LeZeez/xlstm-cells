# xlstm-cells

Pure-PyTorch mLSTM & sLSTM implementing the xLSTM paper (Beck et al., 2024) with slight similarity to `torch.nn.LSTM` structure. (optional `mlstm_kernels` support)

```bash
pip install git+https://github.com/LeZeez/xlstm-cells.git
```

## Classes

| Class | Signature |
|---|---|
| `mLSTMCell` | `mLSTMCell(input_size, hidden_size, num_heads=4)` — single-step recurrence |
| `sLSTMCell` | `sLSTMCell(input_size, hidden_size, num_heads=4)` — single-step recurrence, block-diagonal per head |
| `mLSTM` | `mLSTM(input_size, hidden_size, num_layers=1, num_heads=4, bidirectional=False, dropout=0, bias=True, batch_first=False)` — full sequence |
| `sLSTM` | `sLSTM(input_size, hidden_size, num_layers=1, num_heads=4, bidirectional=False, dropout=0, bias=True, batch_first=False)` — full sequence |
| `mLSTMBlock` | `mLSTMBlock(d_model, expand_factor=2, num_heads=4, conv_kernel=4, dropout=0, bias=True)` — [Figure 11 residual block in the original paper](https://arxiv.org/pdf/2405.04517#page=30) |
| `sLSTMBlock` | `sLSTMBlock(d_model, expand_factor=4/3, num_heads=4, conv_kernel=4, dropout=0, bias=False)` — [Figure 10 residual block in the original paper](https://arxiv.org/pdf/2405.04517#page=29) |
| `mLSTMState` | Dataclass with `.C`, `.n`, `.m` fields |
| `sLSTMState` | Dataclass with `.c`, `.n`, `.m`, `.h` fields |

## Functions

| Function | Usage |
|---|---|
| `detach_states` | `detach_states(states)` — recursively detach all tensors in nested dict/list/tuple/state |
| `zero_rows` | `zero_rows(states, mask)` — in-place zero selected batch rows across nested states (`mask` is a tensor of bools corresponding to the index of the batch you want to zero — `True` = zero)|

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

# output: (8, 50, 512)  — hidden_size * 2 for bidirectional
# states[0].C: (2, 8, 4, 64, 64)  — (D=2, B, H, Dh, Dh)
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
    # states[block_idx].C shape: (1, 4, H, Dh, Dh)

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

## States

### mLSTMState

| Field | Cell shape `(B, H, Dh)` | Layer shape `(D, B, H, Dh)` |
|---|---|---|
| `C` | `(B, H, Dh, Dh)` | `(D, B, H, Dh, Dh)` |
| `n` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `m` | `(B, H)` | `(D, B, H)` |

### sLSTMState

| Field | Cell shape `(B, H, Dh)` | Layer shape `(D, B, H, Dh)` |
|---|---|---|
| `c` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `n` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `m` | `(B, H, Dh)` | `(D, B, H, Dh)` |
| `h` | `(B, H, Dh)` | `(D, B, H, Dh)` |

`B` = batch size, `H` = num_heads, `Dh` = head_dim (= hidden_size / num_heads),
`D` = num_directions (1 or 2).

- `mLSTMCell.init_state(batch)` and `sLSTMCell.init_state(batch)` return cell shapes (no D dim).
- `mLSTM.init_state(batch)` and `sLSTM.init_state(batch)` return a `tuple` of layer-shape states (one per layer). Set `pack_state=False` with `num_layers=1` to get a bare state.
- Blocks use `pack_state=False` internally — `block(x)` returns a bare state.
- All fields are plain `torch.Tensor`. Each state has `.detach()`, `.to()`, `.clone()` methods.

## Development

```bash
git clone https://github.com/LeZeez/xlstm-cells.git
cd xlstm-cells
pip install -e ".[dev]"
pytest
```

## Reference

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M.,
Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended
Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.
