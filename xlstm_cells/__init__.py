"""
xlstm-cells: Fast, pip-installable mLSTM and sLSTM cells with nn.LSTM-compatible interface.

Usage:
    from xlstm_cells import mLSTM, sLSTM, mLSTMCell, sLSTMCell

    # Full sequence (like nn.LSTM)
    layer = mLSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True)
    output, states = layer(x)       # x: (batch, seq, input_size) when batch_first=True

    # Single step (like nn.LSTMCell)
    cell = mLSTMCell(input_size=128, hidden_size=256)
    state = cell.init_state(batch_size=4)
    h_t, state = cell(x_t, state)   # step through time manually

    # TBPTT with state carry-over
    states = None
    for chunk in data.split(chunk_size):
        output, states = layer(chunk, states)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        states = tuple(s.detach() for s in states)  # truncate gradients per layer
"""

from .mlstm import mLSTMCell, mLSTM, mLSTMState
from .slstm import sLSTMCell, sLSTM, sLSTMState
from .block import mLSTMBlock, sLSTMBlock
from ._utils import detach_states, zero_rows

__version__ = "0.5.4"
__all__ = [
    "mLSTMCell",
    "mLSTM",
    "mLSTMState",
    "sLSTMCell",
    "sLSTM",
    "sLSTMState",
    "mLSTMBlock",
    "sLSTMBlock",
    "detach_states",
    "zero_rows",
]
