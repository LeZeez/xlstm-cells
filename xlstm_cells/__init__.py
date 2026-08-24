"""
xlstm-cells: Fast, pip-installable mLSTM and sLSTM cells with nn.LSTM-compatible interface.

Usage:
    from xlstm_cells import mLSTM, sLSTM, mLSTMCell, sLSTMCell, mLSTMBlock, sLSTMBlock, xLSTMLargeBlock

    # Full sequence (like nn.LSTM)
    layer = mLSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True)
    output, states = layer(x)       # x: (batch, seq, input_size) when batch_first=True

    # Single step (like nn.LSTMCell)
    cell = mLSTMCell(input_size=128, hidden_size=256)
    state = cell.init_state(batch_size=4)
    h_t, state = cell(x_t, state)   # step through time manually

    # xLSTMLargeBlock Residual Block
    block = xLSTMLargeBlock(config=xLSTMLargeBlockConfig(embedding_dim=512, num_heads=8))
"""

from __future__ import annotations

from .mlstm import mLSTMCell, mLSTM, mLSTMState
from .slstm import sLSTMCell, sLSTM, sLSTMState
from .block import mLSTMBlock, sLSTMBlock
from .xlstm_large import xLSTMLargeBlock, xLSTMLargeLayer
from .configs import (
    mLSTMCellConfig,
    sLSTMCellConfig,
    mLSTMConfig,
    sLSTMConfig,
    mLSTMBlockConfig,
    sLSTMBlockConfig,
    xLSTMLargeBlockConfig,
)
from ._utils import (
    detach_states,
    zero_rows,
    PackedBoundariesMode,
    get_packed_boundaries_override_mode,
    set_packed_boundaries_override_mode,
)
from .components.ln import LayerNorm, MultiHeadLayerNorm, RMSNorm, MultiHeadRMSNorm
from .components.conv import CausalConv1d, conv1d_step
from .components.feedforward import GatedFeedForward, SwiGLUFeedForward
from .components.linear_headwise import LinearHeadwiseExpand
from .components.utils import soft_cap, round_up_to_next_multiple_of

__version__ = "0.6.1"
__all__ = [
    "mLSTMCell",
    "mLSTM",
    "mLSTMState",
    "sLSTMCell",
    "sLSTM",
    "sLSTMState",
    "mLSTMBlock",
    "sLSTMBlock",
    "xLSTMLargeBlock",
    "xLSTMLargeLayer",
    "mLSTMCellConfig",
    "sLSTMCellConfig",
    "mLSTMConfig",
    "sLSTMConfig",
    "mLSTMBlockConfig",
    "sLSTMBlockConfig",
    "xLSTMLargeBlockConfig",
    "LayerNorm",
    "MultiHeadLayerNorm",
    "RMSNorm",
    "MultiHeadRMSNorm",
    "CausalConv1d",
    "conv1d_step",
    "GatedFeedForward",
    "SwiGLUFeedForward",
    "LinearHeadwiseExpand",
    "soft_cap",
    "round_up_to_next_multiple_of",
    "detach_states",
    "zero_rows",
    "PackedBoundariesMode",
    "get_packed_boundaries_override_mode",
    "set_packed_boundaries_override_mode",
]
