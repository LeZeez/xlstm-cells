"""
xLSTM Blocks — paper-compliant residual blocks with GroupNorm, causal conv, gating.

These are the fundamental building blocks, not customizable wrappers.
They wrap the bare mLSTM/sLSTM cells with the full paper architecture:

    mLSTMBlock (pre up-projection, Figure 11):
        LN → up-project → Conv1d(causal) → mLSTM → GroupNorm → gate → down-project + residual

    sLSTMBlock (post up-projection, Figure 10):
        LN → [Conv1d] → sLSTM → GroupNorm → up-project → gated MLP → down-project + residual
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mlstm import mLSTM, mLSTMState
from .slstm import sLSTM, sLSTMState


class mLSTMBlock(nn.Module):
    """Paper-compliant mLSTM block with pre up-projection.

    Architecture (Figure 11 from Beck et al. 2024):
        x → LayerNorm ─┬→ gate_proj → sigmoid → gate
                        └→ up_proj → Conv1d(causal) → Swish → mLSTM → GroupNorm
                              → gate * lstm_out + learnable_skip → down_proj + x

    Args:
        d_model:        input & output feature dimension
        expand_factor:  up-projection multiplier (paper uses 2)
        num_heads:      number of mLSTM heads
        conv_kernel:    causal conv1d kernel size (paper uses 4, set 0 to disable)
        dropout:        dropout on output
        bias:           whether linear layers use bias
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: int = 2,
        num_heads: int = 4,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        expanded = d_model * expand_factor
        self.expanded = expanded
        self.num_heads = num_heads
        self.conv_kernel = conv_kernel

        self.ln = nn.LayerNorm(d_model)

        # --- gate path: produces sigmoid gate from normed input ---
        self.gate_proj = nn.Linear(d_model, expanded, bias=bias)

        # --- lstm path: up-project → conv → lstm ---
        self.up_proj = nn.Linear(d_model, expanded, bias=bias)

        if conv_kernel > 0:
            self.conv = nn.Conv1d(
                expanded, expanded,
                kernel_size=conv_kernel,
                groups=expanded,         # depthwise: each channel independently
                bias=bias,
            )
        else:
            self.conv = None

        self.lstm = mLSTM(
            expanded, expanded,
            num_layers=1,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            pack_state=False,
        )

        # --- head-wise normalization after LSTM ---
        self.gn = nn.GroupNorm(num_heads, expanded)

        # --- learnable skip (per-channel scalar added post-conv) ---
        self.learnable_skip = nn.Parameter(torch.zeros(1, 1, expanded))

        # --- back to d_model ---
        self.down_proj = nn.Linear(expanded, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.learnable_skip)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[mLSTMState] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        residual = x
        x = self.ln(x)

        # gate from normed input
        gate = torch.sigmoid(self.gate_proj(x))   # (B, T, expanded)

        # lstm path
        h = self.up_proj(x)                        # (B, T, expanded)

        if self.conv is not None:
            h = h.transpose(1, 2)                  # (B, expanded, T) for Conv1d
            h = F.pad(h, (self.conv_kernel - 1, 0))  # causal: pad left only
            h = self.conv(h)                       # (B, expanded, T)
            h = h.transpose(1, 2)                  # back to (B, T, expanded)

        h = F.silu(h)                               # Swish activation
        h = h + self.learnable_skip                 # learnable per-channel bias

        h, state = self.lstm(h, state)              # mLSTM recurrence

        # head-wise norm
        h = h.transpose(1, 2)                       # (B, expanded, T)
        h = self.gn(h)
        h = h.transpose(1, 2)                       # (B, T, expanded)

        h = gate * h                                 # apply external output gate

        h = self.down_proj(h)
        h = self.dropout(h)

        return h + residual, state

    def init_state(self, batch_size: int, device=None, dtype=None):
        return self.lstm.init_state(batch_size, device, dtype)


class sLSTMBlock(nn.Module):
    """Paper-compliant sLSTM block with post up-projection.

    Architecture (Figure 10 from Beck et al. 2024):
        x → LayerNorm → [Conv1d] → sLSTM → GroupNorm ─→ up_proj → GeLU ─→ down_proj + x
                                                     └→ gate_proj → sigmoid ┘

    Args:
        d_model:        input & output feature dimension
        expand_factor:  post up-projection multiplier (paper uses 4/3 ≈ 1.33)
        num_heads:      number of sLSTM heads
        conv_kernel:    causal conv1d kernel size (paper uses 4, set 0 to disable)
        dropout:        dropout on output
        bias:           whether linear layers use bias
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: float = 4.0 / 3.0,
        num_heads: int = 4,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        expanded = int(d_model * expand_factor)
        self.expanded = expanded
        self.num_heads = num_heads
        self.conv_kernel = conv_kernel

        self.ln = nn.LayerNorm(d_model)

        # optional causal conv on input
        if conv_kernel > 0:
            self.conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=conv_kernel,
                groups=d_model,          # depthwise
                bias=bias,
            )
        else:
            self.conv = None

        # sLSTM operates at d_model (no pre up-projection)
        self.lstm = sLSTM(
            d_model, d_model,
            num_layers=1,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            pack_state=False,
        )

        # head-wise normalization after sLSTM
        self.gn = nn.GroupNorm(num_heads, d_model)

        # post up-projection (gated MLP)
        self.up_proj = nn.Linear(d_model, expanded, bias=bias)
        self.gate_proj = nn.Linear(d_model, expanded, bias=bias)
        self.down_proj = nn.Linear(expanded, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[sLSTMState] = None,
    ) -> Tuple[torch.Tensor, sLSTMState]:
        residual = x
        x = self.ln(x)

        # optional causal conv before LSTM
        if self.conv is not None:
            c = x.transpose(1, 2)                  # (B, d_model, T)
            c = F.pad(c, (self.conv_kernel - 1, 0))
            c = self.conv(c)
            c = c.transpose(1, 2)                  # (B, T, d_model)
            x = x + F.silu(c)                       # additive conv with Swish

        x, state = self.lstm(x, state)              # sLSTM recurrence

        # head-wise norm
        x = x.transpose(1, 2)
        x = self.gn(x)
        x = x.transpose(1, 2)

        # post up-projection: gated MLP
        gate = torch.sigmoid(self.gate_proj(x))     # (B, T, expanded)
        up = F.gelu(self.up_proj(x))                # (B, T, expanded)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)

        return x + residual, state

    def init_state(self, batch_size: int, device=None, dtype=None):
        return self.lstm.init_state(batch_size, device, dtype)
