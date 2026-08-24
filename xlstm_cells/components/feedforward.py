# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
"""Gated feedforward (GeGLU) module for sLSTM blocks."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from .init import small_init_init_, wang_init_


class GatedFeedForward(nn.Module):
    """Gated FeedForward (GeGLU) layer for post up-projection in sLSTM blocks.

    Args:
        d_model: Model hidden dimension.
        proj_factor: Projection expansion factor. If None and proj_up_dim is None, defaults to 4.0 / 3.0.
        act_fn: Activation function ("gelu", "silu", "swish", "relu"). Default: "gelu".
        dropout: Dropout probability. Default: 0.0.
        bias: Whether linear projections use bias. Default: False.
        num_blocks: Number of stacked blocks in the model (for Wang init scaling). Default: 1.
        proj_up_dim: Explicit projection up dimension. Cannot be combined with proj_factor.
        hidden_size: Alias for proj_up_dim.
    """

    def __init__(
        self,
        d_model: int,
        proj_factor: Optional[float] = None,
        act_fn: str = "gelu",
        dropout: float = 0.0,
        bias: bool = False,
        num_blocks: int = 1,
        proj_up_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ):
        """Initializes GatedFeedForward layer."""
        super().__init__()
        if proj_up_dim is not None and hidden_size is not None:
            raise ValueError(
                "GatedFeedForward: conflicting arguments: cannot specify both 'proj_up_dim' and 'hidden_size'. "
                "Specify only one."
            )
        explicit_up = proj_up_dim if proj_up_dim is not None else hidden_size
        if proj_factor is not None and explicit_up is not None:
            raise ValueError(
                "GatedFeedForward: conflicting arguments: cannot specify both 'proj_factor' and 'proj_up_dim'/'hidden_size'. "
                "Specify only one."
            )
        if explicit_up is not None:
            if not isinstance(explicit_up, int) or isinstance(explicit_up, bool) or explicit_up <= 0:
                raise ValueError(f"GatedFeedForward: proj_up_dim must be a strictly positive integer, got {explicit_up}")
            self.proj_up_dim = explicit_up
        elif proj_factor is not None:
            if not isinstance(proj_factor, (int, float)) or isinstance(proj_factor, bool) or not math.isfinite(proj_factor) or proj_factor <= 0:
                raise ValueError(f"GatedFeedForward: proj_factor must be a positive finite number, got {proj_factor}")
            self.proj_up_dim = round(proj_factor * d_model)
        else:
            self.proj_up_dim = round((4.0 / 3.0) * d_model)

        if self.proj_up_dim <= 0:
            raise ValueError(f"GatedFeedForward: computed proj_up_dim must be strictly positive, got {self.proj_up_dim}")

        self.d_model = d_model
        self.act_fn = act_fn
        self.num_blocks = num_blocks

        self.proj_up = nn.Linear(d_model, 2 * self.proj_up_dim, bias=bias)
        self.proj_down = nn.Linear(self.proj_up_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights with small_init and wang_init schemes."""
        small_init_init_(self.proj_up.weight, dim=self.d_model)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.d_model, num_blocks=self.num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gated feedforward transformation.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        gate_raw, up = self.proj_up(x).split(self.proj_up_dim, dim=-1)
        if self.act_fn == "gelu":
            act = F.gelu(gate_raw)
        elif self.act_fn == "silu" or self.act_fn == "swish":
            act = F.silu(gate_raw)
        elif self.act_fn == "relu":
            act = F.relu(gate_raw)
        else:
            act = F.gelu(gate_raw)
        return self.dropout(self.proj_down(act * up))
