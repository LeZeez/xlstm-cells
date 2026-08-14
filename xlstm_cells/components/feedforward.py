# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math
import torch
from torch import nn
import torch.nn.functional as F

from .init import small_init_init_, wang_init_


class GatedFeedForward(nn.Module):
    """Gated FeedForward (GeGLU) layer for post up-projection in sLSTM blocks."""

    def __init__(
        self,
        d_model: int,
        proj_factor: float = 4.0 / 3.0,
        act_fn: str = "gelu",
        dropout: float = 0.0,
        bias: bool = False,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.proj_up_dim = round(proj_factor * d_model)
        self.act_fn = act_fn
        self.num_blocks = num_blocks

        self.proj_up = nn.Linear(d_model, 2 * self.proj_up_dim, bias=bias)
        self.proj_down = nn.Linear(self.proj_up_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        small_init_init_(self.proj_up.weight, dim=self.d_model)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.d_model, num_blocks=self.num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
