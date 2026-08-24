# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
"""Structured headwise linear expansion layer."""

from __future__ import annotations

import math
from typing import Optional
import torch
from torch import nn


class LinearHeadwiseExpand(nn.Module):
    """Structured headwise linear expansion layer.

    Projects each of the num_heads slices independently with small_init.

    Args:
        in_features: Input feature dimension.
        num_heads: Number of heads.
        out_features: Output feature dimension. If None, derived from expand_factor.
        expand_factor: Expansion factor if out_features is not explicitly set. Default: 1.0.
        bias: Whether to include a learnable additive bias. Default: True.
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        expand_factor: Optional[float] = None,
        bias: bool = True,
    ):
        """Initializes headwise linear expansion layer."""
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError(f"LinearHeadwiseExpand: in_features must be a positive integer, got {in_features}")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"LinearHeadwiseExpand: num_heads must be a positive integer, got {num_heads}")
        if in_features % num_heads != 0:
            raise ValueError(f"LinearHeadwiseExpand: in_features ({in_features}) must be divisible by num_heads ({num_heads})")

        if expand_factor is not None and out_features is not None:
            raise ValueError(
                "LinearHeadwiseExpand: conflicting arguments: cannot specify both 'expand_factor' and 'out_features'. "
                "Specify only one."
            )
        self.in_features = in_features
        self.num_heads = num_heads
        if out_features is not None:
            if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features <= 0:
                raise ValueError(f"LinearHeadwiseExpand: out_features must be a strictly positive integer, got {out_features}")
            self.out_features = out_features
        elif expand_factor is not None:
            if not isinstance(expand_factor, (int, float)) or isinstance(expand_factor, bool) or not math.isfinite(expand_factor) or expand_factor <= 0:
                raise ValueError(f"LinearHeadwiseExpand: expand_factor must be a positive finite number, got {expand_factor}")
            self.out_features = round(expand_factor * in_features)
        else:
            self.out_features = in_features

        if self.out_features <= 0:
            raise ValueError(f"LinearHeadwiseExpand: resolved out_features must be strictly positive, got {self.out_features}")
        if self.out_features % num_heads != 0:
            raise ValueError(f"LinearHeadwiseExpand: out_features ({self.out_features}) must be divisible by num_heads ({num_heads})")

        in_per_head = self.in_features // self.num_heads
        out_per_head = self.out_features // self.num_heads

        self.weight = nn.Parameter(torch.empty(num_heads, out_per_head, in_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights with small_init normal distribution."""
        in_per_head = self.in_features // self.num_heads
        std = math.sqrt(2.0 / (5.0 * in_per_head))
        nn.init.normal_(self.weight.data, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies headwise linear transformation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        shape = x.shape
        x_view = x.view(*shape[:-1], self.num_heads, -1)
        out = torch.einsum("...hd,hod->...ho", x_view, self.weight)
        out = out.reshape(*shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias
        return out
