# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import math
import torch
from torch import nn


class LinearHeadwiseExpand(nn.Module):
    """Structured headwise projection layer.
    Input:  (B, T, in_features)
    Output: (B, T, out_features)
    Projects each of the num_heads slices independently with small_init.
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        out_features: int = None,
        expand_factor: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert in_features % num_heads == 0, f"in_features ({in_features}) must be divisible by num_heads ({num_heads})"
        self.in_features = in_features
        self.num_heads = num_heads
        if out_features is None:
            out_features = round(expand_factor * in_features)
        assert out_features % num_heads == 0, f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
        self.out_features = out_features

        in_per_head = in_features // num_heads
        out_per_head = out_features // num_heads

        self.weight = nn.Parameter(torch.empty(num_heads, out_per_head, in_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        in_per_head = self.in_features // self.num_heads
        std = math.sqrt(2.0 / (5.0 * in_per_head))
        nn.init.normal_(self.weight.data, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_view = x.view(*shape[:-1], self.num_heads, -1)
        out = torch.einsum("...hd,hod->...ho", x_view, self.weight)
        out = out.reshape(*shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias
        return out
