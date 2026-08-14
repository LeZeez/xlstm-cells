# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
"""Layer normalization and multi-head layer normalization modules."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer Normalization with optional residual weight parametrization.

    Args:
        ndim: Dimensionality of the normalized shape.
        weight: Whether to include learnable affine weight parameter. Default: True.
        bias: Whether to include learnable affine bias parameter. Default: False.
        eps: Small constant added to denominator for numerical stability. Default: 1e-5.
        residual_weight: If True, uses residual weight parametrization (1 + weight),
            initialized to zero. Default: True.
    """

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ):
        """Initializes LayerNorm with optional residual weighting."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> Optional[torch.Tensor]:
        """Returns the effective weight tensor taking residual parametrization into account."""
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization over the last dimension.

        Args:
            x: Input tensor of shape (*, ndim).

        Returns:
            Normalized tensor of same shape as input.
        """
        return F.layer_norm(
            x, normalized_shape=(self.ndim,), weight=self.weight_proxy, bias=self.bias, eps=self.eps
        )

    def reset_parameters(self) -> None:
        """Resets learnable parameters to their initial values."""
        if self.weight is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    """Per-token MultiHeadLayerNorm (GroupNorm where num_groups = num_heads).
    
    Normalizes each token (b, s) over its head dimension DH independently,
    preventing cross-time variance corruption.
    Supports both 4D (B, NH, S, DH) and 3D (B, S, C) layouts.
    """

    def forward(self, x: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
        """Applies multi-head layer normalization per token across head channels.

        Args:
            x: Input tensor of shape (B, NH, S, DH) or (B, S, C).
            num_heads: Number of attention/recurrent heads (required for 3D inputs).

        Returns:
            Normalized tensor matching the input shape.
        """
        if x.dim() == 4:
            B, NH, S, DH = x.shape
            gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
            gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
            out = F.group_norm(
                gn_in_2,
                num_groups=NH,
                weight=self.weight_proxy,
                bias=self.bias,
                eps=self.eps,
            )
            # (B * S, NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
            return out.view(B, S, NH, DH).transpose(1, 2)
        elif x.dim() == 3:
            B, S, C = x.shape
            assert num_heads is not None, "num_heads must be provided for 3D input (B, S, C)"
            gn_in = x.reshape(B * S, C)
            out = F.group_norm(
                gn_in,
                num_groups=num_heads,
                weight=self.weight_proxy,
                bias=self.bias,
                eps=self.eps,
            )
            return out.view(B, S, C)
        else:
            raise ValueError(f"MultiHeadLayerNorm expects 3D or 4D input, got {x.dim()}D")
