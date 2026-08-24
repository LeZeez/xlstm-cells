# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
"""Layer normalization, RMS normalization, and multi-head normalization modules."""

from __future__ import annotations

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
    Supports 4D layouts ((B, NH, S, DH) and (B, S, NH, DH)) and 3D layout (B, S, C).
    """

    def forward(self, x: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
        """Applies multi-head layer normalization per token across head channels.

        Args:
            x: Input tensor of shape (B, NH, S, DH), (B, S, NH, DH), or (B, S, C).
            num_heads: Number of attention/recurrent heads (required for 3D inputs).

        Returns:
            Normalized tensor matching the input shape.
        """
        if x.dim() == 4:
            B, D1, D2, DH = x.shape
            if self.ndim > 0 and D2 * DH == self.ndim:
                # Layout is (B, S, NH, DH) where D2 = NH, D1 = S
                S, NH = D1, D2
                gn_in = x.reshape(B * S, NH * DH)
                out = F.group_norm(
                    gn_in,
                    num_groups=NH,
                    weight=self.weight_proxy,
                    bias=self.bias,
                    eps=self.eps,
                )
                return out.view(B, S, NH, DH)
            elif num_heads is not None and D2 == num_heads:
                # Layout is (B, S, NH, DH) where D2 = num_heads
                S, NH = D1, D2
                gn_in = x.reshape(B * S, NH * DH)
                out = F.group_norm(
                    gn_in,
                    num_groups=NH,
                    weight=self.weight_proxy,
                    bias=self.bias,
                    eps=self.eps,
                )
                return out.view(B, S, NH, DH)
            else:
                # Layout is (B, NH, S, DH) where D1 = NH, D2 = S
                NH, S = D1, D2
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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input tensor by the root mean square of the last dimension:
        y = x / sqrt(mean(x^2) + eps) * weight + bias

    Args:
        ndim: Number of features in the input tensor.
        eps: Small constant added to denominator for numerical stability. Default: 1e-6.
        weight: Whether to include learnable affine scale parameter. Default: True.
        bias: Whether to include learnable affine bias parameter. Default: False.
        force_float32_reductions: Whether to compute reductions in float32 for stability. Default: True.
    """

    def __init__(
        self,
        ndim: int,
        eps: float = 1e-6,
        weight: bool = True,
        bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.ndim = ndim
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if weight:
            self.weight = nn.Parameter(torch.ones(ndim))
        else:
            self.weight = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.bias = None

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_calc = x.float() if self.force_float32_reductions else x
        norm = x_calc * torch.rsqrt(x_calc.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._rms_normalize(x)
        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadRMSNorm(RMSNorm):
    """Multi-head per-token RMSNorm.

    Normalizes the head dimension DH of the input tensor across heads.
    Supports both 4D ((B, NH, S, DH) and (B, S, NH, DH)) and 3D (B, S, C) layouts.
    """

    def __init__(
        self,
        ndim: int = -1,
        num_heads: int = -1,
        head_dim: int = -1,
        eps: float = 1e-6,
        weight: bool = True,
        bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        if ndim <= 0 and num_heads > 0 and head_dim > 0:
            ndim = num_heads * head_dim
        super().__init__(
            ndim=ndim,
            eps=eps,
            weight=weight,
            bias=bias,
            force_float32_reductions=force_float32_reductions,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
        if x.dim() == 4:
            B, D1, D2, DH = x.shape
            if self.ndim > 0 and D2 * DH == self.ndim:
                # (B, S, NH, DH)
                S, NH = D1, D2
                x_norm = self._rms_normalize(x)
                out = x_norm.reshape(B, S, NH * DH)
                if self.weight is not None: out = out * self.weight
                if self.bias is not None: out = out + self.bias
                return out.view(B, S, NH, DH)
            else:
                # (B, NH, S, DH)
                NH, S = D1, D2
                x_norm = self._rms_normalize(x)
                out = x_norm.transpose(1, 2).reshape(B, S, NH * DH)
                if self.weight is not None: out = out * self.weight
                if self.bias is not None: out = out + self.bias
                return out.view(B, S, NH, DH).transpose(1, 2)
        elif x.dim() == 3:
            B, S, C = x.shape
            nh = num_heads if num_heads is not None else self.num_heads
            assert nh > 0, "num_heads must be specified for 3D input to MultiHeadRMSNorm"
            dh = C // nh
            x_view = x.view(B, S, nh, dh)
            x_norm = self._rms_normalize(x_view)
            out = x_norm.reshape(B, S, C)
            if self.weight is not None:
                out = out * self.weight
            if self.bias is not None:
                out = out + self.bias
            return out
        else:
            raise ValueError(f"MultiHeadRMSNorm expects 3D or 4D input, got {x.dim()}D")
