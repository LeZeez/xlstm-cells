# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
"""Causal 1D depthwise convolution module for local context mixing."""

from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


def conv1d_step(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single step of causal 1D depthwise convolution.
    Args:
        x: (B, 1, D)
        conv_state: (B, KS, D)
        conv1d_weight: (KS, D)
    Returns:
        y: (B, 1, D), new_conv_state: (B, KS, D)
    """
    assert x.shape[0] == conv_state.shape[0]
    assert x.shape[2] == conv_state.shape[2]
    assert x.shape[1] == 1
    new_conv_state = conv_state.clone()
    new_conv_state = torch.roll(new_conv_state, shifts=-1, dims=1)
    new_conv_state[:, -1:, :] = x
    y = torch.sum(new_conv_state * conv1d_weight, dim=1, keepdim=True)
    if conv1d_bias is not None:
        y += conv1d_bias
    return y, new_conv_state


class CausalConv1d(nn.Module):
    """Causal depthwise 1D convolution with left-padding for sequence modeling.

    Args:
        feature_dim: Number of input and output channels.
        kernel_size: Size of the 1D convolution kernel. Default: 4.
        bias: Whether to add a learnable bias. Default: True.
        channel_mixing: If True, uses groups=1; if False, uses groups=feature_dim (depthwise). Default: False.
    """

    def __init__(
        self,
        feature_dim: int,
        kernel_size: int = 4,
        bias: bool = True,
        channel_mixing: bool = False,
    ):
        """Initializes CausalConv1d with depthwise 1D convolution."""
        super().__init__()
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.groups = 1 if channel_mixing else feature_dim

        if kernel_size > 0:
            self.pad = kernel_size - 1
            self.conv = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=bias,
            )
        else:
            self.conv = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets convolution parameters."""
        if self.conv is not None:
            self.conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        return_last_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of causal 1D depthwise convolution.

        Args:
            x: Input tensor of shape (B, T, D).
            conv_state: Optional left context tensor of shape (B, KS-1, D).
            return_last_state: If True, returns (output, next_conv_state).

        Returns:
            Output tensor of shape (B, T, D), or (output, next_conv_state).
        """
        if self.conv is None or self.kernel_size == 0:
            return (x, None) if return_last_state else x

        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=1)

        y = x.transpose(2, 1)  # (B, D, T)
        y = self.conv(y)
        if conv_state is not None:
            y = y[:, :, conv_state.shape[1] :]

        if self.pad > 0:
            out = y[:, :, : -self.pad].transpose(2, 1)
            last_state = x[:, -self.pad :]
        else:
            out = y.transpose(2, 1)
            last_state = torch.empty(x.shape[0], 0, x.shape[2], device=x.device, dtype=x.dtype)

        if return_last_state:
            return out, last_state
        else:
            return out

    def step(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step causal convolution for autoregressive generation.

        Args:
            x: Input token tensor of shape (B, 1, D).
            conv_state: Previous state buffer of shape (B, KS, D).

        Returns:
            Tuple of (output token tensor of shape (B, 1, D), updated conv_state).
        """
        if self.conv is None:
            return x, None
        B, S, D = x.shape
        assert S == 1
        if conv_state is None:
            conv_state = torch.zeros(B, self.kernel_size, D, device=x.device, dtype=x.dtype)
        # Weight shape in depthwise Conv1d: (D, 1, KS) -> transpose to (KS, D)
        w = self.conv.weight.squeeze(1).transpose(0, 1)
        b = self.conv.bias
        return conv1d_step(x, conv_state, w, b)
