# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
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
    """Causal depthwise 1D convolution with left-padding.
    Input:  (B, T, D)
    Output: (B, T, D)
    """

    def __init__(
        self,
        feature_dim: int,
        kernel_size: int = 4,
        bias: bool = True,
        channel_mixing: bool = False,
    ):
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

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        return_last_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.conv is None:
            return (x, None) if return_last_state else x

        B, T, D = x.shape
        x_in = x.transpose(1, 2)  # (B, D, T)
        out = self.conv(x_in)
        out = out[:, :, :T].transpose(1, 2)  # (B, T, D)

        if return_last_state:
            # Extract last KS tokens for stateful continuation
            last_k = min(self.kernel_size, T)
            new_conv_state = x[:, -last_k:, :]
            if last_k < self.kernel_size:
                pad_needed = self.kernel_size - last_k
                new_conv_state = F.pad(new_conv_state, (0, 0, pad_needed, 0))
            return out, new_conv_state
        return out

    def step(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
