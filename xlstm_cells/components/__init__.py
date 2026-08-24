# Copyright (c) NXAI GmbH and its affiliates 2024
"""Component modules for xLSTM blocks: layers, initializations, and normalizations."""

from __future__ import annotations

from .init import bias_linspace_init_, small_init_init_, wang_init_
from .ln import LayerNorm, MultiHeadLayerNorm, RMSNorm, MultiHeadRMSNorm
from .conv import CausalConv1d, conv1d_step
from .linear_headwise import LinearHeadwiseExpand
from .feedforward import GatedFeedForward, SwiGLUFeedForward
from .utils import soft_cap, round_up_to_next_multiple_of

__all__ = [
    "bias_linspace_init_",
    "small_init_init_",
    "wang_init_",
    "LayerNorm",
    "MultiHeadLayerNorm",
    "RMSNorm",
    "MultiHeadRMSNorm",
    "CausalConv1d",
    "conv1d_step",
    "LinearHeadwiseExpand",
    "GatedFeedForward",
    "SwiGLUFeedForward",
    "soft_cap",
    "round_up_to_next_multiple_of",
]
