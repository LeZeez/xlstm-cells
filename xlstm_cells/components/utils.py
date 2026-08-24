# Copyright (c) NXAI GmbH and its affiliates 2024
"""Utility functions for xLSTM components, capping, and dimension rounding."""

from __future__ import annotations

from typing import Optional
import torch


def soft_cap(values: torch.Tensor, cap_value: Optional[float] = None) -> torch.Tensor:
    """Soft caps a tensor using tanh scaling: cap_value * tanh(values / cap_value).

    Args:
        values: The input tensor to soft-cap.
        cap_value: The maximum magnitude cap. If None, returns values unchanged.

    Returns:
        The soft-capped tensor.
    """
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)


def round_up_to_next_multiple_of(x: int | float, multiple_of: int) -> int:
    """Rounds up x to the next integer multiple of multiple_of.

    Args:
        x: The value to round up.
        multiple_of: The positive integer multiple.

    Returns:
        Integer rounded up to the next multiple of multiple_of.
    """
    x_int = int(round(x)) if isinstance(x, float) else int(x)
    return int(((x_int + multiple_of - 1) // multiple_of) * multiple_of)
