"""State utilities: recursive detach and zero-row for nested state structures."""

from __future__ import annotations

import torch
from typing import Dict, List, Tuple, Union


def detach_states(
    states: Union[Dict, List, Tuple, object]
) -> Union[Dict, List, Tuple, object]:
    """Recursively detach every tensor in a nested state structure.

    Walks dicts, lists, tuples, and state dataclasses (anything with a
    ``.detach()`` method).  Keeps the exact same shape — dicts stay dicts,
    tuples stay tuples, etc.

    Usage::

        states = detach_states(states)     # after loss.backward(), before next batch
    """
    if isinstance(states, dict):
        return {k: detach_states(v) for k, v in states.items()}
    if isinstance(states, (list, tuple)):
        return type(states)(detach_states(v) for v in states)
    if hasattr(states, "detach"):
        return states.detach()
    return states


def zero_rows(
    states: Union[Dict, List, Tuple, object],
    mask: torch.Tensor,
) -> None:
    """In-place zero selected batch rows across a nested state structure.

    ``mask`` must be a boolean tensor of shape ``(batch_size,)``.
    For every tensor field inside state dataclasses, ``tensor[:, mask] = 0``
    is applied — this zeroes the masked rows while preserving the leading
    dimension (num_directions) and all other dimensions.

    Usage::

        zero_rows(states, mask)            # mask = torch.tensor([True, False, False])
    """
    if isinstance(states, dict):
        for v in states.values():
            zero_rows(v, mask)
    elif isinstance(states, (list, tuple)):
        for v in states:
            zero_rows(v, mask)
    elif hasattr(states, "__dataclass_fields__"):
        for fname in states.__dataclass_fields__:
            tensor = getattr(states, fname)
            tensor[:, mask] = 0
