"""State utilities: recursive detach, zero-row, packed-boundaries mode toggle."""

from __future__ import annotations

import torch
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# PackedBoundariesMode -- how the (mLSTM/sLSTM).forward boundaries= kwarg
# interacts with activation checkpointing during packed-document sequences.
# ---------------------------------------------------------------------------

class PackedBoundariesMode(Enum):
    """How `boundaries=...` interacts with `use_checkpoint=True`.

    USE_REENTRANT_CKPT (default):
        When the user passes `boundaries=...` and `use_checkpoint=True`, the
        inner `_torch_checkpoint` call is switched to `use_reentrant=True`
        for the duration of that single forward. This avoids the
        saved-tensor count check at torch.utils.checkpoint:873 that
        triggers when the wrapper adds an extra saved tensor via the
        autograd graph (which the boundaries override does). Strictly
        more memory-efficient than `use_reentrant=False`.

    DISABLE_CKPT_IN_PACKED:
        Fallback path. When the user passes `boundaries=...`,
        `use_checkpoint` is silenced (set to False) for the duration of
        that forward call. The chunkwise kernel recomputes activations
        in the standard way, no override behaviour. Slower per backward
        (full activations kept), but works on any kernel / GPU.

    Default is USE_REENTRANT_CKPT. Set globally via
    `set_packed_boundaries_override_mode(...)`.
    """
    USE_REENTRANT_CKPT = "reentrant"
    DISABLE_CKPT_IN_PACKED = "disable"


_GLOBAL_BOUNDS_MODE: PackedBoundariesMode = PackedBoundariesMode.USE_REENTRANT_CKPT


def get_packed_boundaries_override_mode() -> PackedBoundariesMode:
    return _GLOBAL_BOUNDS_MODE


def set_packed_boundaries_override_mode(
    mode: PackedBoundariesMode,
) -> PackedBoundariesMode:
    """Set the global default for how `boundaries=...` interacts with
    activation checkpointing. Returns the previous mode.
    """
    global _GLOBAL_BOUNDS_MODE
    prev = _GLOBAL_BOUNDS_MODE
    _GLOBAL_BOUNDS_MODE = mode
    return prev


def detach_states(
    states: Union[Dict, List, Tuple, Any]
) -> Union[Dict, List, Tuple, Any]:
    """Recursively detach every tensor in a nested state structure.

    Walks dicts, lists, tuples, and state dataclasses (anything with a
    ``.detach()`` method).  Keeps the exact same shape --- dicts stay dicts,
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
    states: Union[Dict, List, Tuple, Any],
    mask: torch.Tensor,
) -> None:
    """In-place zero selected batch rows across a nested state structure.

    ``mask`` must be a boolean tensor of shape ``(batch_size,)``.
    For every tensor field inside state dataclasses, ``tensor[:, mask] = 0``
    is applied --- this zeroes the masked rows while preserving the leading
    dimension (num_directions) and all other dimensions.

    .. warning::
        Only call on detached states.  If any tensor in the state is still
        attached to the autograd graph, in-place modification will raise an
        error (or silently corrupt gradients on older PyTorch versions).

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
            if tensor.requires_grad:
                raise RuntimeError(
                    f"zero_rows: tensor '{fname}' requires grad. "
                    f"Detach states first with detach_states()."
                )
            tensor[:, mask] = 0
