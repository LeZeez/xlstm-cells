"""State utilities: recursive detach, zero-row, packed-boundaries mode toggle."""

from __future__ import annotations

import torch
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# PackedBoundariesMode -- legacy knob for how (mLSTM/sLSTM).forward
# boundaries= kwarg interacted with activation checkpointing under packed-
# document sequences.  USE_REENTRANT_CKPT is now a no-op: the boundaries
# override is compatible with use_reentrant=False on modern PyTorch, and the
# non-reentrant path is preferred for detached-input / frozen-embedding
# TBPTT.  See tests/test_packed_non_reentrant.py.
# ---------------------------------------------------------------------------

class PackedBoundariesMode(Enum):
    """How `boundaries=...` interacts with `use_checkpoint=True` (legacy).

    USE_REENTRANT_CKPT (default):
        Maintained for backwards compatibility.  No longer forces a switch
        to `use_reentrant=True` -- the runtime uses the non-reentrant path
        in all cases.

    DISABLE_CKPT_IN_PACKED:
        When the user passes `boundaries=...`, `use_checkpoint` is silenced
        (set to False) for the duration of that forward call. Still honoured.

    Default is USE_REENTRANT_CKPT. Set globally via
    `set_packed_boundaries_override_mode(...)`.
    """
    USE_REENTRANT_CKPT = "reentrant"
    DISABLE_CKPT_IN_PACKED = "disable"


_GLOBAL_BOUNDS_MODE: PackedBoundariesMode = PackedBoundariesMode.USE_REENTRANT_CKPT


def get_packed_boundaries_override_mode() -> PackedBoundariesMode:
    """Returns the current global packed boundaries override mode."""
    return _GLOBAL_BOUNDS_MODE


def set_packed_boundaries_override_mode(
    mode: PackedBoundariesMode,
) -> PackedBoundariesMode:
    """Set the global default for how `boundaries=...` interacts with
    activation checkpointing. Returns the previous mode.

    .. note::
        Setting this to ``USE_REENTRANT_CKPT`` is a no-op as of the
        non-reentrant switch. Only ``DISABLE_CKPT_IN_PACKED`` alters
        runtime behaviour.
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
    batch_dim: Optional[int] = None,
) -> None:
    """In-place zero selected batch rows across a nested state structure.

    ``mask`` must be a boolean tensor of shape ``(batch_size,)``.
    Supports both 3D/4D cell & block states ``(B, ...)`` and 4D/5D multi-layer
    sequence states ``(D, B, ...)``.

    .. warning::
        Only call on detached states.  If any tensor in the state is still
        attached to the autograd graph, in-place modification will raise an
        error (or silently corrupt gradients on older PyTorch versions).

    Usage::

        zero_rows(states, mask)            # mask = torch.tensor([True, False, False])
    """
    if isinstance(states, dict):
        for v in states.values():
            zero_rows(v, mask, batch_dim=batch_dim)
    elif isinstance(states, (list, tuple)):
        for v in states:
            zero_rows(v, mask, batch_dim=batch_dim)
    elif hasattr(states, "__dataclass_fields__"):
        for fname in states.__dataclass_fields__:
            tensor = getattr(states, fname)
            if tensor.requires_grad:
                raise RuntimeError(
                    f"zero_rows: tensor '{fname}' requires grad. "
                    f"Detach states first with detach_states()."
                )
            if batch_dim is not None:
                idx = [slice(None)] * tensor.dim()
                idx[batch_dim] = mask
                tensor[tuple(idx)] = 0
            else:
                dim0_match = (tensor.shape[0] == mask.shape[0])
                dim1_match = (tensor.dim() in (4, 5) and tensor.shape[1] == mask.shape[0])
                if dim0_match and dim1_match:
                    raise ValueError(
                        f"zero_rows: ambiguous batch dimension for tensor '{fname}' with shape {list(tensor.shape)} "
                        f"(both dim 0 and dim 1 match mask length {mask.shape[0]}). Specify batch_dim explicitly."
                    )
                elif dim1_match:
                    tensor[:, mask] = 0
                elif dim0_match:
                    tensor[mask] = 0
                else:
                    raise ValueError(
                        f"zero_rows: tensor '{fname}' shape {list(tensor.shape)} does not match mask length {mask.shape[0]}"
                    )


def normalize_and_validate_num_heads(
    num_heads: Union[int, Sequence[int]],
    num_layers: int,
    hidden_size: int,
    class_name: str,
) -> List[int]:
    """Validates and normalizes num_heads into a list of ints per layer."""
    if isinstance(num_heads, int) and not isinstance(num_heads, bool):
        if num_heads <= 0:
            raise ValueError(f"{class_name}: num_heads must be a strictly positive integer, got {num_heads}")
        heads_list = [num_heads] * num_layers
    elif isinstance(num_heads, (list, tuple)):
        if len(num_heads) != num_layers:
            raise ValueError(
                f"{class_name}: length of num_heads ({len(num_heads)}) must match num_layers ({num_layers})"
            )
        for h in num_heads:
            if not isinstance(h, int) or isinstance(h, bool) or h <= 0:
                raise ValueError(f"{class_name}: every element of num_heads must be a positive integer, got {h}")
        heads_list = list(num_heads)
    else:
        raise TypeError(f"{class_name}: num_heads must be an integer or a sequence of integers, got {type(num_heads)}")

    for idx, h in enumerate(heads_list):
        if hidden_size % h != 0:
            raise ValueError(
                f"{class_name}: hidden_size ({hidden_size}) must be divisible by num_heads at layer {idx} ({h})"
            )
    return heads_list
