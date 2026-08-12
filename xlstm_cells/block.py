"""
xLSTM Blocks -- paper-compliant residual blocks with GroupNorm, causal conv, gating.

These are the fundamental building blocks, not customizable wrappers.
They wrap the bare mLSTM/sLSTM cells with the full paper architecture:

    mLSTMBlock (pre up-projection, Figure 11):
        LN -> up-project -> Conv1d(causal) -> mLSTM -> GroupNorm -> gate -> down-project + residual

    sLSTMBlock (post up-projection, Figure 10):
        LN -> [Conv1d] -> sLSTM -> GroupNorm -> up-project -> gated MLP -> down-project + residual
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mlstm import _MAX_FORGET_BIAS, _MLSTM_CHUNK_SIZE, mLSTM, mLSTMState
from .slstm import sLSTM, sLSTMState

_GN_EPS = 1e-5


def _group_norm_bhwc(
    x: torch.Tensor, num_groups: int, weight: torch.Tensor,
    bias: torch.Tensor, eps: float = _GN_EPS,
) -> torch.Tensor:
    """GroupNorm working directly on (B, T, C) layout -- no transposes.

    Equivalent to ``nn.GroupNorm(groups, C)`` applied to (B, C, T) but
    operates via reshape so that the tensor stays contiguous.  This eliminates
    two ``aten::copy_`` calls per block per forward pass.

    Normalisation is computed over (T, D) for each (B, G) where
    D = C // G, matching the standard GroupNorm semantics.
    """
    B, T, C = x.shape
    G = num_groups
    D = C // G

    y = x.reshape(B, T, G, D)

    mean = y.mean(dim=(1, 3), keepdim=True)
    var = y.var(dim=(1, 3), keepdim=True, unbiased=False)
    y = (y - mean) / torch.sqrt(var + eps)

    y = y.reshape(B, T, C)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def _segment_aware_pre_conv(
    h: torch.Tensor,
    boundaries: Optional[torch.Tensor],
    K_m1: int,
) -> torch.Tensor:
    """Zero the K_m1 input positions immediately preceding each True
    in `boundaries`, before a depthwise causal Conv1d runs.  Ensures
    the boundary token's conv receptive field contains no
    cross-document context (its kernel sees ``[0, 0, 0, h[boundary]]``,
    not the prior-document tail).

    For a boundary at position p, the K_m1 positions ``[p-K_m1, p)``
    become zero.  Positions ``[p+1, p+K_m1)`` then see partially-zeroed
    receptive fields that converge to fully-clean by position ``p+K_m1``,
    matching the natural K-step warmup that depthwise causal convolution
    applies to any sequence start.

    Returns `h` unchanged when `boundaries is None`, ``K_m1 <= 0``, or
    ``T == 0``.

    Handles boundaries at/near the sequence end (p > T-K_m1): the
    right-padded boundary mask produces windows for every start i in
    [0, T), so a doc ending at T-1 no longer leaks its tail into the
    conv kernel.  See tests/test_segment_aware_pre_conv.py.
    """
    if boundaries is None or K_m1 <= 0:
        return h
    B, T, _ = h.shape
    if T == 0:
        return h
    b_pad = F.pad(boundaries, (0, K_m1), value=False)
    fresh = b_pad.unfold(1, K_m1 + 1, 1)[:, :, 1:].any(dim=-1)
    return h.masked_fill(fresh.unsqueeze(-1), 0.0)


class mLSTMBlock(nn.Module):
    """Paper-compliant mLSTM block with pre up-projection.

    Architecture (Figure 11 from Beck et al. 2024):
        x -> LayerNorm -+-> gate_proj -> sigmoid -> gate
                        +-> up_proj -> Conv1d(causal) -> Swish -> mLSTM -> GroupNorm
                              -> gate * lstm_out + learnable_skip -> down_proj + x

    Args:
        d_model:        input & output feature dimension
        expand_factor:  up-projection multiplier (paper uses 2)
        num_heads:      number of mLSTM heads (default 16; higher head counts
                        are more memory-efficient: the C state scales as
                        num_heads * head_dim^2, so more heads with smaller
                        head_dim = less total memory)
        conv_kernel:    causal conv1d kernel size (paper uses 4, set 0 to disable)
        dropout:        dropout on output
        bias:           whether linear layers use bias
        use_checkpoint:     activation checkpointing for mLSTM recurrence
        use_triton_kernels: use mlstm_kernels triton backend if available
        chunkwise_kernel:   triton chunkwise kernel (both exp-gate):
                            "xl_chunk" (default), "limit_chunk"
        chunk_size:         chunk size for the chunkwise kernel (default 128)

    .. hint::
        **Triton kernels vs. activation checkpointing**
        The triton backend computes the mLSTM recurrence chunk-wise and keeps
        peak activation memory far below the native chunked-parallel scan.
        ``use_checkpoint=True`` still cuts retained activation memory by
        roughly half at the cost of recomputing the sequence during the
        backward pass.  Prefer checkpointing when VRAM-bound, omit it when
        compute-bound.
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: int = 2,
        num_heads: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        use_checkpoint: bool = False,
        use_triton_kernels: bool = True,
        chunkwise_kernel: str = "xl_chunk",
        chunk_size: int = _MLSTM_CHUNK_SIZE,
        eps: Optional[float] = None,
    ):
        super().__init__()
        expanded = d_model * expand_factor
        assert expanded % num_heads == 0, (
            f"expanded ({expanded}) must be divisible by num_heads ({num_heads}). "
            f"Choose expand_factor such that d_model * expand_factor % num_heads == 0."
        )
        self.expanded = expanded
        self.num_heads = num_heads
        self.conv_kernel = conv_kernel

        self.ln = nn.LayerNorm(d_model)

        self.fused_proj = nn.Linear(d_model, 2 * expanded, bias=bias)

        if conv_kernel > 0:
            self.conv = nn.Conv1d(
                expanded, expanded,
                kernel_size=conv_kernel,
                groups=expanded,         # depthwise: each channel independently
                bias=bias,
            )
        else:
            self.conv = None

        self.lstm = mLSTM(
            expanded, expanded,
            num_layers=1,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            pack_state=False,
            use_checkpoint=use_checkpoint,
            use_triton_kernels=use_triton_kernels,
            chunkwise_kernel=chunkwise_kernel,
            chunk_size=chunk_size,
            eps=eps,
        )

        self.gn = nn.GroupNorm(num_heads, expanded)

        self.learnable_skip = nn.Parameter(torch.zeros(1, 1, expanded))

        self.down_proj = nn.Linear(expanded, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.learnable_skip)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[mLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        """Run the residual block.

        Args:
            x: (B, T, d_model) input.
            state: prior `mLSTMState`; `None` zero-initialises.
            boundaries: optional (B, T) bool tensor marking the FIRST
                position of every packed document. Pass-through to the
                inner ``mLSTM.forward``. See ``mLSTM.forward`` for the
                math.
        """
        residual = x
        x = self.ln(x)

        fused = self.fused_proj(x)                 # (B, T, 2*expanded)
        gate_raw, h = fused.chunk(2, dim=-1)       # each (B, T, expanded)
        gate = torch.sigmoid(gate_raw)

        if self.conv is not None:
            # Segment-aware conv: zero the K-1 input slots right before
            # each packed-document boundary so the boundary token's causal
            # convolution receptive field contains no prior-document context
            # (the recurrent state is also reset by the boundaries kwarg on
            # self.lstm). See ``_segment_aware_pre_conv``.
            h = _segment_aware_pre_conv(h, boundaries, self.conv_kernel - 1)
            h = h.transpose(1, 2)                  # (B, expanded, T) for Conv1d
            h = F.pad(h, (self.conv_kernel - 1, 0))  # causal: pad left only
            h = self.conv(h)                       # (B, expanded, T)
            h = h.transpose(1, 2)                  # back to (B, T, expanded)

        h = F.silu(h)                               # Swish activation
        h = h + self.learnable_skip                 # learnable per-channel bias

        h, state = self.lstm(h, state, boundaries=boundaries)  # mLSTM recurrence

        gn_weight = self.gn.weight
        gn_bias = self.gn.bias
        gn_eps = self.gn.eps
        h = _group_norm_bhwc(h, self.num_heads, gn_weight, gn_bias, gn_eps)

        h = gate * h                                 # apply external output gate

        h = self.down_proj(h)
        h = self.dropout(h)

        return h + residual, state

    def init_state(self, batch_size: int, device=None, dtype=None):
        return self.lstm.init_state(batch_size, device, dtype)

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamp the inner mLSTM forget-gate bias to [-max_val, max_val].

        Call after ``optimizer.step()`` to prevent the forget bias from
        drifting into saturation (logsigmoid(b_f) ≈ 0), which causes
        the log-normalizer m to grow unboundedly and can make the
        boundary reset ineffective.
        """
        self.lstm.clamp_forget_bias(max_val)


class sLSTMBlock(nn.Module):
    """Paper-compliant sLSTM block with post up-projection.

    Architecture (Figure 10 from Beck et al. 2024):
        x -> LayerNorm -> [Conv1d] -> sLSTM -> GroupNorm -> up_proj -> GeLU -> down_proj + x
                                                     +-> gate_proj -> sigmoid -+

    Args:
        d_model:        input & output feature dimension
        expand_factor:  post up-projection multiplier (paper uses 4/3 ~ 1.33)
        num_heads:      number of sLSTM heads
        conv_kernel:    causal conv1d kernel size (paper uses 4, set 0 to disable)
        dropout:        dropout on output
        bias:           whether linear layers use bias
        use_checkpoint:   activation checkpointing for sLSTM recurrence
        fast_mode:        compile sequential scan with torch.compile
        fast_chunk_size:  chunk size for compiled scan (default 32)
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: float = 4.0 / 3.0,
        num_heads: int = 4,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
        use_checkpoint: bool = False,
        fast_mode: bool = False,
        fast_chunk_size: int = 32,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) for GroupNorm."
        )
        expanded = int(d_model * expand_factor)
        self.expanded = expanded
        self.num_heads = num_heads
        self.conv_kernel = conv_kernel

        self.ln = nn.LayerNorm(d_model)

        if conv_kernel > 0:
            self.conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=conv_kernel,
                groups=d_model,          # depthwise
                bias=bias,
            )
        else:
            self.conv = None

        self.lstm = sLSTM(
            d_model, d_model,
            num_layers=1,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            pack_state=False,
            use_checkpoint=use_checkpoint,
            fast_mode=fast_mode,
            fast_chunk_size=fast_chunk_size,
        )

        self.gn = nn.GroupNorm(num_heads, d_model)

        self.fused_proj = nn.Linear(d_model, 2 * expanded, bias=bias)
        self.down_proj = nn.Linear(expanded, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialize learnable parameters of this block."""
        self.ln.reset_parameters()
        if self.conv is not None:
            self.conv.reset_parameters()
        self.lstm.reset_parameters()
        self.gn.reset_parameters()
        self.fused_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[sLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, sLSTMState]:
        """Run the residual block.

        Args:
            x: (B, T, d_model) input.
            state: prior `sLSTMState`; `None` zero-initialises.
            boundaries: optional (B, T) bool tensor marking the FIRST
                position of every packed document. Pass-through to the
                inner ``sLSTM.forward``. See ``sLSTM.forward`` for the
                math.
        """
        residual = x
        x = self.ln(x)

        if self.conv is not None:
            # Segment-aware conv: see ``_segment_aware_pre_conv`` in
            # mLSTMBlock. Same logic, applied to sLSTM's additive-conv
            # stack so the boundary token's receptive field is clean.
            x = _segment_aware_pre_conv(x, boundaries, self.conv_kernel - 1)
            c = x.transpose(1, 2)                  # (B, d_model, T)
            c = F.pad(c, (self.conv_kernel - 1, 0))
            c = self.conv(c)
            c = c.transpose(1, 2)                  # (B, T, d_model)
            x = x + F.silu(c)                       # additive conv with Swish

        x, state = self.lstm(x, state, boundaries=boundaries)  # sLSTM recurrence

        gn_weight = self.gn.weight
        gn_bias = self.gn.bias
        gn_eps = self.gn.eps
        x = _group_norm_bhwc(x, self.num_heads, gn_weight, gn_bias, gn_eps)

        fused = self.fused_proj(x)                  # (B, T, 2*expanded)
        gate_raw, up_raw = fused.chunk(2, dim=-1)   # each (B, T, expanded)
        gate = torch.sigmoid(gate_raw)
        up = F.gelu(up_raw)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)

        return x + residual, state

    def init_state(self, batch_size: int, device=None, dtype=None):
        return self.lstm.init_state(batch_size, device, dtype)

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamp the inner sLSTM forget-gate bias to [-max_val, max_val].

        Call after ``optimizer.step()`` to prevent the forget bias from
        drifting into saturation (logsigmoid(b_f) ≈ 0), which causes
        the log-normalizer m to grow unboundedly and can make the
        boundary reset ineffective.
        """
        self.lstm.clamp_forget_bias(max_val)
