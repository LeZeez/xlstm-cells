"""
sLSTM: Scalar-memory LSTM cell with block-diagonal (per-head) recurrence.

Optimized:
1. Input projections fused into a single F.linear call (4->1 GEMM per layer).
2. Recurrent weights stored as a single fused parameter (no torch.cat on forward).
3. ``fast_mode=True`` compiles the sequential scan (recurrence loop) chunk-by-chunk
   with ``torch.compile(dynamic=False)``.  Compile time is O(fast_chunk_size),
   not O(sequence length).
4. nn.LSTM-compatible: multi-layer, bidirectional, dropout, batch_first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
import torch.nn as nn
import torch.nn.functional as F

from ._utils import (
    PackedBoundariesMode,
    get_packed_boundaries_override_mode,
)
from .mlstm import _MAX_FORGET_BIAS

_EPS = 1e-6
_BOUNDARY_RESET_LOGF = -1000.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class sLSTMState:
    """State for sLSTM, held as named tensors for full user control.

    Shapes (single direction, single layer):
        c  (B, H, Dh)   cell state
        n  (B, H, Dh)   normalizer
        m  (B, H, Dh)   log-space stabilizer
        h  (B, H, Dh)   previous hidden output (required for recurrence)
    """

    c: torch.Tensor
    n: torch.Tensor
    m: torch.Tensor
    h: torch.Tensor

    @classmethod
    def init(cls, batch_size: int, num_heads: int, head_dim: int,
             device=None, dtype=None) -> "sLSTMState":
        H, Dh = num_heads, head_dim
        return cls(
            c=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            n=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            m=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            h=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
        )

    def detach(self) -> "sLSTMState":
        return sLSTMState(self.c.detach(), self.n.detach(),
                          self.m.detach(), self.h.detach())

    def to(self, *args, **kwargs) -> "sLSTMState":
        return sLSTMState(self.c.to(*args, **kwargs), self.n.to(*args, **kwargs),
                          self.m.to(*args, **kwargs), self.h.to(*args, **kwargs))

    def clone(self) -> "sLSTMState":
        return sLSTMState(self.c.clone(), self.n.clone(),
                          self.m.clone(), self.h.clone())

    def __repr__(self) -> str:
        return (f"sLSTMState(c={list(self.c.shape)}, n={list(self.n.shape)}, "
                f"m={list(self.m.shape)}, h={list(self.h.shape)})")


# ---------------------------------------------------------------------------
# Fused sequential scan -- compilation target and eager fallback
# ---------------------------------------------------------------------------

def _slstm_scan_sequential(
    all_in: torch.Tensor,
    R_fused: torch.Tensor,
    c: torch.Tensor,
    n: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
    boundaries: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Run sLSTM recurrence over the full sequence in a single for-loop.

    Uses a single fused einsum per timestep (all four gates at once).
    Designed as a compilation target so TorchInductor can unroll the
    ``for t in range(T)`` loop and fuse all operations into a handful of
    optimised kernels.

    Args:
        all_in:  (B, T, H, 4*Dh)   fused z_in,i_in,f_in,o_in
        R_fused: (H, Dh, 4*Dh)     fused Rz,Ri,Rf,Ro
        c:       (B, H, Dh)          cell state before sequence
        n:       (B, H, Dh)          normalizer before sequence
        m:       (B, H, Dh)          stabiliser before sequence
        h:       (B, H, Dh)          hidden state before sequence
        boundaries:    optional (B, T) bool -- True at the FIRST position
                       of every packed document.  At those positions the
                       raw forget gate ``f_tilde`` is overridden to
                       _BOUNDARY_RESET_LOGF (-1000) so its log_sigmoid
                       contributes a near-zero cumulative forgetting
                       factor to the recurrence, effectively resetting
                       the (c, n, m) state from that point onward.

    Returns:
        outputs:  (B, T, hidden_size)
        c, n, m, h: final states after full sequence

    .. note::
        The boundary reset overrides ``f_tilde`` with a large negative
        constant (_BOUNDARY_RESET_LOGF = -1000).  This yields
        ``f' = exp(logsig(-1000) + m_prev - m_new) ≈ 0`` unconditionally
        for any realistic ``m_prev``, since ``m_new`` is driven by
        ``i_tilde`` (the ``m_new = max(...)`` formula picks max of
        m + logsig(f) and i_tilde).  At boundary, m_new ≈ i_tilde,
        f_prime ≈ 0; carry into c, n is effectively killed.  The reset
        holds for any m_prev < i_tilde + 1000, which covers all
        realistic scenarios.
    """
    _eps_local = _EPS
    B, T, H, Dh_x4 = all_in.shape
    Dh = Dh_x4 // 4
    hidden_size = H * Dh
    output_list: List[torch.Tensor] = []

    if boundaries is not None:
        b_dev = boundaries.to(device=all_in.device, dtype=torch.bool)
    else:
        b_dev = None

    for t in range(T):
        all_tilde = all_in[:, t] + torch.einsum("bhd,hde->bhe", h, R_fused)
        z_tilde, i_tilde, f_tilde, o_tilde = all_tilde.chunk(4, dim=-1)

        # Boundary reset: override f_tilde at boundary positions.
        # logsigmoid(_BOUNDARY_RESET_LOGF) ≈ _BOUNDARY_RESET_LOGF, so
        # m_new = max(log_f + m, i_tilde) is dominated by i_tilde for
        # any realistic m, and f_prime ≈ exp(-1000) ≈ 0.
        if b_dev is not None:
            b_t = b_dev[:, t].view(B, 1, 1)
            f_tilde = torch.where(b_t, torch.full_like(f_tilde, _BOUNDARY_RESET_LOGF), f_tilde)

        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)
        log_f = F.logsigmoid(f_tilde)

        m_new = torch.maximum(log_f + m, i_tilde)
        i_prime = torch.exp(i_tilde - m_new)
        f_prime = torch.exp(log_f + m - m_new)

        c = f_prime * c + i_prime * z
        n = f_prime * n + i_prime
        h = o * (c / n.clamp_min(_eps_local))
        m = m_new

        output_list.append(h.reshape(B, hidden_size))

    outputs = torch.stack(output_list, dim=1)
    return outputs, c, n, m, h


# ---------------------------------------------------------------------------
# Eager reference -- for environments without torch.compile, for testing
# ---------------------------------------------------------------------------

def _slstm_recurrent_scan(
    z_in: torch.Tensor, i_in: torch.Tensor, f_in: torch.Tensor, o_in: torch.Tensor,
    Rz: torch.Tensor, Ri: torch.Tensor, Rf: torch.Tensor, Ro: torch.Tensor,
    c: torch.Tensor, n: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation for testing only. Not used by any module at runtime.

    Separate-argument signature kept for backward compatibility with tests.
    """
    B, T, H, Dh = z_in.shape
    hidden_size = H * Dh
    output_list: List[torch.Tensor] = []

    R_fused = torch.cat([Rz, Ri, Rf, Ro], dim=-1)
    all_in = torch.cat([z_in, i_in, f_in, o_in], dim=-1)

    for t in range(T):
        all_tilde = all_in[:, t] + torch.einsum("bhd,hde->bhe", h, R_fused)
        z_tilde, i_tilde, f_tilde, o_tilde = all_tilde.chunk(4, dim=-1)
        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)
        log_f = F.logsigmoid(f_tilde)
        m_new = torch.maximum(log_f + m, i_tilde)
        i_prime = torch.exp(i_tilde - m_new)
        f_prime = torch.exp(log_f + m - m_new)
        c = f_prime * c + i_prime * z
        n = f_prime * n + i_prime
        h = o * (c / n.clamp_min(_EPS))
        m = m_new
        output_list.append(h.reshape(B, hidden_size))

    outputs = torch.stack(output_list, dim=1)
    return outputs, c, n, m, h


# ---------------------------------------------------------------------------
# sLSTMCell -- single step, like nn.LSTMCell
# ---------------------------------------------------------------------------

class sLSTMCell(nn.Module):
    """One time-step of sLSTM.  Analogous to nn.LSTMCell.

    Recurrent weights are block-diagonal per head -- each head only sees its
    own previous hidden slice, matching the paper's design.

    Weights are stored as fused parameters to avoid torch.cat on every step.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4,
                 bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        Dh = self.head_dim

        # Fused input projection: [Wz; Wi; Wf; Wo] as one linear
        self.W_all = nn.Linear(input_size, 4 * hidden_size, bias=bias)

        # Fused recurrent weights: [Rz | Ri | Rf | Ro] concatenated on last dim
        self.R_fused = nn.Parameter(torch.empty(num_heads, Dh, 4 * Dh))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        Dh = self.head_dim
        HS = self.hidden_size
        # Initialize each gate's slice of the fused weight
        # W_all.weight is (4*HS, input_size), rows: [Wz | Wi | Wf | Wo]
        w = self.W_all.weight.data
        nn.init.normal_(w[:HS], std=std)          # Wz
        nn.init.normal_(w[HS:2*HS], std=std)      # Wi
        nn.init.normal_(w[2*HS:3*HS], std=1e-2)   # Wf
        nn.init.xavier_normal_(w[3*HS:4*HS])      # Wo
        if self.W_all.bias is not None:
            nn.init.zeros_(self.W_all.bias)
            # Forget gate is the 3rd chunk [Wz | Wi | Wf | Wo]
            self.W_all.bias.data[2*HS:3*HS].fill_(3.0)
        # R_fused is (H, Dh, 4*Dh), columns: [Rz | Ri | Rf | Ro]
        # orthogonal_ needs contiguous memory, so init into temp and copy back
        R = self.R_fused.data
        for h in range(self.num_heads):
            for g in range(4):
                tmp = torch.empty_like(R[h, :, g*Dh:(g+1)*Dh])
                nn.init.orthogonal_(tmp)
                R[h, :, g*Dh:(g+1)*Dh] = tmp

    def init_state(self, batch_size: int, device=None, dtype=None) -> sLSTMState:
        return sLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def forward(self, x_t: torch.Tensor, state: sLSTMState) -> Tuple[torch.Tensor, sLSTMState]:
        B = x_t.size(0)
        H, Dh = self.num_heads, self.head_dim

        wx = self.W_all(x_t).view(B, 4, H, Dh).permute(0, 2, 1, 3).reshape(B, H, 4 * Dh)
        all_tilde = wx + torch.einsum("bhd,hde->bhe", state.h, self.R_fused)
        z_tilde, i_tilde, f_tilde, o_tilde = all_tilde.chunk(4, dim=-1)

        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)
        log_f = F.logsigmoid(f_tilde)

        m_new = torch.maximum(log_f + state.m, i_tilde)
        i_prime = torch.exp(i_tilde - m_new)
        f_prime = torch.exp(log_f + state.m - m_new)

        c_new = f_prime * state.c + i_prime * z
        n_new = f_prime * state.n + i_prime
        h_new = o * (c_new / n_new.clamp_min(_EPS))

        return h_new.reshape(B, self.hidden_size), sLSTMState(c_new, n_new, m_new, h_new)

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamp the forget-gate bias to [-max_val, max_val].

        Call after ``optimizer.step()`` to prevent the forget bias from
        drifting into saturation (logsigmoid(b_f) ≈ 0), which causes
        the log-normalizer m to grow unboundedly and can make the
        boundary reset ineffective.
        """
        HS = self.hidden_size
        if self.W_all.bias is not None:
            self.W_all.bias.data[2*HS:3*HS].clamp_(-max_val, max_val)


# ---------------------------------------------------------------------------
# sLSTM -- full sequence, multi-layer, bidirectional  (like nn.LSTM)
# ---------------------------------------------------------------------------

class sLSTM(nn.Module):
    """Multi-layer sLSTM with bidirectional support.  Interface mirrors nn.LSTM.

    Args:
        input_size:    feature dimension of input
        hidden_size:   feature dimension of hidden state
        num_layers:    stacked sLSTM layers (default 1)
        num_heads:     heads per layer (int or list of len num_layers)
        bidirectional: if True, bidirectional (default False)
        dropout:       inter-layer dropout (default 0)
        bias:          use bias in linear layers (default True)
        batch_first:   (batch, seq, feature) if True (default True)
        pack_state:    if True, states are always packed in a tuple (default True)
        use_checkpoint:  if True, use activation checkpointing (default False)
        fast_mode:     compile the per-layer scan chunk-by-chunk with
                       ``torch.compile(dynamic=False)``.  Compile time is
                       O(fast_chunk_size), not O(sequence length).
                       Works in both train and eval.  False by default.
        fast_chunk_size: number of timesteps unrolled per compiled graph when
                       ``fast_mode=True`` (default 32).  Larger chunks fuse
                       more aggressively but cost more to compile once; smaller
                       chunks compile faster.  Only matters when ``fast_mode=True``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: Union[int, List[int]] = 4,
        bidirectional: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        pack_state: bool = True,
        use_checkpoint: bool = False,
        fast_mode: bool = False,
        fast_chunk_size: int = 32,
    ):
        super().__init__()

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers
        assert len(num_heads) == num_layers

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pack_state = pack_state
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.use_checkpoint = use_checkpoint
        self.fast_mode = fast_mode
        self.fast_chunk_size = fast_chunk_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._compiled_chunk = None

        for d in range(self.num_directions):
            for l in range(num_layers):
                assert hidden_size % num_heads[l] == 0, (
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_heads[{l}] ({num_heads[l]})."
                )
                layer_input = input_size if l == 0 else hidden_size * self.num_directions
                Dh = hidden_size // num_heads[l]
                Hh = num_heads[l]

                # Fused input projection: [Wz; Wi; Wf; Wo]
                setattr(self, f"W_all_{d}_{l}",
                        nn.Linear(layer_input, 4 * hidden_size, bias=bias))

                # Fused recurrent weights: [Rz | Ri | Rf | Ro]
                setattr(self, f"R_fused_{d}_{l}",
                        nn.Parameter(torch.empty(Hh, Dh, 4 * Dh)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        HS = self.hidden_size
        for d in range(self.num_directions):
            for layer_idx in range(self.num_layers):
                Hh = self.num_heads[layer_idx]
                Dh = HS // Hh

                # W_all weight: (4*HS, input_size), rows: [Wz | Wi | Wf | Wo]
                W = getattr(self, f"W_all_{d}_{layer_idx}")
                w = W.weight.data
                nn.init.normal_(w[:HS], std=std)          # Wz
                nn.init.normal_(w[HS:2*HS], std=std)      # Wi
                nn.init.normal_(w[2*HS:3*HS], std=1e-2)   # Wf
                nn.init.xavier_normal_(w[3*HS:4*HS])      # Wo
                if W.bias is not None:
                    nn.init.zeros_(W.bias)
                    # Forget gate is the 3rd chunk [Wz | Wi | Wf | Wo]
                    W.bias.data[2*HS:3*HS].fill_(3.0)

                # R_fused: (H, Dh, 4*Dh), columns: [Rz | Ri | Rf | Ro]
                # orthogonal_ needs contiguous memory, so init into temp and copy
                R = getattr(self, f"R_fused_{d}_{layer_idx}")
                for h in range(Hh):
                    for g in range(4):
                        tmp = torch.empty_like(R.data[h, :, g*Dh:(g+1)*Dh])
                        nn.init.orthogonal_(tmp)
                        R.data[h, :, g*Dh:(g+1)*Dh] = tmp

    def init_state(self, batch_size: int, device=None, dtype=None):
        states = []
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = self.hidden_size // Hh
            D = self.num_directions
            s = sLSTMState(
                c=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                n=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                m=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                h=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
            )
            states.append(s)
        result = tuple(states)
        if not self.pack_state and self.num_layers == 1:
            return result[0]
        return result

    def _project_sequence(self, x: torch.Tensor, d: int, layer_idx: int) -> torch.Tensor:
        """Fused input projection for all 4 gates at once."""
        Hh = self.num_heads[layer_idx]
        Dh = self.hidden_size // Hh
        B, T, _ = x.shape
        W = getattr(self, f"W_all_{d}_{layer_idx}")
        all_out = W(x)
        return all_out.view(B, T, 4, Hh, Dh).permute(0, 1, 3, 2, 4).reshape(B, T, Hh, 4 * Dh)

    def _run_layer(
        self, x: torch.Tensor, d: int, layer_idx: int,
        c: torch.Tensor, n: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ):
        all_in = self._project_sequence(x, d, layer_idx)
        R_fused = getattr(self, f"R_fused_{d}_{layer_idx}")

        if not self.fast_mode:
            return _slstm_scan_sequential(all_in, R_fused, c, n, m, h, boundaries)

        if self._compiled_chunk is None:
            self._compiled_chunk = torch.compile(
                _slstm_scan_sequential, dynamic=False,
            )

        T = all_in.shape[1]
        C = self.fast_chunk_size
        outputs_chunks = []

        # If boundaries are present, slice them per chunk so the inner
        # torch.compile graph sees the same signature.
        if boundaries is not None:
            b_chunks = []
            i = 0
            while i + C <= T:
                b_chunks.append(boundaries[:, i:i + C])
                i += C
            if i < T:
                b_chunks.append(boundaries[:, i:])
        else:
            b_chunks = [None] * ((T + C - 1) // C)

        t = 0
        idx = 0
        while t + C <= T:
            out_c, c, n, m, h = self._compiled_chunk(
                all_in[:, t:t + C], R_fused, c, n, m, h, b_chunks[idx],
            )
            outputs_chunks.append(out_c)
            t += C
            idx += 1

        if t < T:
            out_c, c, n, m, h = _slstm_scan_sequential(
                all_in[:, t:], R_fused, c, n, m, h, b_chunks[idx],
            )
            outputs_chunks.append(out_c)

        outputs = torch.cat(outputs_chunks, dim=1)
        return outputs, c, n, m, h

    def forward(
        self,
        input: torch.Tensor,
        state=None,
        boundaries: Optional[torch.Tensor] = None,
    ):
        """Run sLSTM recurrence over `input`.

        Args:
            input: (B, T, input_size) when batch_first=True.
            state: prior `sLSTMState`(s); `None` zero-initialises.
            boundaries: optional (B, T) bool tensor marking the FIRST
                position of every packed document. At boundary
                positions the raw forget gate ``f_tilde`` is forced to
                _BOUNDARY_RESET_LOGF (-1000), killing the cumulative
                forgetting factor past that position.  See
                ``PACKED_FORGET_RESET_RESULTS.md``.
        """
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)
        if not isinstance(state, tuple):
            state = (state,)

        # When `boundaries` is passed, decide how to interact with
        # activation checkpointing per the global PackedBoundariesMode.
        packed = boundaries is not None
        bounds_mode = get_packed_boundaries_override_mode()
        ckpt_active = bool(self.use_checkpoint and self.training)
        if packed and bounds_mode == PackedBoundariesMode.DISABLE_CKPT_IN_PACKED:
            ckpt_active = False
        # USE_REENTRANT_CKPT no longer applied -- see PACKED_FORGET_RESET_RESULTS.md / mLSTM for rationale.
        ckpt_use_reentrant = False

        layer_input = input
        final_states: List[sLSTMState] = []

        for layer_idx in range(self.num_layers):
            s_l = state[layer_idx]

            if self.num_directions == 1:
                c_dl = s_l.c.squeeze(0)
                n_dl = s_l.n.squeeze(0)
                m_dl = s_l.m.squeeze(0)
                h_dl = s_l.h.squeeze(0)

                if ckpt_active:
                    out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                        self._run_layer, layer_input, 0, layer_idx,
                        c_dl, n_dl, m_dl, h_dl, boundaries,
                        use_reentrant=ckpt_use_reentrant,
                    )
                else:
                    out, c_out, n_out, m_out, h_out = self._run_layer(
                        layer_input, 0, layer_idx, c_dl, n_dl, m_dl, h_dl,
                        boundaries=boundaries,
                    )

                layer_output = out
                final_states.append(sLSTMState(
                    c_out.unsqueeze(0), n_out.unsqueeze(0),
                    m_out.unsqueeze(0), h_out.unsqueeze(0),
                ))
            else:
                dir_outputs: List[torch.Tensor] = []
                c_dirs: List[torch.Tensor] = []
                n_dirs: List[torch.Tensor] = []
                m_dirs: List[torch.Tensor] = []
                h_dirs: List[torch.Tensor] = []

                for d in range(self.num_directions):
                    if d == 1:
                        layer_input = torch.flip(layer_input, [1])

                    # Direction-local boundary: in the reverse direction
                    # (d == 1) the input has been flipped, so position j in
                    # the flipped stream corresponds to original position
                    # (T-1-j). The boundary mask must flip with the input
                    # so a boundary at original position p triggers the
                    # forget-gate override at flipped position (T-1-p)
                    # -- which is where the reverse recurrence is
                    # processing original token p.
                    b_d = boundaries
                    if d == 1 and boundaries is not None:
                        b_d = torch.flip(boundaries, [1])

                    c_dl = s_l.c[d]
                    n_dl = s_l.n[d]
                    m_dl = s_l.m[d]
                    h_dl = s_l.h[d]

                    if ckpt_active:
                        out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                            self._run_layer, layer_input, d, layer_idx,
                            c_dl, n_dl, m_dl, h_dl, b_d,
                            use_reentrant=ckpt_use_reentrant,
                        )
                    else:
                        out, c_out, n_out, m_out, h_out = self._run_layer(
                            layer_input, d, layer_idx, c_dl, n_dl, m_dl, h_dl,
                            boundaries=b_d,
                        )

                    if d == 1:
                        out = torch.flip(out, [1])

                    dir_outputs.append(out)
                    c_dirs.append(c_out)
                    n_dirs.append(n_out)
                    m_dirs.append(m_out)
                    h_dirs.append(h_out)

                layer_output = torch.cat(dir_outputs, dim=-1)
                final_states.append(sLSTMState(
                    torch.stack(c_dirs, dim=0), torch.stack(n_dirs, dim=0),
                    torch.stack(m_dirs, dim=0), torch.stack(h_dirs, dim=0),
                ))

            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            layer_input = layer_output

        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)

        result = tuple(final_states)
        if not self.pack_state and self.num_layers == 1:
            return layer_output, result[0]
        return layer_output, result

    def flatten_parameters(self) -> None:
        pass

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamp the forget-gate bias to [-max_val, max_val] across all layers.

        Call after ``optimizer.step()`` to prevent the forget bias from
        drifting into saturation (logsigmoid(b_f) ≈ 0), which causes
        the log-normalizer m to grow unboundedly and can make the
        boundary reset ineffective.
        """
        HS = self.hidden_size
        for d in range(self.num_directions):
            for layer_idx in range(self.num_layers):
                W = getattr(self, f"W_all_{d}_{layer_idx}")
                if W.bias is not None:
                    W.bias.data[2*HS:3*HS].clamp_(-max_val, max_val)

    def __repr__(self) -> str:
        return (
            f"sLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}, "
            f"batch_first={self.batch_first}, use_checkpoint={self.use_checkpoint}, "
            f"fast_mode={self.fast_mode}, fast_chunk_size={self.fast_chunk_size})"
        )
