"""
sLSTM: Scalar-memory LSTM cell with block-diagonal (per-head) recurrence.

Optimized:
1. Input projections fused into a single F.linear call (4->1 GEMM per layer).
2. Recurrent weights fused into a single einsum (4->1 per step).
3. ``fast_mode=True`` compiles the full per-layer computation (projection +
   sequential scan) with ``torch.compile(dynamic=True)``.  This gives
   inductor-level kernel fusion across the entire sequence while only
   incurring one compilation per shape -- not per chunk.
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

_EPS = 1e-6


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

    Returns:
        outputs:  (B, T, hidden_size)
        c, n, m, h: final states after full sequence
    """
    _eps_local = _EPS
    B, T, H, Dh_x4 = all_in.shape
    Dh = Dh_x4 // 4
    hidden_size = H * Dh
    outputs = all_in.new_empty(B, T, hidden_size)

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
        h = o * (c / n.clamp_min(_eps_local))
        m = m_new

        outputs[:, t] = h.reshape(B, hidden_size)

    return outputs, c, n, m, h


# ---------------------------------------------------------------------------
# Eager fallback -- for environments without torch.compile, for debugging
# ---------------------------------------------------------------------------

def _slstm_recurrent_scan(
    z_in: torch.Tensor, i_in: torch.Tensor, f_in: torch.Tensor, o_in: torch.Tensor,
    Rz: torch.Tensor, Ri: torch.Tensor, Rf: torch.Tensor, Ro: torch.Tensor,
    c: torch.Tensor, n: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager fallback: sLSTM recurrence over pre-computed projections.

    Internally fuses R matrices into a single einsum per step.
    Kept for environments without torch.compile and for debugging.
    The 12-arg signature is preserved for backward compatibility.
    """
    B, T, H, Dh = z_in.shape
    hidden_size = H * Dh
    outputs = z_in.new_empty(B, T, hidden_size)

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
        outputs[:, t] = h.reshape(B, hidden_size)

    return outputs, c, n, m, h


# ---------------------------------------------------------------------------
# sLSTMCell -- single step, like nn.LSTMCell
# ---------------------------------------------------------------------------

class sLSTMCell(nn.Module):
    """One time-step of sLSTM.  Analogous to nn.LSTMCell.

    Recurrent weights are block-diagonal per head -- each head only sees its
    own previous hidden slice, matching the paper's design.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        Dh = self.head_dim

        self.Wz = nn.Linear(input_size, hidden_size, bias=False)
        self.Wi = nn.Linear(input_size, hidden_size, bias=False)
        self.Wf = nn.Linear(input_size, hidden_size, bias=False)
        self.Wo = nn.Linear(input_size, hidden_size, bias=False)

        self.Rz = nn.Parameter(torch.empty(num_heads, Dh, Dh))
        self.Ri = nn.Parameter(torch.empty(num_heads, Dh, Dh))
        self.Rf = nn.Parameter(torch.empty(num_heads, Dh, Dh))
        self.Ro = nn.Parameter(torch.empty(num_heads, Dh, Dh))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.normal_(self.Wz.weight, std=std)
        nn.init.normal_(self.Wi.weight, std=std)
        nn.init.normal_(self.Wf.weight, std=1e-2)
        nn.init.xavier_normal_(self.Wo.weight)
        for R in (self.Rz, self.Ri, self.Rf, self.Ro):
            nn.init.orthogonal_(R)

    def init_state(self, batch_size: int, device=None, dtype=None) -> sLSTMState:
        return sLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def forward(self, x_t: torch.Tensor, state: sLSTMState) -> Tuple[torch.Tensor, sLSTMState]:
        B = x_t.size(0)
        H, Dh = self.num_heads, self.head_dim

        w_all = torch.cat([self.Wz.weight, self.Wi.weight,
                           self.Wf.weight, self.Wo.weight], dim=0)
        wx = F.linear(x_t, w_all, None).view(B, H, 4 * Dh)

        R_fused = torch.cat([self.Rz, self.Ri, self.Rf, self.Ro], dim=-1)
        all_tilde = wx + torch.einsum("bhd,hde->bhe", state.h, R_fused)
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
        fast_mode:     compile the per-layer scan with ``torch.compile(dynamic=True)``.
                       One compilation per shape, then zero dispatch overhead.
                       Works in both train and eval.  False by default.
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._compiled_run = None

        for d in range(self.num_directions):
            for l in range(num_layers):
                assert hidden_size % num_heads[l] == 0, (
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_heads[{l}] ({num_heads[l]})."
                )
                layer_input = input_size if l == 0 else hidden_size * self.num_directions
                Dh = hidden_size // num_heads[l]

                setattr(self, f"Wz_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wi_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wf_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wo_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))

                setattr(self, f"Rz_{d}_{l}", nn.Parameter(torch.empty(num_heads[l], Dh, Dh)))
                setattr(self, f"Ri_{d}_{l}", nn.Parameter(torch.empty(num_heads[l], Dh, Dh)))
                setattr(self, f"Rf_{d}_{l}", nn.Parameter(torch.empty(num_heads[l], Dh, Dh)))
                setattr(self, f"Ro_{d}_{l}", nn.Parameter(torch.empty(num_heads[l], Dh, Dh)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for d in range(self.num_directions):
            for l in range(self.num_layers):
                nn.init.normal_(getattr(self, f"Wz_{d}_{l}").weight, std=std)
                nn.init.normal_(getattr(self, f"Wi_{d}_{l}").weight, std=std)
                nn.init.normal_(getattr(self, f"Wf_{d}_{l}").weight, std=1e-2)
                nn.init.xavier_normal_(getattr(self, f"Wo_{d}_{l}").weight)
                for gate in ("Wz", "Wi", "Wf", "Wo"):
                    lin = getattr(self, f"{gate}_{d}_{l}")
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)
                for gate in ("Rz", "Ri", "Rf", "Ro"):
                    nn.init.orthogonal_(getattr(self, f"{gate}_{d}_{l}"))

    def init_state(self, batch_size: int, device=None, dtype=None):
        states = []
        for l in range(self.num_layers):
            Hh = self.num_heads[l]; Dh = self.hidden_size // Hh; D = self.num_directions
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

    def _fuse_w_weights(self, d: int, l: int):
        Wz = getattr(self, f"Wz_{d}_{l}"); Wi = getattr(self, f"Wi_{d}_{l}")
        Wf = getattr(self, f"Wf_{d}_{l}"); Wo = getattr(self, f"Wo_{d}_{l}")
        w_all = torch.cat([Wz.weight, Wi.weight, Wf.weight, Wo.weight], dim=0)
        if Wz.bias is not None:
            b_all = torch.cat([Wz.bias, Wi.bias, Wf.bias, Wo.bias], dim=0)
        else:
            b_all = None
        return w_all, b_all

    def _fuse_r_weights(self, d: int, l: int):
        Rz = getattr(self, f"Rz_{d}_{l}"); Ri = getattr(self, f"Ri_{d}_{l}")
        Rf = getattr(self, f"Rf_{d}_{l}"); Ro = getattr(self, f"Ro_{d}_{l}")
        return torch.cat([Rz, Ri, Rf, Ro], dim=-1)

    def _project_sequence(self, x: torch.Tensor, d: int, l: int) -> torch.Tensor:
        Hh = self.num_heads[l]
        B, T, _ = x.shape
        w_all, b_all = self._fuse_w_weights(d, l)
        all_out = F.linear(x, w_all, b_all)
        return all_out.view(B, T, Hh, -1)

    def _run_layer(
        self, x: torch.Tensor, d: int, l: int,
        c: torch.Tensor, n: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
    ):
        all_in = self._project_sequence(x, d, l)
        R_fused = self._fuse_r_weights(d, l)

        if self.fast_mode:
            if self._compiled_run is None:
                self._compiled_run = torch.compile(
                    _slstm_scan_sequential, dynamic=True,
                )
            return self._compiled_run(all_in, R_fused, c, n, m, h)
        else:
            return _slstm_scan_sequential(all_in, R_fused, c, n, m, h)

    def forward(self, input: torch.Tensor, state=None):
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)
        if not isinstance(state, tuple):
            state = (state,)

        layer_input = input
        final_states: List[sLSTMState] = []

        for l in range(self.num_layers):
            s_l = state[l]

            if self.num_directions == 1:
                c_dl = s_l.c.squeeze(0); n_dl = s_l.n.squeeze(0)
                m_dl = s_l.m.squeeze(0); h_dl = s_l.h.squeeze(0)

                if self.use_checkpoint and self.training:
                    out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                        self._run_layer, layer_input, 0, l,
                        c_dl, n_dl, m_dl, h_dl, use_reentrant=False,
                    )
                else:
                    out, c_out, n_out, m_out, h_out = self._run_layer(
                        layer_input, 0, l, c_dl, n_dl, m_dl, h_dl,
                    )

                layer_output = out
                final_states.append(sLSTMState(
                    c_out.unsqueeze(0), n_out.unsqueeze(0),
                    m_out.unsqueeze(0), h_out.unsqueeze(0),
                ))
            else:
                dir_outputs: List[torch.Tensor] = []
                c_dirs: List[torch.Tensor] = []; n_dirs: List[torch.Tensor] = []
                m_dirs: List[torch.Tensor] = []; h_dirs: List[torch.Tensor] = []

                for d in range(self.num_directions):
                    if d == 1:
                        layer_input = torch.flip(layer_input, [1])

                    c_dl = s_l.c[d]; n_dl = s_l.n[d]
                    m_dl = s_l.m[d]; h_dl = s_l.h[d]

                    if self.use_checkpoint and self.training:
                        out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                            self._run_layer, layer_input, d, l,
                            c_dl, n_dl, m_dl, h_dl, use_reentrant=False,
                        )
                    else:
                        out, c_out, n_out, m_out, h_out = self._run_layer(
                            layer_input, d, l, c_dl, n_dl, m_dl, h_dl,
                        )

                    if d == 1:
                        out = torch.flip(out, [1])

                    dir_outputs.append(out)
                    c_dirs.append(c_out); n_dirs.append(n_out)
                    m_dirs.append(m_out); h_dirs.append(h_out)

                layer_output = torch.cat(dir_outputs, dim=-1)
                final_states.append(sLSTMState(
                    torch.stack(c_dirs, dim=0), torch.stack(n_dirs, dim=0),
                    torch.stack(m_dirs, dim=0), torch.stack(h_dirs, dim=0),
                ))

            if l < self.num_layers - 1:
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

    def __repr__(self) -> str:
        return (
            f"sLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}, "
            f"batch_first={self.batch_first}, use_checkpoint={self.use_checkpoint}, "
            f"fast_mode={self.fast_mode})"
        )
