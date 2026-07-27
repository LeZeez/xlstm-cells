"""
sLSTM: Scalar-memory LSTM cell with block-diagonal (per-head) recurrence.

Optimized over the reference implementation:
1. All input projections (Wz, Wi, Wf, Wo) are computed once over the full
   sequence *before* the time loop — no redundant Linear calls at each step.
2. The recurrent (R * h_prev) term must stay in the loop (it depends on the
   previous hidden state), but is computed as a batched einsum.
3. Full nn.LSTM-compatible interface: multi-layer, bidirectional, dropout, batch_first.
4. Explicit state objects for TBPTT and single-step control.
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
    def init(
        cls,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device=None,
        dtype=None,
    ) -> "sLSTMState":
        H, Dh = num_heads, head_dim
        return cls(
            c=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            n=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            m=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            h=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
        )

    def detach(self) -> "sLSTMState":
        return sLSTMState(
            self.c.detach(), self.n.detach(), self.m.detach(), self.h.detach()
        )

    def to(self, *args, **kwargs) -> "sLSTMState":
        return sLSTMState(
            self.c.to(*args, **kwargs),
            self.n.to(*args, **kwargs),
            self.m.to(*args, **kwargs),
            self.h.to(*args, **kwargs),
        )

    def clone(self) -> "sLSTMState":
        return sLSTMState(
            self.c.clone(), self.n.clone(), self.m.clone(), self.h.clone()
        )

    def __repr__(self) -> str:
        return (
            f"sLSTMState(c={list(self.c.shape)}, n={list(self.n.shape)}, "
            f"m={list(self.m.shape)}, h={list(self.h.shape)})"
        )


# ---------------------------------------------------------------------------
# Optimised step function (standalone so torch.compile can trace it cleanly)
# ---------------------------------------------------------------------------

def _slstm_recurrent_scan(
    z_in: torch.Tensor,
    i_in: torch.Tensor,
    f_in: torch.Tensor,
    o_in: torch.Tensor,
    Rz: torch.Tensor,
    Ri: torch.Tensor,
    Rf: torch.Tensor,
    Ro: torch.Tensor,
    c: torch.Tensor,
    n: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the sLSTM recurrence over pre-computed input projections.

    Args:
        z_in:  (B, T, H, Dh)   Wz * x    (input contribution to cell proposal)
        i_in:  (B, T, H, Dh)   Wi * x    (input gate input)
        f_in:  (B, T, H, Dh)   Wf * x    (forget gate input)
        o_in:  (B, T, H, Dh)   Wo * x    (output gate input)
        Rz:    (H, Dh, Dh)     recurrent weights
        Ri:    (H, Dh, Dh)
        Rf:    (H, Dh, Dh)
        Ro:    (H, Dh, Dh)
        c:     (B, H, Dh)      initial cell state
        n:     (B, H, Dh)      initial norm
        m:     (B, H, Dh)      initial stabiliser
        h:     (B, H, Dh)      initial hidden state

    Returns:
        outputs: (B, T, Hs)  where Hs = H * Dh
        c: final cell
        n: final norm
        m: final stabiliser
        h: final hidden
    """
    B, T, H, Dh = z_in.shape
    hidden_size = H * Dh

    outputs = z_in.new_empty(B, T, hidden_size)

    for t in range(T):
        h_prev = h

        z_tilde = z_in[:, t] + torch.einsum("bhd,hde->bhe", h_prev, Rz)
        i_tilde = i_in[:, t] + torch.einsum("bhd,hde->bhe", h_prev, Ri)
        f_tilde = f_in[:, t] + torch.einsum("bhd,hde->bhe", h_prev, Rf)
        o_tilde = o_in[:, t] + torch.einsum("bhd,hde->bhe", h_prev, Ro)

        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)
        log_f = F.logsigmoid(f_tilde)

        m_prev = m
        m = torch.maximum(log_f + m_prev, i_tilde)
        i_prime = torch.exp(i_tilde - m)
        f_prime = torch.exp(log_f + m_prev - m)

        c = f_prime * c + i_prime * z
        n = f_prime * n + i_prime
        h = o * (c / n.clamp_min(_EPS))

        outputs[:, t] = h.reshape(B, hidden_size)

    return outputs, c, n, m, h


@torch.compile(dynamic=True)
def _slstm_step_compiled(
    z_t: torch.Tensor,
    i_t: torch.Tensor,
    f_t: torch.Tensor,
    o_t: torch.Tensor,
    h_prev: torch.Tensor,
    Rz: torch.Tensor,
    Ri: torch.Tensor,
    Rf: torch.Tensor,
    Ro: torch.Tensor,
    c: torch.Tensor,
    n: torch.Tensor,
    m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused single timestep of sLSTM (compiled via torch.compile)."""
    z_tilde = z_t + torch.einsum("bhd,hde->bhe", h_prev, Rz)
    i_tilde = i_t + torch.einsum("bhd,hde->bhe", h_prev, Ri)
    f_tilde = f_t + torch.einsum("bhd,hde->bhe", h_prev, Rf)
    o_tilde = o_t + torch.einsum("bhd,hde->bhe", h_prev, Ro)

    z = torch.tanh(z_tilde)
    o = torch.sigmoid(o_tilde)
    log_f = F.logsigmoid(f_tilde)

    m_new = torch.maximum(log_f + m, i_tilde)
    i_prime = torch.exp(i_tilde - m_new)
    f_prime = torch.exp(log_f + m - m_new)

    c_new = f_prime * c + i_prime * z
    n_new = f_prime * n + i_prime
    h_new = o * (c_new / n_new.clamp_min(_EPS))

    return c_new, n_new, m_new, h_new


def _slstm_recurrent_scan_optimized(
    z_in: torch.Tensor,
    i_in: torch.Tensor,
    f_in: torch.Tensor,
    o_in: torch.Tensor,
    Rz: torch.Tensor,
    Ri: torch.Tensor,
    Rf: torch.Tensor,
    Ro: torch.Tensor,
    c: torch.Tensor,
    n: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the sLSTM recurrence using a compiled per-step kernel.

    Mathematically identical to :func:`_slstm_recurrent_scan` but uses
    ``torch.compile`` on the step body to fuse ~12 GPU kernel launches
    into 2-3 fused kernels per timestep.
    """
    B, T, H, Dh = z_in.shape
    hidden_size = H * Dh

    outputs = z_in.new_empty(B, T, hidden_size)

    for t in range(T):
        ht = h
        c, n, m, h = _slstm_step_compiled(
            z_in[:, t], i_in[:, t], f_in[:, t], o_in[:, t],
            ht, Rz, Ri, Rf, Ro, c, n, m,
        )
        outputs[:, t] = h.reshape(B, hidden_size)

    return outputs, c, n, m, h


# ---------------------------------------------------------------------------
# sLSTMCell — single step, like nn.LSTMCell
# ---------------------------------------------------------------------------

class sLSTMCell(nn.Module):
    """One time-step of sLSTM.  Analogous to nn.LSTMCell.

    Recurrent weights are block-diagonal per head — each head only sees its
    own previous hidden slice, matching the paper's design.

    Call signature:
        h_t, new_state = cell(x_t, old_state)
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
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
        h_prev = state.h

        z_tilde = self.Wz(x_t).view(B, H, Dh) + torch.einsum("bhd,hde->bhe", h_prev, self.Rz)
        i_tilde = self.Wi(x_t).view(B, H, Dh) + torch.einsum("bhd,hde->bhe", h_prev, self.Ri)
        f_tilde = self.Wf(x_t).view(B, H, Dh) + torch.einsum("bhd,hde->bhe", h_prev, self.Rf)
        o_tilde = self.Wo(x_t).view(B, H, Dh) + torch.einsum("bhd,hde->bhe", h_prev, self.Ro)

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
# sLSTM — full sequence, multi-layer, bidirectional  (like nn.LSTM)
# ---------------------------------------------------------------------------

class sLSTM(nn.Module):
    """Multi-layer sLSTM with bidirectional support.

    Interface mirrors nn.LSTM.

    Args:
        input_size:   feature dimension of input
        hidden_size:  feature dimension of hidden state
        num_layers:   number of stacked sLSTM layers (default 1)
        num_heads:    number of heads per layer (can be int or list of ints
                       of length num_layers)
        bidirectional: if True, becomes bidirectional (default False)
        dropout:      dropout applied between layers (except after last layer,
                       default 0)
        bias:         whether to use bias in linear layers (default True)
        batch_first:  if True, input is (batch, seq, feature) instead of
                       (seq, batch, feature) (default True)

    Forward:
        input:  (batch, seq, input_size)  if batch_first=True,
                (seq, batch, input_size)  otherwise
        state:  an sLSTMState (optional, zero-initialised if None)
        returns: output, state
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for d in range(self.num_directions):
            for l in range(num_layers):
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
            Hh = self.num_heads[l]
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

    def _project_sequence(self, x: torch.Tensor, d: int, l: int):
        """Compute all input-dependent projections for one direction/layer."""
        Hh = self.num_heads[l]
        Wz = getattr(self, f"Wz_{d}_{l}")
        Wi = getattr(self, f"Wi_{d}_{l}")
        Wf = getattr(self, f"Wf_{d}_{l}")
        Wo = getattr(self, f"Wo_{d}_{l}")

        B, T, _ = x.shape
        z_in = Wz(x).view(B, T, Hh, -1)
        i_in = Wi(x).view(B, T, Hh, -1)
        f_in = Wf(x).view(B, T, Hh, -1)
        o_in = Wo(x).view(B, T, Hh, -1)

        return z_in, i_in, f_in, o_in

    def _run_layer(
        self, x: torch.Tensor, d: int, l: int,
        c: torch.Tensor, n: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
    ):
        Hh = self.num_heads[l]
        Dh = self.hidden_size // Hh

        z_in, i_in, f_in, o_in = self._project_sequence(x, d, l)
        Rz = getattr(self, f"Rz_{d}_{l}")
        Ri = getattr(self, f"Ri_{d}_{l}")
        Rf = getattr(self, f"Rf_{d}_{l}")
        Ro = getattr(self, f"Ro_{d}_{l}")

        if self.use_checkpoint and self.training:
            return _slstm_recurrent_scan(
                z_in, i_in, f_in, o_in, Rz, Ri, Rf, Ro, c, n, m, h,
            )
        return _slstm_recurrent_scan_optimized(
            z_in, i_in, f_in, o_in, Rz, Ri, Rf, Ro, c, n, m, h,
        )

    def forward(
        self,
        input: torch.Tensor,
        state=None,
    ):
        """If pack_state=False and num_layers=1, state is a bare sLSTMState."""
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)

        # Normalise: if bare state (pack_state=False, num_layers=1), wrap in tuple
        if not isinstance(state, tuple):
            state = (state,)

        layer_input = input
        final_states: List[sLSTMState] = []

        for l in range(self.num_layers):
            dir_outputs: List[torch.Tensor] = []
            c_dirs: List[torch.Tensor] = []
            n_dirs: List[torch.Tensor] = []
            m_dirs: List[torch.Tensor] = []
            h_dirs: List[torch.Tensor] = []

            s_l = state[l]

            for d in range(self.num_directions):
                if d == 1:
                    layer_input = torch.flip(layer_input, [1])

                c_dl = s_l.c[d]
                n_dl = s_l.n[d]
                m_dl = s_l.m[d]
                h_dl = s_l.h[d]

                if self.use_checkpoint and self.training:
                    out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                        self._run_layer, layer_input, d, l, c_dl, n_dl, m_dl, h_dl,
                        use_reentrant=False,
                    )
                else:
                    out, c_out, n_out, m_out, h_out = self._run_layer(
                        layer_input, d, l, c_dl, n_dl, m_dl, h_dl,
                    )

                if d == 1:
                    out = torch.flip(out, [1])

                dir_outputs.append(out)
                c_dirs.append(c_out)
                n_dirs.append(n_out)
                m_dirs.append(m_out)
                h_dirs.append(h_out)

            layer_output = torch.cat(dir_outputs, dim=-1) if self.num_directions > 1 else dir_outputs[0]

            if l < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            final_states.append(sLSTMState(
                torch.stack(c_dirs, dim=0),
                torch.stack(n_dirs, dim=0),
                torch.stack(m_dirs, dim=0),
                torch.stack(h_dirs, dim=0),
            ))

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
            f"batch_first={self.batch_first}, use_checkpoint={self.use_checkpoint})"
        )
