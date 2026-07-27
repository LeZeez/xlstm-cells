"""
mLSTM: Matrix-memory LSTM cell and layer.

Optimized over the reference implementation:
1. All input projections (Wq, Wk, Wv, Wo, Wi, Wf) are computed once over the
   full sequence *before* the time loop — no redundant Linear calls at each step.
2. Only the true recurrence (C, n, m updates + output readout) stays in the loop.
3. Full nn.LSTM-compatible interface: multi-layer, bidirectional, dropout, batch_first.
4. Explicit state objects for TBPTT and single-step control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class mLSTMState:
    """State for mLSTM, held as named tensors for full user control.

    Shapes (single direction, single layer):
        C  (B, H, Dh, Dh)   matrix memory per head
        n  (B, H, Dh)       key normalizer per head
        m  (B, H)           log-space stabilizer per head
    """

    C: torch.Tensor
    n: torch.Tensor
    m: torch.Tensor

    @classmethod
    def init(
        cls,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device=None,
        dtype=None,
    ) -> "mLSTMState":
        H, Dh = num_heads, head_dim
        return cls(
            C=torch.zeros(batch_size, H, Dh, Dh, device=device, dtype=dtype),
            n=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            m=torch.zeros(batch_size, H, device=device, dtype=dtype),
        )

    def detach(self) -> "mLSTMState":
        return mLSTMState(self.C.detach(), self.n.detach(), self.m.detach())

    def to(self, *args, **kwargs) -> "mLSTMState":
        return mLSTMState(
            self.C.to(*args, **kwargs),
            self.n.to(*args, **kwargs),
            self.m.to(*args, **kwargs),
        )

    def clone(self) -> "mLSTMState":
        return mLSTMState(self.C.clone(), self.n.clone(), self.m.clone())

    def __repr__(self) -> str:
        return (
            f"mLSTMState(C={list(self.C.shape)}, n={list(self.n.shape)}, "
            f"m={list(self.m.shape)})"
        )


# ---------------------------------------------------------------------------
# Optimized step function (standalone so torch.compile can trace it cleanly)
# ---------------------------------------------------------------------------

def _mlstm_recurrent_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    i_tilde: torch.Tensor,
    log_f: torch.Tensor,
    C: torch.Tensor,
    n: torch.Tensor,
    m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the mLSTM recurrence over pre-computed projections.

    Args:
        q:        (B, T, H, Dh)   query projections
        k:        (B, T, H, Dh)   key projections (already scaled by 1/sqrt(Dh))
        v:        (B, T, H, Dh)   value projections
        o:        (B, T, H, Dh)   output gate projections (sigmoi'd)
        i_tilde:  (B, T, H)      input gate pre-activations
        log_f:    (B, T, H)      log(sigmoid(forget pre-activation))
        C:        (B, H, Dh, Dh)  initial matrix memory
        n:        (B, H, Dh)     initial normalizer
        m:        (B, H)         initial stabilizer

    Returns:
        outputs: (B, T, Hs)  where Hs = H * Dh
        C_final: (B, H, Dh, Dh)
        n_final: (B, H, Dh)
        m_final: (B, H)
    """
    B, T, H, Dh = q.shape
    hidden_size = H * Dh

    outputs = q.new_empty(B, T, hidden_size)

    for t in range(T):
        qt = q[:, t]    # (B, H, Dh)
        kt = k[:, t]
        vt = v[:, t]
        ot = o[:, t]
        i_tilde_t = i_tilde[:, t]   # (B, H)
        log_f_t = log_f[:, t]

        m_prev = m

        m = torch.maximum(log_f_t + m_prev, i_tilde_t)
        i_prime = torch.exp(i_tilde_t - m)          # (B, H)
        f_prime = torch.exp(log_f_t + m_prev - m)    # (B, H)

        # outer product v ⊗ k  -> (B, H, Dh, Dh)
        vk = vt.unsqueeze(-1) * kt.unsqueeze(-2)

        C = f_prime[..., None, None] * C + i_prime[..., None, None] * vk
        n = f_prime[..., None] * n + i_prime[..., None] * kt

        # readout
        h_tilde = torch.einsum("bhde,bhe->bhd", C, qt)            # (B, H, Dh)
        qn = torch.einsum("bhd,bhd->bh", n, qt).abs()             # (B, H)
        denom = torch.maximum(qn, torch.exp(-m)).clamp_min(1e-6)
        h = ot * (h_tilde / denom[..., None])
        outputs[:, t] = h.reshape(B, hidden_size)

    return outputs, C, n, m


# ---------------------------------------------------------------------------
# mLSTMCell — single step, like nn.LSTMCell
# ---------------------------------------------------------------------------

class mLSTMCell(nn.Module):
    """One time-step of mLSTM.  Analogous to nn.LSTMCell.

    Call signature:
        h_t, new_state = cell(x_t, old_state)

    Args:
        input_size:  dimensionality of x_t
        hidden_size: dimensionality of h_t
        num_heads:   number of independent heads (hidden_size must be divisible by this)
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.Wq = nn.Linear(input_size, hidden_size, bias=False)
        self.Wk = nn.Linear(input_size, hidden_size, bias=False)
        self.Wv = nn.Linear(input_size, hidden_size, bias=False)
        self.Wo = nn.Linear(input_size, hidden_size, bias=False)
        self.Wi = nn.Linear(input_size, num_heads, bias=False)
        self.Wf = nn.Linear(input_size, num_heads, bias=False)

        self._sf = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in (self.Wq, self.Wk, self.Wv):
            nn.init.normal_(w.weight, std=std)
        nn.init.xavier_normal_(self.Wo.weight)
        nn.init.normal_(self.Wi.weight, std=1e-2)
        nn.init.zeros_(self.Wf.weight)

    def init_state(self, batch_size: int, device=None, dtype=None) -> mLSTMState:
        return mLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def forward(self, x_t: torch.Tensor, state: mLSTMState) -> Tuple[torch.Tensor, mLSTMState]:
        B = x_t.size(0)
        H, Dh = self.num_heads, self.head_dim

        q = self.Wq(x_t).view(B, H, Dh)
        k = self.Wk(x_t).view(B, H, Dh) / self._sf
        v = self.Wv(x_t).view(B, H, Dh)
        o = torch.sigmoid(self.Wo(x_t)).view(B, H, Dh)

        i_tilde = self.Wi(x_t)              # (B, H)
        f_tilde = self.Wf(x_t)
        log_f = F.logsigmoid(f_tilde)

        m_prev = state.m
        m = torch.maximum(log_f + m_prev, i_tilde)
        i_prime = torch.exp(i_tilde - m)
        f_prime = torch.exp(log_f + m_prev - m)

        vk = v.unsqueeze(-1) * k.unsqueeze(-2)
        C = f_prime[..., None, None] * state.C + i_prime[..., None, None] * vk
        n = f_prime[..., None] * state.n + i_prime[..., None] * k

        h_tilde = torch.einsum("bhde,bhe->bhd", C, q)
        qn = torch.einsum("bhd,bhd->bh", n, q).abs()
        denom = torch.maximum(qn, torch.exp(-m)).clamp_min(1e-6)
        h = o * (h_tilde / denom[..., None])

        return h.reshape(B, self.hidden_size), mLSTMState(C, n, m)


# ---------------------------------------------------------------------------
# mLSTM — full sequence, multi-layer, bidirectional  (like nn.LSTM)
# ---------------------------------------------------------------------------

class mLSTM(nn.Module):
    """Multi-layer mLSTM with bidirectional support.

    Interface mirrors nn.LSTM.

    Args:
        input_size:   feature dimension of input
        hidden_size:  feature dimension of hidden state
        num_layers:   number of stacked mLSTM layers (default 1)
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
        state:  an mLSTMState (optional, zero-initialised if None)
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
    ):
        super().__init__()

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers
        assert len(num_heads) == num_layers

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pack_state = pack_state
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Build per-direction per-layer linear projections
        # Shape convention: (num_directions, num_layers, ...)
        for d in range(self.num_directions):
            for l in range(num_layers):
                layer_input = input_size if l == 0 else hidden_size * self.num_directions
                Hh = num_heads[l]
                setattr(self, f"Wq_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wk_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wv_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wo_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wi_{d}_{l}", nn.Linear(layer_input, Hh, bias=bias))
                setattr(self, f"Wf_{d}_{l}", nn.Linear(layer_input, Hh, bias=bias))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for d in range(self.num_directions):
            for l in range(self.num_layers):
                Hh = self.num_heads[l]
                std = 1.0 / math.sqrt(self.hidden_size)
                getattr(self, f"Wq_{d}_{l}").reset_parameters()
                getattr(self, f"Wk_{d}_{l}").reset_parameters()
                getattr(self, f"Wv_{d}_{l}").reset_parameters()
                nn.init.normal_(getattr(self, f"Wq_{d}_{l}").weight, std=std)
                nn.init.normal_(getattr(self, f"Wk_{d}_{l}").weight, std=std)
                nn.init.normal_(getattr(self, f"Wv_{d}_{l}").weight, std=std)
                nn.init.xavier_normal_(getattr(self, f"Wo_{d}_{l}").weight)
                nn.init.normal_(getattr(self, f"Wi_{d}_{l}").weight, std=1e-2)
                nn.init.zeros_(getattr(self, f"Wf_{d}_{l}").weight)
                if getattr(self, f"Wf_{d}_{l}").bias is not None:
                    nn.init.constant_(getattr(self, f"Wf_{d}_{l}").bias, 3.0)

    def init_state(self, batch_size: int, device=None, dtype=None):
        states = []
        for l in range(self.num_layers):
            Hh = self.num_heads[l]
            Dh = self.hidden_size // Hh
            D = self.num_directions
            s = mLSTMState(
                C=torch.zeros(D, batch_size, Hh, Dh, Dh, device=device, dtype=dtype),
                n=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                m=torch.zeros(D, batch_size, Hh, device=device, dtype=dtype),
            )
            states.append(s)
        result = tuple(states)
        if not self.pack_state and self.num_layers == 1:
            return result[0]
        return result

    def _project_sequence(
        self, x: torch.Tensor, d: int, l: int
    ) -> Tuple[torch.Tensor, ...]:
        """Compute all input-dependent projections for one direction/layer."""
        Hh = self.num_heads[l]
        Dh = self.hidden_size // Hh
        Wq = getattr(self, f"Wq_{d}_{l}")
        Wk = getattr(self, f"Wk_{d}_{l}")
        Wv = getattr(self, f"Wv_{d}_{l}")
        Wo = getattr(self, f"Wo_{d}_{l}")
        Wi = getattr(self, f"Wi_{d}_{l}")
        Wf = getattr(self, f"Wf_{d}_{l}")

        B, T, _ = x.shape
        q = Wq(x).view(B, T, Hh, Dh)
        k = Wk(x).view(B, T, Hh, Dh) / math.sqrt(Dh)
        v = Wv(x).view(B, T, Hh, Dh)
        o = torch.sigmoid(Wo(x)).view(B, T, Hh, Dh)
        i_tilde = Wi(x)          # (B, T, Hh)
        f_tilde = Wf(x)
        log_f = F.logsigmoid(f_tilde)

        return q, k, v, o, i_tilde, log_f

    def _run_layer(
        self,
        x: torch.Tensor,
        d: int,
        l: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project full sequence then run the recurrent scan."""
        Hh = self.num_heads[l]
        Dh = self.hidden_size // Hh

        q, k, v, o, i_tilde, log_f = self._project_sequence(x, d, l)
        return _mlstm_recurrent_scan(q, k, v, o, i_tilde, log_f, C, n, m)

    def forward(
        self,
        input: torch.Tensor,
        state=None,
    ):
        """If pack_state=False and num_layers=1, state is a bare mLSTMState."""
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)

        # Normalise: if bare state (pack_state=False, num_layers=1), wrap in tuple
        if not isinstance(state, tuple):
            state = (state,)

        layer_input = input
        final_states: List[mLSTMState] = []

        for l in range(self.num_layers):
            dir_outputs: List[torch.Tensor] = []
            C_dirs: List[torch.Tensor] = []
            n_dirs: List[torch.Tensor] = []
            m_dirs: List[torch.Tensor] = []

            s_l = state[l]

            for d in range(self.num_directions):
                if d == 1:
                    layer_input = torch.flip(layer_input, [1])

                C_dl = s_l.C[d]
                n_dl = s_l.n[d]
                m_dl = s_l.m[d]

                out, C_out, n_out, m_out = self._run_layer(
                    layer_input, d, l, C_dl, n_dl, m_dl
                )

                if d == 1:
                    out = torch.flip(out, [1])

                dir_outputs.append(out)
                C_dirs.append(C_out)
                n_dirs.append(n_out)
                m_dirs.append(m_out)

            layer_output = torch.cat(dir_outputs, dim=-1) if self.num_directions > 1 else dir_outputs[0]

            if l < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            final_states.append(mLSTMState(
                torch.stack(C_dirs, dim=0),
                torch.stack(n_dirs, dim=0),
                torch.stack(m_dirs, dim=0),
            ))

            layer_input = layer_output

        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)

        result = tuple(final_states)
        if not self.pack_state and self.num_layers == 1:
            return layer_output, result[0]
        return layer_output, result

    def flatten_parameters(self) -> None:
        """No-op: provided for nn.LSTM compatibility."""
        pass
