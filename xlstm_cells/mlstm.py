"""
mLSTM: Matrix-memory LSTM cell and layer.

Optimized over the reference implementation:
1. All input projections are fused: Wq/Wk/Wv/Wo -> 1 F.linear, Wi/Wf -> 1 F.linear
   (6 separate GEMM calls -> 2).
2. Only the true recurrence (C, n, m updates + output readout) stays in the loop.
3. Full nn.LSTM-compatible interface: multi-layer, bidirectional, dropout, batch_first.
4. Explicit state objects for TBPTT and single-step control.
5. Optional triton kernel backend via mlstm_kernels for zero-overhead recurrence.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
import torch.nn as nn
import torch.nn.functional as F

try:
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
    )
    _HAS_MLSTM_KERNELS = True
except ImportError:
    _HAS_MLSTM_KERNELS = False

_EPS = 1e-6
_MLSTM_CHUNK_SIZE = 64


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
# Scan kernels
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
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        ot = o[:, t]
        i_tilde_t = i_tilde[:, t]
        log_f_t = log_f[:, t]

        m_prev = m

        m = torch.maximum(log_f_t + m_prev, i_tilde_t)
        i_prime = torch.exp(i_tilde_t - m)
        f_prime = torch.exp(log_f_t + m_prev - m)

        vk = vt.unsqueeze(-1) * kt.unsqueeze(-2)

        C = f_prime[..., None, None] * C + i_prime[..., None, None] * vk
        n = f_prime[..., None] * n + i_prime[..., None] * kt

        h_tilde = torch.einsum("bhde,bhe->bhd", C, qt)
        qn = torch.einsum("bhd,bhd->bh", n, qt).abs()

        m_safe = m.clamp_max(0)
        exp_m_safe = torch.exp(m_safe)
        denom = torch.maximum(qn * exp_m_safe, torch.exp(m_safe - m)).clamp_min(_EPS)
        h = ot * ((h_tilde * exp_m_safe[..., None]) / denom[..., None])
        outputs[:, t] = h.reshape(B, hidden_size)

    return outputs, C, n, m


def _mlstm_recurrent_scan_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    i_tilde: torch.Tensor,
    log_f: torch.Tensor,
    C_init: torch.Tensor,
    n_init: torch.Tensor,
    m_init: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel mLSTM recurrence via numerically stable linear attention.

    Eliminates the Python ``for t in range(T)`` loop by expressing the
    recurrence as a causal linear attention over pre-computed projections.

    Args:
        q:        (B, T, H, Dh)  query projections
        k:        (B, T, H, Dh)  key projections (already scaled)
        v:        (B, T, H, Dh)  value projections
        o:        (B, T, H, Dh)  output gate projections (sigmoid'd)
        i_tilde:  (B, T, H)      input gate pre-activations
        log_f:    (B, T, H)      log(sigmoid(forget pre-activation))
        C_init:   (B, H, Dh, Dh) initial matrix memory
        n_init:   (B, H, Dh)     initial normalizer
        m_init:   (B, H)         initial stabilizer

    Returns:
        outputs:  (B, T, Hs)  where Hs = H * Dh
        C_final:  (B, H, Dh, Dh)
        n_final:  (B, H, Dh)
        m_final:  (B, H)
    """
    B, T, H, Dh = q.shape
    hidden_size = H * Dh

    cumsum_logf = torch.cumsum(log_f, dim=1)

    L = i_tilde - cumsum_logf
    L_cummax = L.cummax(dim=1).values

    L_cummax = torch.maximum(L_cummax, m_init.unsqueeze(1))

    m_attn = cumsum_logf + L_cummax
    m_final = m_attn[:, -1]

    L_bhxt = L.permute(0, 2, 1).unsqueeze(-1)
    Lcummax_bhxt = L_cummax.permute(0, 2, 1).unsqueeze(-2)

    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool))
    mask = mask[None, None, :, :]

    log_w_stable = L_bhxt - Lcummax_bhxt
    log_w_stable = log_w_stable.masked_fill(~mask, float('-inf'))
    w_stable = torch.exp(log_w_stable)

    dots = torch.einsum("bshd,bthd->bhst", k, q)
    w_stable = w_stable * dots

    numerator = torch.einsum("bhst,bshd->bthd", w_stable, v)

    denom_raw = w_stable.sum(dim=2).permute(0, 2, 1)

    init_scale_stable = torch.exp(m_init.unsqueeze(1) - L_cummax)

    C_init_flat = C_init.reshape(B * H, Dh, Dh)
    q_flat = q.permute(0, 2, 1, 3).reshape(B * H, T, Dh)
    init_h_tilde = torch.bmm(
        C_init_flat, q_flat.transpose(1, 2)
    ).reshape(B, H, Dh, T).permute(0, 3, 1, 2)
    init_h_tilde = init_h_tilde * init_scale_stable.unsqueeze(-1)

    init_qn = ((n_init.unsqueeze(1) * q).sum(dim=-1)) * init_scale_stable

    numerator = numerator + init_h_tilde
    denom_raw = denom_raw + init_qn

    m_safe = m_attn.clamp_max(0)
    exp_m_safe = torch.exp(m_safe)
    denom = torch.maximum(
        denom_raw.abs() * exp_m_safe, torch.exp(m_safe - m_attn)
    ).clamp_min(_EPS)
    h = o * ((numerator * exp_m_safe.unsqueeze(-1)) / denom.unsqueeze(-1))
    outputs = h.reshape(B, T, hidden_size)

    log_initial_decay = cumsum_logf[:, -1] + m_init - m_final

    D_stable_for_final = torch.exp(i_tilde - cumsum_logf - L_cummax[:, -1, :].unsqueeze(1))
    C_unnorm = (D_stable_for_final.unsqueeze(-1).unsqueeze(-1) * v.unsqueeze(-1) * k.unsqueeze(-2)).sum(dim=1)
    n_unnorm = (D_stable_for_final.unsqueeze(-1) * k).sum(dim=1)

    C_final = torch.exp(log_initial_decay)[:, :, None, None] * C_init + C_unnorm
    n_final = torch.exp(log_initial_decay)[:, :, None] * n_init + n_unnorm

    return outputs, C_final, n_final, m_final


def _mlstm_recurrent_scan_parallel_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    i_tilde: torch.Tensor,
    log_f: torch.Tensor,
    C_init: torch.Tensor,
    n_init: torch.Tensor,
    m_init: torch.Tensor,
    chunk_size: int = _MLSTM_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked parallel mLSTM scan: O(T * chunk_size) memory instead of O(T^2).

    Splits the sequence into chunks of ``chunk_size``, runs the parallel scan
    within each chunk, and carries the final (C, n, m) triplet forward as the
    initial state for the next chunk.  Mathematically exact (modulo FP
    associativity) and does NOT truncate the receptive field --- every
    timestep still attends to the full history through the state triplet.

    Args:
        q,k,v,o,i_tilde,log_f:  pre-computed projections (B, T, ...)
        C_init,n_init,m_init:   initial states (user-supplied or zero)
        chunk_size:             steps per parallel scan tile (default 64)

    Returns:
        outputs, C_final, n_final, m_final  (same signatures as the other scans)
    """
    B, T, H, Dh = q.shape
    hidden_size = H * Dh
    outputs = q.new_empty(B, T, hidden_size)

    C, n, m = C_init, n_init, m_init

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        out_chunk, C, n, m = _mlstm_recurrent_scan_parallel(
            q[:, start:end], k[:, start:end], v[:, start:end],
            o[:, start:end], i_tilde[:, start:end], log_f[:, start:end],
            C, n, m,
        )
        outputs[:, start:end] = out_chunk

    return outputs, C, n, m


# ---------------------------------------------------------------------------
# mLSTMCell -- single step, like nn.LSTMCell
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

        i_tilde = self.Wi(x_t)
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
        m_safe = m.clamp_max(0)
        exp_m_safe = torch.exp(m_safe)
        denom = torch.maximum(qn * exp_m_safe, torch.exp(m_safe - m)).clamp_min(_EPS)
        h = o * ((h_tilde * exp_m_safe[..., None]) / denom[..., None])

        return h.reshape(B, self.hidden_size), mLSTMState(C, n, m)


# ---------------------------------------------------------------------------
# mLSTM -- full sequence, multi-layer, bidirectional  (like nn.LSTM)
# ---------------------------------------------------------------------------

class mLSTM(nn.Module):
    """Multi-layer mLSTM with bidirectional support.

    Interface mirrors nn.LSTM.

    Args:
        input_size:      feature dimension of input
        hidden_size:     feature dimension of hidden state
        num_layers:      number of stacked mLSTM layers (default 1)
        num_heads:       number of heads per layer (can be int or list of ints
                         of length num_layers)
        bidirectional:   if True, becomes bidirectional (default False)
        dropout:         dropout applied between layers (except after last layer,
                         default 0)
        bias:            whether to use bias in linear layers (default True)
        batch_first:     if True, input is (batch, seq, feature) instead of
                         (seq, batch, feature) (default True)
        use_triton_kernels:  if True, use mlstm_kernels triton backend for
                             the recurrent scan (default True if available).
                             Falls back to chunked PyTorch parallel scan if
                             unavailable.
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
        use_triton_kernels: bool = True,
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
        self.use_checkpoint = use_checkpoint
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._has_bias = bias

        self._use_triton_kernels = use_triton_kernels and _HAS_MLSTM_KERNELS
        self._chunk_size = _MLSTM_CHUNK_SIZE
        self._mlstm_backend = None

        for d in range(self.num_directions):
            for l in range(num_layers):
                assert hidden_size % num_heads[l] == 0, (
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_heads[{l}] ({num_heads[l]})."
                )
                layer_input = input_size if l == 0 else hidden_size * self.num_directions
                Hh = num_heads[l]
                setattr(self, f"Wq_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wk_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wv_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wo_{d}_{l}", nn.Linear(layer_input, hidden_size, bias=bias))
                setattr(self, f"Wi_{d}_{l}", nn.Linear(layer_input, Hh, bias=bias))
                setattr(self, f"Wf_{d}_{l}", nn.Linear(layer_input, Hh, bias=bias))

        self.reset_parameters()
        if self._use_triton_kernels:
            self._init_triton_backend()

    def _init_triton_backend(self):
        config = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_limit_chunk",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            mode="train",
            chunk_size=self._chunk_size,
            return_last_states=True,
            autocast_kernel_dtype="float32",
        )
        self._mlstm_backend = mLSTMBackend(config=config)

    def reset_parameters(self) -> None:
        for d in range(self.num_directions):
            for l in range(self.num_layers):
                Hh = self.num_heads[l]
                std = 1.0 / math.sqrt(self.hidden_size)
                nn.init.normal_(getattr(self, f"Wq_{d}_{l}").weight, std=std)
                nn.init.normal_(getattr(self, f"Wk_{d}_{l}").weight, std=std)
                nn.init.normal_(getattr(self, f"Wv_{d}_{l}").weight, std=std)
                nn.init.xavier_normal_(getattr(self, f"Wo_{d}_{l}").weight)
                nn.init.normal_(getattr(self, f"Wi_{d}_{l}").weight, std=1e-2)
                nn.init.zeros_(getattr(self, f"Wf_{d}_{l}").weight)
                for gate in ("Wq", "Wk", "Wv", "Wo", "Wi", "Wf"):
                    lin = getattr(self, f"{gate}_{d}_{l}")
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)
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

    def _fuse_qkvo_weights(self, d: int, l: int):
        Wq = getattr(self, f"Wq_{d}_{l}")
        Wk = getattr(self, f"Wk_{d}_{l}")
        Wv = getattr(self, f"Wv_{d}_{l}")
        Wo = getattr(self, f"Wo_{d}_{l}")
        w = torch.cat([Wq.weight, Wk.weight, Wv.weight, Wo.weight], dim=0)
        if Wq.bias is not None:
            b = torch.cat([Wq.bias, Wk.bias, Wv.bias, Wo.bias], dim=0)
        else:
            b = None
        return w, b

    def _fuse_if_weights(self, d: int, l: int):
        Wi = getattr(self, f"Wi_{d}_{l}")
        Wf = getattr(self, f"Wf_{d}_{l}")
        w = torch.cat([Wi.weight, Wf.weight], dim=0)
        if Wi.bias is not None:
            b = torch.cat([Wi.bias, Wf.bias], dim=0)
        else:
            b = None
        return w, b

    def _project_sequence(
        self, x: torch.Tensor, d: int, l: int
    ) -> Tuple[torch.Tensor, ...]:
        Hh = self.num_heads[l]
        Dh = self.hidden_size // Hh
        B, T, _ = x.shape
        sf = math.sqrt(Dh)

        w_qkvo, b_qkvo = self._fuse_qkvo_weights(d, l)
        qkvo = F.linear(x, w_qkvo, b_qkvo)
        q, k, v, o_raw = qkvo.chunk(4, dim=-1)
        q = q.view(B, T, Hh, Dh)
        k = k.view(B, T, Hh, Dh) / sf
        v = v.view(B, T, Hh, Dh)
        o = torch.sigmoid(o_raw).view(B, T, Hh, Dh)

        w_if, b_if = self._fuse_if_weights(d, l)
        iff = F.linear(x, w_if, b_if)
        i_tilde, f_tilde = iff.chunk(2, dim=-1)
        log_f = F.logsigmoid(f_tilde)

        return q, k, v, o, i_tilde, log_f

    def _project_sequence_raw(
        self, x: torch.Tensor, d: int, l: int
    ) -> Tuple[torch.Tensor, ...]:
        """Same as _project_sequence but returns raw o and f (no sigmoid/logsigmoid)
        for the triton backend which applies its own activations."""
        Hh = self.num_heads[l]
        Dh = self.hidden_size // Hh
        B, T, _ = x.shape
        sf = math.sqrt(Dh)

        w_qkvo, b_qkvo = self._fuse_qkvo_weights(d, l)
        qkvo = F.linear(x, w_qkvo, b_qkvo)
        q, k, v, o_raw = qkvo.chunk(4, dim=-1)
        q = q.view(B, T, Hh, Dh)
        k = k.view(B, T, Hh, Dh) / sf
        v = v.view(B, T, Hh, Dh)
        o_raw = o_raw.view(B, T, Hh, Dh)

        w_if, b_if = self._fuse_if_weights(d, l)
        iff = F.linear(x, w_if, b_if)
        i_tilde, f_tilde = iff.chunk(2, dim=-1)

        return q, k, v, o_raw, i_tilde, f_tilde

    def _run_layer(
        self,
        x: torch.Tensor,
        d: int,
        l: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self._use_triton_kernels
                and x.size(1) % self._chunk_size == 0
                and not torch._dynamo.is_compiling()):
            return self._run_layer_kernels(x, d, l, C, n, m)
        return self._run_layer_native(x, d, l, C, n, m)

    def _run_layer_native(
        self,
        x: torch.Tensor,
        d: int,
        l: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, o, i_tilde, log_f = self._project_sequence(x, d, l)
        return _mlstm_recurrent_scan_parallel_chunked(
            q, k, v, o, i_tilde, log_f, C, n, m,
            chunk_size=self._chunk_size,
        )

    def _run_layer_kernels(
        self,
        x: torch.Tensor,
        d: int,
        l: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Hh = self.num_heads[l]
        Dh = self.hidden_size // Hh
        B, T, _ = x.shape

        q, k, v, o_raw, i_tilde, f_tilde = self._project_sequence_raw(x, d, l)

        q_k = q.permute(0, 2, 1, 3).contiguous()
        k_k = k.permute(0, 2, 1, 3).contiguous()
        v_k = v.permute(0, 2, 1, 3).contiguous()
        i_k = i_tilde.permute(0, 2, 1).contiguous()
        f_k = f_tilde.permute(0, 2, 1).contiguous()

        m_k = m.unsqueeze(-1)

        h_k, (C_out, n_out, m_out_k) = self._mlstm_backend(
            q=q_k, k=k_k, v=v_k, i=i_k, f=f_k,
            c_initial=C, n_initial=n, m_initial=m_k,
            return_last_states=True,
        )

        h_out = h_k.permute(0, 2, 1, 3).reshape(B, T, -1)
        o = torch.sigmoid(o_raw.reshape(B, T, Hh * Dh))
        h_out = o * h_out

        m_out = m_out_k.squeeze(-1)

        return h_out, C_out, n_out, m_out

    def forward(
        self,
        input: torch.Tensor,
        state=None,
    ):
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)

        if not isinstance(state, tuple):
            state = (state,)

        layer_input = input
        final_states: List[mLSTMState] = []

        for l in range(self.num_layers):
            s_l = state[l]

            if self.num_directions == 1:
                C_dl = s_l.C.squeeze(0)
                n_dl = s_l.n.squeeze(0)
                m_dl = s_l.m.squeeze(0)

                if self.use_checkpoint and self.training:
                    out, C_out, n_out, m_out = _torch_checkpoint(
                        self._run_layer, layer_input, 0, l, C_dl, n_dl, m_dl,
                        use_reentrant=False,
                    )
                else:
                    out, C_out, n_out, m_out = self._run_layer(
                        layer_input, 0, l, C_dl, n_dl, m_dl,
                    )

                layer_output = out
                final_states.append(mLSTMState(
                    C_out.unsqueeze(0), n_out.unsqueeze(0), m_out.unsqueeze(0),
                ))
            else:
                dir_outputs: List[torch.Tensor] = []
                C_dirs: List[torch.Tensor] = []
                n_dirs: List[torch.Tensor] = []
                m_dirs: List[torch.Tensor] = []

                for d in range(self.num_directions):
                    if d == 1:
                        layer_input = torch.flip(layer_input, [1])

                    C_dl = s_l.C[d]
                    n_dl = s_l.n[d]
                    m_dl = s_l.m[d]

                    if self.use_checkpoint and self.training:
                        out, C_out, n_out, m_out = _torch_checkpoint(
                            self._run_layer, layer_input, d, l, C_dl, n_dl, m_dl,
                            use_reentrant=False,
                        )
                    else:
                        out, C_out, n_out, m_out = self._run_layer(
                            layer_input, d, l, C_dl, n_dl, m_dl,
                        )

                    if d == 1:
                        out = torch.flip(out, [1])

                    dir_outputs.append(out)
                    C_dirs.append(C_out)
                    n_dirs.append(n_out)
                    m_dirs.append(m_out)

                layer_output = torch.cat(dir_outputs, dim=-1)
                final_states.append(mLSTMState(
                    torch.stack(C_dirs, dim=0),
                    torch.stack(n_dirs, dim=0),
                    torch.stack(m_dirs, dim=0),
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
            f"mLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}, "
            f"batch_first={self.batch_first}, use_checkpoint={self.use_checkpoint}, "
            f"use_triton_kernels={self._use_triton_kernels})"
        )
