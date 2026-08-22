"""
mLSTM: Matrix-memory LSTM cell and layer.

100% aligned with the official xLSTM paper (arXiv:2405.04517v2, Figure 11) and NX-AI/xlstm:
1. Gating from concatenated [q, k, v] vectors (dimension 3 * inner_dim -> 2 * num_heads).
2. Linspace forget-gate bias init (3.4 to 6.0 across heads) for diverse memory timescales.
3. Input-gate bias init ~ N(0.0, 0.1).
4. Matrix memory covariance update with exp-stabilizer m and configurable denominator floor eps (default 1e-6).
5. Optional Triton chunkwise kernel backends (xl_chunk and limit_chunk via mlstm_kernels).
6. Native chunked-parallel scan fallback for CPU, non-divisible sequence lengths, or compilation.
7. Packed document boundaries reset (f_tilde = -1000.0).
8. Non-reentrant activation checkpointing support.
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

from ._utils import (
    PackedBoundariesMode,
    get_packed_boundaries_override_mode,
    normalize_and_validate_num_heads,
)
from .components.init import bias_linspace_init_, small_init_init_, wang_init_
from .components.ln import MultiHeadLayerNorm

try:
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
    )
    _HAS_MLSTM_KERNELS = True
except ImportError:
    _HAS_MLSTM_KERNELS = False

_EPS = 1e-6
_MLSTM_CHUNK_SIZE = 128
_BOUNDARY_RESET_LOGF = -1000.0
_MAX_FORGET_BIAS = 4.0

_TRITON_CHUNKWISE_KERNELS = {
    "limit_chunk": "chunkwise--triton_limit_chunk",
    "xl_chunk": "chunkwise--triton_xl_chunk",
}

_triton_fallback_warned = False


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
    def init(cls, batch_size: int, num_heads: int, head_dim: int,
             device=None, dtype=None) -> "mLSTMState":
        """Initializes zero state tensors for mLSTM."""
        H, Dh = num_heads, head_dim
        return cls(
            C=torch.zeros(batch_size, H, Dh, Dh, device=device, dtype=dtype),
            n=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            m=torch.zeros(batch_size, H, device=device, dtype=dtype),
        )

    def detach(self) -> "mLSTMState":
        """Detaches state tensors from the autograd graph."""
        return mLSTMState(self.C.detach(), self.n.detach(), self.m.detach())

    def to(self, *args, **kwargs) -> "mLSTMState":
        """Moves state tensors to specified device or dtype."""
        return mLSTMState(self.C.to(*args, **kwargs),
                          self.n.to(*args, **kwargs),
                          self.m.to(*args, **kwargs))

    def clone(self) -> "mLSTMState":
        """Returns a cloned copy of the state object."""
        return mLSTMState(self.C.clone(), self.n.clone(), self.m.clone())

    def __repr__(self) -> str:
        return (f"mLSTMState(C={list(self.C.shape)}, n={list(self.n.shape)}, "
                f"m={list(self.m.shape)})")


# ---------------------------------------------------------------------------
# Native Scans (Recurrent & Parallel Chunked)
# ---------------------------------------------------------------------------

def _mlstm_recurrent_scan(
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recurrent scan for step-by-step sequential processing.
    Accepts (q, k, v, [o], i_tilde, log_f, C, n, m, [eps]).
    """
    if len(args) >= 9 and args[3].dim() == 4:
        q, k, v, o, i_tilde, log_f, C, n, m = args[:9]
        eps = args[9] if len(args) > 9 else kwargs.get("eps", _EPS)
    else:
        q, k, v, i_tilde, log_f, C, n, m = args[:8]
        o = None
        eps = args[8] if len(args) > 8 else kwargs.get("eps", _EPS)

    B, T, H, Dh = q.shape
    hidden_size = H * Dh
    output_list: List[torch.Tensor] = []

    for t in range(T):
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        it_raw = i_tilde[:, t]
        log_ft = log_f[:, t]

        m_prev = m
        m = torch.maximum(log_ft + m_prev, it_raw)
        i_prime = torch.exp(it_raw - m)
        f_prime = torch.exp(log_ft + m_prev - m)

        vk = vt.unsqueeze(-1) * kt.unsqueeze(-2)
        C = f_prime[..., None, None] * C + i_prime[..., None, None] * vk
        n = f_prime[..., None] * n + i_prime[..., None] * kt

        h_tilde = torch.einsum("bhde,bhe->bhd", C, qt)
        qn = torch.einsum("bhd,bhd->bh", n, qt).abs()
        denom = torch.maximum(qn, torch.exp(-m)) + eps
        h = h_tilde / denom[..., None]
        if o is not None:
            h = o[:, t] * h
        output_list.append(h.reshape(B, hidden_size))

    outputs = torch.stack(output_list, dim=1)
    return outputs, C, n, m


def _mlstm_recurrent_scan_parallel(
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel mLSTM recurrence via numerically stable linear attention.
    Accepts (q, k, v, [o], i_tilde, log_f, C_init, n_init, m_init, [eps]).
    """
    if len(args) >= 9 and args[3].dim() == 4:
        q, k, v, o, i_tilde, log_f, C_init, n_init, m_init = args[:9]
        eps = args[9] if len(args) > 9 else kwargs.get("eps", _EPS)
    else:
        q, k, v, i_tilde, log_f, C_init, n_init, m_init = args[:8]
        o = None
        eps = args[8] if len(args) > 8 else kwargs.get("eps", _EPS)

    B, T, H, Dh = q.shape
    hidden_size = H * Dh

    log_f_perm = log_f.permute(0, 2, 1)
    f_cum = torch.cumsum(log_f_perm, dim=-1)

    f_cum_inter = f_cum[..., None]
    f_cum_intra = f_cum[..., None, :]
    decay_intra = f_cum_inter - f_cum_intra

    causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    decay_intra = decay_intra.masked_fill(~causal_mask, float("-inf"))

    i_tilde_perm = i_tilde.permute(0, 2, 1)
    d_raw = decay_intra + i_tilde_perm[..., None, :]

    m_intra = torch.max(d_raw, dim=-1)[0]
    m_inter = m_init[..., None] + f_cum
    m_attn = torch.maximum(m_inter, m_intra)

    d_stable = d_raw - m_attn[..., None]
    D_intra = torch.exp(d_stable).masked_fill(~causal_mask, 0.0)

    f_carry = torch.exp(m_inter - m_attn)
    C_inter_val = torch.einsum("bht,bhde->bhtde", f_carry, C_init)
    n_inter_val = torch.einsum("bht,bhd->bhtd", f_carry, n_init)

    q_p = q.permute(0, 2, 1, 3)
    k_p = k.permute(0, 2, 1, 3)
    v_p = v.permute(0, 2, 1, 3)

    S = torch.einsum("bhid,bhjd->bhij", q_p, k_p)
    A = S * D_intra
    H_intra = torch.einsum("bhij,bhjd->bhid", A, v_p)

    h_inter = torch.einsum("bhtde,bhte->bhtd", C_inter_val, q_p)
    numerator = H_intra + h_inter

    denom_intra = A.sum(dim=-1)
    denom_inter = torch.einsum("bhtd,bhtd->bht", n_inter_val, q_p)
    denom_raw = (denom_intra + denom_inter).permute(0, 2, 1)

    denom = torch.maximum(denom_raw.abs(), torch.exp(-m_attn.permute(0, 2, 1))) + eps
    h = numerator.permute(0, 2, 1, 3) / denom.unsqueeze(-1)
    if o is not None:
        h = o * h
    outputs = h.reshape(B, T, hidden_size)

    m_final = m_attn[..., -1].detach()
    f_decay_total = torch.exp(f_cum[..., -1] + m_init - m_final)
    i_decay_total = torch.exp(d_raw[..., -1, :] - m_final[..., None])
    v_weighted = torch.einsum("bht,bhtd->bhtd", i_decay_total, v_p)
    C_final = (f_decay_total[..., None, None] * C_init +
               torch.einsum("bhtd,bhte->bhde", v_weighted, k_p))
    n_final = (f_decay_total[..., None] * n_init +
               torch.einsum("bht,bhtd->bhd", i_decay_total, k_p))

    return outputs, C_final, n_final, m_final


def _mlstm_recurrent_scan_parallel_chunked(
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked parallel mLSTM scan: O(T * chunk_size) memory."""
    if len(args) >= 9 and args[3].dim() == 4:
        q, k, v, o, i_tilde, log_f, C_init, n_init, m_init = args[:9]
        chunk_size = kwargs.get("chunk_size", _MLSTM_CHUNK_SIZE)
        eps = args[9] if len(args) > 9 else kwargs.get("eps", _EPS)
    else:
        q, k, v, i_tilde, log_f, C_init, n_init, m_init = args[:8]
        o = None
        chunk_size = kwargs.get("chunk_size", _MLSTM_CHUNK_SIZE)
        eps = args[8] if len(args) > 8 else kwargs.get("eps", _EPS)

    T = q.shape[1]
    output_chunks = []
    C, n, m = C_init, n_init, m_init

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        o_chunk = o[:, start:end] if o is not None else None
        if o_chunk is not None:
            out_chunk, C, n, m = _mlstm_recurrent_scan_parallel(
                q[:, start:end], k[:, start:end], v[:, start:end], o_chunk,
                i_tilde[:, start:end], log_f[:, start:end],
                C, n, m,
                eps=eps,
            )
        else:
            out_chunk, C, n, m = _mlstm_recurrent_scan_parallel(
                q[:, start:end], k[:, start:end], v[:, start:end],
                i_tilde[:, start:end], log_f[:, start:end],
                C, n, m,
                eps=eps,
            )
        output_chunks.append(out_chunk)

    outputs = torch.cat(output_chunks, dim=1)
    return outputs, C, n, m


# ---------------------------------------------------------------------------
# mLSTMCell -- Single step
# ---------------------------------------------------------------------------

class mLSTMCell(nn.Module):
    """Single time-step Matrix LSTM (mLSTM) cell.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden feature dimension (must be divisible by num_heads).
        num_heads: Number of matrix-memory heads. Default: 4.
        bias: Whether projection layers include bias. Default: False.
        eps: Denominator stabilizer epsilon. Default: 1e-6.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4,
                 bias: bool = False, eps: float = _EPS):
        """Initializes single-step mLSTMCell."""
        super().__init__()
        if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size <= 0:
            raise ValueError(f"mLSTMCell: input_size must be a positive integer, got {input_size}")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError(f"mLSTMCell: hidden_size must be a positive integer, got {hidden_size}")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"mLSTMCell: num_heads must be a positive integer, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(f"mLSTMCell: hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite float")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = float(eps)

        self.q_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(input_size, hidden_size, bias=bias)

        # Gates from [q, k, v]
        self.igate = nn.Linear(3 * hidden_size, num_heads, bias=True)
        self.fgate = nn.Linear(3 * hidden_size, num_heads, bias=True)

        self._sf = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes projection and gate parameters."""
        small_init_init_(self.q_proj.weight, dim=self.input_size)
        small_init_init_(self.k_proj.weight, dim=self.input_size)
        small_init_init_(self.v_proj.weight, dim=self.input_size)
        if self.q_proj.bias is not None: nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None: nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None: nn.init.zeros_(self.v_proj.bias)

        nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.4, end=6.0)
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def init_state(self, batch_size: int, device=None, dtype=None) -> mLSTMState:
        """Initializes zero state for one step of mLSTM."""
        return mLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def forward(
        self,
        x_or_q: torch.Tensor,
        state_or_k: Union[mLSTMState, torch.Tensor],
        v: Optional[torch.Tensor] = None,
        state: Optional[mLSTMState] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        """Runs a single recurrent time-step of mLSTM.

        Args:
            x_or_q: Input token tensor (B, input_size) or precomputed q (B, hidden_size).
            state_or_k: Previous mLSTMState, or precomputed k (B, hidden_size).
            v: Optional precomputed v (B, hidden_size).
            state: Previous mLSTMState when precomputed (q, k, v) are provided.

        Returns:
            Tuple of (output token tensor (B, hidden_size), new_state).
        """
        if v is None and isinstance(state_or_k, mLSTMState):
            x_t = x_or_q
            st = state_or_k
            q = self.q_proj(x_t)
            k = self.k_proj(x_t)
            v_vec = self.v_proj(x_t)
        else:
            q = x_or_q
            k = state_or_k
            v_vec = v
            st = state

        B = q.size(0)
        H, Dh = self.num_heads, self.head_dim

        # Gating from [q, k, v]
        if_input = torch.cat([q, k, v_vec], dim=-1)
        i_tilde = self.igate(if_input)
        f_tilde = self.fgate(if_input)
        log_f = F.logsigmoid(f_tilde)

        q_view = q.view(B, H, Dh)
        k_view = (k / self._sf).view(B, H, Dh)
        v_view = v_vec.view(B, H, Dh)

        m_prev = st.m
        m = torch.maximum(log_f + m_prev, i_tilde)
        i_prime = torch.exp(i_tilde - m)
        f_prime = torch.exp(log_f + m_prev - m)

        vk = v_view.unsqueeze(-1) * k_view.unsqueeze(-2)
        C = f_prime[..., None, None] * st.C + i_prime[..., None, None] * vk
        n = f_prime[..., None] * st.n + i_prime[..., None] * k_view

        h_tilde = torch.einsum("bhde,bhe->bhd", C, q_view)
        qn = torch.einsum("bhd,bhd->bh", n, q_view).abs()
        denom = torch.maximum(qn, torch.exp(-m)) + self.eps
        h = h_tilde / denom[..., None]

        return h.reshape(B, self.hidden_size), mLSTMState(C, n, m)

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamps forget gate bias to prevent numerical saturation."""
        if self.fgate.bias is not None:
            self.fgate.bias.data.clamp_(-max_val, max_val)


# ---------------------------------------------------------------------------
# mLSTM -- Full Sequence Layer
# ---------------------------------------------------------------------------

class mLSTM(nn.Module):
    """Multi-layer Matrix LSTM (mLSTM) sequence model.

    Implements official xLSTM matrix memory recurrence with exponential gating,
    supporting Triton kernel acceleration, native chunked parallel scans,
    non-reentrant activation checkpointing, and packed document boundary resets.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden feature dimension (must be divisible by num_heads).
        num_layers: Number of stacked mLSTM layers. Default: 1.
        num_heads: Number of attention/recurrent heads per layer. Default: 4.
        bias: Whether projection layers include bias. Default: False.
        batch_first: If True, inputs/outputs have shape (B, T, D); otherwise (T, B, D). Default: True.
        dropout: Dropout applied between stacked layers. Default: 0.0.
        bidirectional: If True, processes sequence in both forward and backward directions. Default: False.
        pack_state: If True, returns states as a tuple across layers. Default: True.
        use_checkpoint: Whether to use gradient activation checkpointing. Default: False.
        use_triton_kernels: Whether to use Triton kernels from mlstm_kernels if available. Default: True.
        chunkwise_kernel: Name of Triton chunkwise kernel ("limit_chunk" or "xl_chunk"). Default: "xl_chunk".
        chunk_size: Chunk size for chunked parallel scan. Default: 128.
        eps: Denominator stabilizer epsilon. Default: 1e-6.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: Union[int, List[int], Tuple[int, ...]] = 4,
        bias: bool = False,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pack_state: bool = True,
        use_checkpoint: bool = False,
        use_triton_kernels: bool = True,
        chunkwise_kernel: str = "xl_chunk",
        chunk_size: int = _MLSTM_CHUNK_SIZE,
        eps: Optional[float] = None,
    ):
        """Initializes multi-layer mLSTM sequence model."""
        super().__init__()
        heads_list = normalize_and_validate_num_heads(num_heads, num_layers, hidden_size, "mLSTM")
        if chunkwise_kernel not in _TRITON_CHUNKWISE_KERNELS:
            raise ValueError(
                f"mLSTM: unknown chunkwise_kernel {chunkwise_kernel!r}, "
                f"expected one of {list(_TRITON_CHUNKWISE_KERNELS.keys())}"
            )

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a strictly positive integer, got {chunk_size}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = heads_list
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pack_state = pack_state
        self.use_checkpoint = use_checkpoint
        self._use_triton_kernels = use_triton_kernels and _HAS_MLSTM_KERNELS
        self._chunkwise_kernel = chunkwise_kernel
        self._chunk_size = chunk_size
        self._mlstm_backend = None

        if eps is None:
            eps = _EPS
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite float")
        self._eps = float(eps)

        for layer_idx in range(num_layers):
            for d in range(self.num_directions):
                in_sz = input_size if layer_idx == 0 else hidden_size * self.num_directions
                q_p = nn.Linear(in_sz, hidden_size, bias=bias)
                k_p = nn.Linear(in_sz, hidden_size, bias=bias)
                v_p = nn.Linear(in_sz, hidden_size, bias=bias)
                ig = nn.Linear(3 * hidden_size, self.num_heads[layer_idx], bias=True)
                fg = nn.Linear(3 * hidden_size, self.num_heads[layer_idx], bias=True)

                setattr(self, f"q_proj_{d}_{layer_idx}", q_p)
                setattr(self, f"k_proj_{d}_{layer_idx}", k_p)
                setattr(self, f"v_proj_{d}_{layer_idx}", v_p)
                setattr(self, f"igate_{d}_{layer_idx}", ig)
                setattr(self, f"fgate_{d}_{layer_idx}", fg)

        if dropout > 0.0 and num_layers > 1:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

        self.reset_parameters()
        if self._use_triton_kernels:
            self._init_triton_backend()

    def flatten_parameters(self) -> None:
        """No-op for nn.LSTM compatibility."""
        pass

    def _init_triton_backend(self) -> None:
        """Initializes Triton backend configuration."""
        config = mLSTMBackendConfig(
            chunkwise_kernel=_TRITON_CHUNKWISE_KERNELS[self._chunkwise_kernel],
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            chunk_size=self._chunk_size,
            return_last_states=True,
            autocast_kernel_dtype="float32",
            eps=self._eps,
        )
        self._mlstm_backend = mLSTMBackend(config=config)

    def reset_parameters(self) -> None:
        """Initializes layer parameters."""
        for layer_idx in range(self.num_layers):
            for d in range(self.num_directions):
                in_sz = self.input_size if layer_idx == 0 else self.hidden_size * self.num_directions
                qp = getattr(self, f"q_proj_{d}_{layer_idx}")
                kp = getattr(self, f"k_proj_{d}_{layer_idx}")
                vp = getattr(self, f"v_proj_{d}_{layer_idx}")
                ig = getattr(self, f"igate_{d}_{layer_idx}")
                fg = getattr(self, f"fgate_{d}_{layer_idx}")

                small_init_init_(qp.weight, dim=in_sz)
                small_init_init_(kp.weight, dim=in_sz)
                small_init_init_(vp.weight, dim=in_sz)
                if qp.bias is not None: nn.init.zeros_(qp.bias)
                if kp.bias is not None: nn.init.zeros_(kp.bias)
                if vp.bias is not None: nn.init.zeros_(vp.bias)

                nn.init.zeros_(fg.weight)
                bias_linspace_init_(fg.bias, start=3.4, end=6.0)
                nn.init.zeros_(ig.weight)
                nn.init.normal_(ig.bias, mean=0.0, std=0.1)

    def init_state(self, batch_size: int, device=None, dtype=None) -> Union[mLSTMState, Tuple[mLSTMState, ...]]:
        """Initializes zero state structures for all layers."""
        states = []
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = self.hidden_size // Hh
            D = self.num_directions
            states.append(mLSTMState(
                C=torch.zeros(D, batch_size, Hh, Dh, Dh, device=device, dtype=dtype),
                n=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                m=torch.zeros(D, batch_size, Hh, device=device, dtype=dtype),
            ))
        return tuple(states) if self.pack_state else (states[0] if self.num_layers == 1 else tuple(states))

    def _project(self, x: torch.Tensor, d: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        Hh = self.num_heads[layer_idx]
        Dh = self.hidden_size // Hh

        qp = getattr(self, f"q_proj_{d}_{layer_idx}")
        kp = getattr(self, f"k_proj_{d}_{layer_idx}")
        vp = getattr(self, f"v_proj_{d}_{layer_idx}")
        ig = getattr(self, f"igate_{d}_{layer_idx}")
        fg = getattr(self, f"fgate_{d}_{layer_idx}")

        q_raw = qp(x)
        k_raw = kp(x)
        v_raw = vp(x)

        if_input = torch.cat([q_raw, k_raw, v_raw], dim=-1)
        i_tilde = ig(if_input)
        f_raw = fg(if_input)

        q = q_raw.view(B, T, Hh, Dh)
        k = (k_raw / math.sqrt(Dh)).view(B, T, Hh, Dh)
        v = v_raw.view(B, T, Hh, Dh)

        return q, k, v, i_tilde, f_raw

    def _run_layer_native(
        self,
        x: torch.Tensor,
        d: int,
        layer_idx: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, i_tilde, f_raw = self._project(x, d, layer_idx)
        log_f = F.logsigmoid(f_raw)
        if boundaries is not None:
            log_f = log_f.masked_fill(
                boundaries.to(device=log_f.device, dtype=torch.bool).unsqueeze(-1),
                _BOUNDARY_RESET_LOGF,
            )
        return _mlstm_recurrent_scan_parallel_chunked(
            q, k, v, i_tilde, log_f, C, n, m,
            chunk_size=self._chunk_size,
            eps=self._eps,
        )

    def _run_layer_kernels(
        self,
        x: torch.Tensor,
        d: int,
        layer_idx: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Hh = self.num_heads[layer_idx]
        Dh = self.hidden_size // Hh
        sf = math.sqrt(Dh)
        B, T, _ = x.shape

        q, k, v, i_tilde, f_raw = self._project(x, d, layer_idx)
        k = k * sf

        f_tilde = f_raw
        if boundaries is not None:
            b = boundaries.to(device=f_tilde.device, dtype=torch.bool).unsqueeze(-1)
            f_tilde = f_tilde.masked_fill(b, _BOUNDARY_RESET_LOGF)

        q_k = q.permute(0, 2, 1, 3).contiguous()
        k_k = k.permute(0, 2, 1, 3).contiguous()
        v_k = v.permute(0, 2, 1, 3).contiguous()
        i_k = i_tilde.permute(0, 2, 1).contiguous()
        f_k = f_tilde.permute(0, 2, 1).contiguous()
        m_k = m.unsqueeze(-1)

        h_k, (C_out, n_out, m_out_k) = self._mlstm_backend(
            q=q_k, k=k_k, v=v_k, i=i_k, f=f_k,
            c_initial=C * sf, n_initial=n * sf, m_initial=m_k,
            return_last_states=True,
        )

        h_out = h_k.permute(0, 2, 1, 3).reshape(B, T, -1)
        m_out = m_out_k.squeeze(-1)
        C_out = C_out / sf
        n_out = n_out / sf

        return h_out, C_out, n_out, m_out

    def _run_layer(
        self,
        x: torch.Tensor,
        d: int,
        layer_idx: int,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self._use_triton_kernels
                and x.is_cuda
                and x.size(1) % self._chunk_size == 0
                and not torch._dynamo.is_compiling()):
            return self._run_layer_kernels(x, d, layer_idx, C, n, m, boundaries)

        if self._use_triton_kernels:
            global _triton_fallback_warned
            if not _triton_fallback_warned:
                if not x.is_cuda:
                    msg = "mLSTM: triton kernels require CUDA tensors. Falling back to native scan."
                else:
                    msg = "mLSTM: falling back to native chunked-parallel scan."
                warnings.warn(msg, stacklevel=3)
                _triton_fallback_warned = True

        return self._run_layer_native(x, d, layer_idx, C, n, m, boundaries=boundaries)

    def forward(
        self,
        input: torch.Tensor,
        state=None,
        boundaries: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the multi-layer mLSTM model.

        Args:
            input: Input tensor of shape (B, T, D) if batch_first else (T, B, D).
            state: Optional previous state (mLSTMState or tuple of states per layer).
            boundaries: Optional boolean mask of shape (B, T) indicating document start boundaries.

        Returns:
            Tuple of (output tensor, final state).
        """
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)
        if not isinstance(state, (tuple, list)):
            state = (state,)

        packed = boundaries is not None
        bounds_mode = get_packed_boundaries_override_mode()
        ckpt_active = bool(self.use_checkpoint and self.training)
        if packed and bounds_mode == PackedBoundariesMode.DISABLE_CKPT_IN_PACKED:
            ckpt_active = False

        final_states: List[mLSTMState] = []
        layer_input = input

        for layer_idx in range(self.num_layers):
            s_l = state[layer_idx]

            if self.num_directions == 1:
                C_dl = s_l.C.squeeze(0)
                n_dl = s_l.n.squeeze(0)
                m_dl = s_l.m.squeeze(0)

                if ckpt_active:
                    out, C_out, n_out, m_out = _torch_checkpoint(
                        self._run_layer, layer_input, 0, layer_idx,
                        C_dl, n_dl, m_dl, boundaries,
                        use_reentrant=False,
                    )
                else:
                    out, C_out, n_out, m_out = self._run_layer(
                        layer_input, 0, layer_idx, C_dl, n_dl, m_dl,
                        boundaries=boundaries,
                    )

                layer_output = out
                final_states.append(mLSTMState(
                    C=C_out.unsqueeze(0),
                    n=n_out.unsqueeze(0),
                    m=m_out.unsqueeze(0),
                ))
            else:
                dir_outputs: List[torch.Tensor] = []
                C_dirs: List[torch.Tensor] = []
                n_dirs: List[torch.Tensor] = []
                m_dirs: List[torch.Tensor] = []

                for d in range(self.num_directions):
                    l_in = torch.flip(layer_input, [1]) if d == 1 else layer_input
                    if d == 1 and boundaries is not None:
                        b_flipped = torch.flip(boundaries, [1])
                        b_d = torch.zeros_like(b_flipped)
                        b_d[:, 1:] = b_flipped[:, :-1]
                    else:
                        b_d = boundaries
                    C_dl = s_l.C[d]
                    n_dl = s_l.n[d]
                    m_dl = s_l.m[d]

                    if ckpt_active:
                        out, C_out, n_out, m_out = _torch_checkpoint(
                            self._run_layer, l_in, d, layer_idx,
                            C_dl, n_dl, m_dl, b_d,
                            use_reentrant=False,
                        )
                    else:
                        out, C_out, n_out, m_out = self._run_layer(
                            l_in, d, layer_idx, C_dl, n_dl, m_dl,
                            boundaries=b_d,
                        )

                    if d == 1:
                        out = torch.flip(out, [1])
                    dir_outputs.append(out)
                    C_dirs.append(C_out)
                    n_dirs.append(n_out)
                    m_dirs.append(m_out)

                layer_output = torch.cat(dir_outputs, dim=-1)
                final_states.append(mLSTMState(
                    C=torch.stack(C_dirs, dim=0),
                    n=torch.stack(n_dirs, dim=0),
                    m=torch.stack(m_dirs, dim=0),
                ))

            if self.drop is not None and layer_idx < self.num_layers - 1:
                layer_output = self.drop(layer_output)
            layer_input = layer_output

        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)

        if self.pack_state:
            ret_state = tuple(final_states)
        else:
            ret_state = final_states[0] if self.num_layers == 1 else tuple(final_states)

        return layer_output, ret_state

    def extra_repr(self) -> str:
        """Returns extra representation string for module display."""
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout}, bidirectional={self.bidirectional}, "
            f"use_checkpoint={self.use_checkpoint}, "
            f"use_triton_kernels={self._use_triton_kernels}, "
            f"chunkwise_kernel={self._chunkwise_kernel!r}, chunk_size={self._chunk_size}"
        )

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamps forget gate bias across all layers and directions."""
        for layer_idx in range(self.num_layers):
            for d in range(self.num_directions):
                fg = getattr(self, f"fgate_{d}_{layer_idx}")
                if fg.bias is not None:
                    fg.bias.data.clamp_(-max_val, max_val)
