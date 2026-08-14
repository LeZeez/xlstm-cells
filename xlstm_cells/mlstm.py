"""
mLSTM: Matrix-memory LSTM cell and layer.

Optimized over the reference implementation:
1. All input projections are fused: Wq/Wk/Wv/Wo -> 1 F.linear, Wi/Wf -> 1 F.linear
   (6 separate GEMM calls -> 2).  Weights stored natively fused (no torch.cat on forward).
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

from ._utils import (
    PackedBoundariesMode,
    get_packed_boundaries_override_mode,
)

try:
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
    )
    _HAS_MLSTM_KERNELS = True
except ImportError:
    _HAS_MLSTM_KERNELS = False

_EPS = 1e-3
_MLSTM_CHUNK_SIZE = 128
_BOUNDARY_RESET_LOGF = -1000.0
_MAX_FORGET_BIAS = 4.0

# Accessible short names for the mlstm_kernels chunkwise triton kernels.
# The shared "chunkwise--triton" prefix is omitted; see mLSTM.chunkwise_kernel.
# Both kernels use exponential input gating (i_prime = exp(i_tilde - m)) with a
# running log-space max state m for numerical stability.  When triton is
# unavailable the native chunked-parallel scan is used — same exp-gate math.
_TRITON_CHUNKWISE_KERNELS = {
    "limit_chunk": "chunkwise--triton_limit_chunk",
    "xl_chunk": "chunkwise--triton_xl_chunk",
}

# Track whether we've already warned about triton fallback
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
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation for testing only. Not used by any module at runtime.

    Step-by-step sequential scan kept for correctness verification in tests.

    Args:
        q:        (B, T, H, Dh)   query projections
        k:        (B, T, H, Dh)   key projections (already scaled by 1/sqrt(Dh))
        v:        (B, T, H, Dh)   value projections
        o:        (B, T, H, Dh)   output gate projections (sigmoid'd)
        i_tilde:  (B, T, H)      input gate pre-activations
        log_f:    (B, T, H)      log(sigmoid(forget pre-activation))
        C:        (B, H, Dh, Dh)  initial matrix memory
        n:        (B, H, Dh)     initial normalizer
        m:        (B, H)         initial stabilizer
        eps:      denominator floor (bounds cancellation amplification)

    Returns:
        outputs: (B, T, Hs)  where Hs = H * Dh
        C_final: (B, H, Dh, Dh)
        n_final: (B, H, Dh)
        m_final: (B, H)
    """
    B, T, H, Dh = q.shape
    hidden_size = H * Dh
    output_list: List[torch.Tensor] = []

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
        denom = torch.maximum(qn * exp_m_safe, torch.exp(m_safe - m)).clamp_min(eps)
        h = ot * ((h_tilde * exp_m_safe[..., None]) / denom[..., None])
        output_list.append(h.reshape(B, hidden_size))

    outputs = torch.stack(output_list, dim=1)
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
    eps: float = _EPS,
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
        eps:      denominator floor. Bounds the output amplification
                  (1/eps) in the pathological cancellation regime where
                  the signed denom_raw sum and the exp-floor both go to
                  zero. Configurable per layer for training stability.

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
    ).clamp_min(eps)
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
    eps: float = _EPS,
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
        eps:                    denominator floor, forwarded to the scan

    Returns:
        outputs, C_final, n_final, m_final  (same signatures as the other scans)
    """
    B, T, H, Dh = q.shape
    hidden_size = H * Dh
    output_chunks: List[torch.Tensor] = []

    C, n, m = C_init, n_init, m_init

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        out_chunk, C, n, m = _mlstm_recurrent_scan_parallel(
            q[:, start:end], k[:, start:end], v[:, start:end],
            o[:, start:end], i_tilde[:, start:end], log_f[:, start:end],
            C, n, m,
            eps=eps,
        )
        output_chunks.append(out_chunk)

    outputs = torch.cat(output_chunks, dim=1)
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

    Weights are stored as fused parameters to avoid torch.cat on every step.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4,
                 bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused: [Wq; Wk; Wv; Wo] as one linear
        self.W_qkvo = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        # Fused: [Wi; Wf] as one linear
        self.W_if = nn.Linear(input_size, 2 * num_heads, bias=bias)

        self._sf = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        HS = self.hidden_size
        std = 1.0 / math.sqrt(HS)
        NH = self.num_heads
        # W_qkvo: (4*HS, input_size), rows: [Wq | Wk | Wv | Wo]
        w = self.W_qkvo.weight.data
        nn.init.normal_(w[:HS], std=std)          # Wq
        nn.init.normal_(w[HS:2*HS], std=std)      # Wk
        nn.init.normal_(w[2*HS:3*HS], std=std)    # Wv
        nn.init.xavier_normal_(w[3*HS:4*HS])      # Wo
        if self.W_qkvo.bias is not None:
            nn.init.zeros_(self.W_qkvo.bias)
        # W_if: (2*NH, input_size), rows: [Wi | Wf]
        wif = self.W_if.weight.data
        nn.init.normal_(wif[:NH], std=1e-2)       # Wi
        nn.init.zeros_(wif[NH:])                  # Wf
        if self.W_if.bias is not None:
            nn.init.zeros_(self.W_if.bias)
            # Forget gate is the 2nd chunk [Wi | Wf]
            self.W_if.bias.data[NH:].fill_(3.0)

    def init_state(self, batch_size: int, device=None, dtype=None) -> mLSTMState:
        return mLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def forward(self, x_t: torch.Tensor, state: mLSTMState) -> Tuple[torch.Tensor, mLSTMState]:
        B = x_t.size(0)
        H, Dh = self.num_heads, self.head_dim

        qkvo = self.W_qkvo(x_t)
        q, k, v, o_raw = qkvo.view(B, 4, H, Dh).unbind(1)
        k = k / self._sf
        o = torch.sigmoid(o_raw)

        iff = self.W_if(x_t)
        i_tilde, f_tilde = iff.view(B, 2, H).unbind(1)
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

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamp the forget-gate bias to [-max_val, max_val].

        Call after ``optimizer.step()`` to prevent the forget bias from
        drifting into saturation (logsigmoid(b_f) ≈ 0), which causes
        the log-normalizer m to grow unboundedly and can make the
        boundary reset ineffective.
        """
        NH = self.num_heads
        if self.W_if.bias is not None:
            self.W_if.bias.data[NH:].clamp_(-max_val, max_val)


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
        pack_state:      if True, states are always packed in a tuple (default True)
        use_checkpoint:  if True, use activation checkpointing (default False)
        use_triton_kernels:  if True, use mlstm_kernels triton backend for
                             the recurrent scan (default True if available).
                             Falls back to chunked PyTorch parallel scan if
                             unavailable or if sequence length is not a multiple
                             of chunk_size (default 128).
        chunkwise_kernel:  triton chunkwise kernel for the recurrent scan.
                             Both use exponential input gating
                             (i_prime = exp(i_tilde - m)) with a running
                             log-space max state ``m`` for numerical stability.
                             When triton is unavailable (missing mlstm_kernels,
                             CPU input, non-divisible sequence length, or
                             torch.compile) the native chunked-parallel scan is
                             used — same exp-gate math, no semantic change.
                               "xl_chunk"     TFLA kernel optimized for larger
                                              chunk sizes and lower backward
                                              memory usage (default)
                               "limit_chunk"  standard TFLA chunkwise kernel
        chunk_size:      chunk size for the chunkwise kernel (default 128; must
                         divide the sequence length when triton is used)

    .. hint::
        **Triton kernels vs. activation checkpointing**
        The triton backend computes the recurrence chunk-wise and keeps peak
        activation memory far below the native chunked-parallel scan.  On top
        of that, ``use_checkpoint=True`` still cuts retained activation memory
        by roughly half (measured ~45% on mLSTMBlock at B=1, T=1024,
        expanded=2048) at the cost of recomputing the sequence during the
        backward pass.  Prefer checkpointing when VRAM-bound, omit it when
        compute-bound.
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
        chunkwise_kernel: str = "xl_chunk",
        chunk_size: int = _MLSTM_CHUNK_SIZE,
        eps: Optional[float] = None,
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
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")
        self._chunk_size = chunk_size
        self._chunkwise_kernel = chunkwise_kernel
        self._mlstm_backend = None

        if eps is None:
            eps = _EPS
        if (
            not isinstance(eps, (int, float))
            or isinstance(eps, bool)
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError("eps must be a positive finite float")
        self._eps = float(eps)

        if chunkwise_kernel not in _TRITON_CHUNKWISE_KERNELS:
            raise ValueError(
                f"mLSTM: unknown chunkwise_kernel {chunkwise_kernel!r}, "
                f"choose one of {sorted(_TRITON_CHUNKWISE_KERNELS)}."
            )

        if use_triton_kernels and not _HAS_MLSTM_KERNELS:
            global _triton_fallback_warned
            if not _triton_fallback_warned:
                warnings.warn(
                    "mLSTM: use_triton_kernels=True but 'mlstm_kernels' is not "
                    "installed. Falling back to the native chunked-parallel scan, "
                    "which is significantly slower. Install with "
                    "'pip install mlstm_kernels' (requires a CUDA GPU).",
                    stacklevel=3,
                )
                _triton_fallback_warned = True

        for d in range(self.num_directions):
            for l in range(num_layers):
                assert hidden_size % num_heads[l] == 0, (
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_heads[{l}] ({num_heads[l]})."
                )
                layer_input = input_size if l == 0 else hidden_size * self.num_directions
                Hh = num_heads[l]

                # Fused: [Wq; Wk; Wv; Wo]
                setattr(self, f"W_qkvo_{d}_{l}",
                        nn.Linear(layer_input, 4 * hidden_size, bias=bias))
                # Fused: [Wi; Wf]
                setattr(self, f"W_if_{d}_{l}",
                        nn.Linear(layer_input, 2 * Hh, bias=bias))

        self.reset_parameters()
        if self._use_triton_kernels:
            self._init_triton_backend()

    def _init_triton_backend(self):
        # eps bounds the triton kernel's denominator floor. The kernel's
        # default (1e-6) allows up to 1,000,000x output amplification when
        # the signed denom_raw sum cancels at high stabilizer m. Passing a
        # larger eps caps the amplification; the value is configurable per
        # instance via the eps= constructor arg.
        config = mLSTMBackendConfig(
            chunkwise_kernel=_TRITON_CHUNKWISE_KERNELS[self._chunkwise_kernel],
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            mode="train",
            chunk_size=self._chunk_size,
            return_last_states=True,
            autocast_kernel_dtype="float32",
            eps=self._eps,
        )
        self._mlstm_backend = mLSTMBackend(config=config)

    def reset_parameters(self) -> None:
        HS = self.hidden_size
        for d in range(self.num_directions):
            for layer_idx in range(self.num_layers):
                Hh = self.num_heads[layer_idx]
                std = 1.0 / math.sqrt(HS)

                # W_qkvo: (4*HS, in), rows: [Wq | Wk | Wv | Wo]
                W_qkvo = getattr(self, f"W_qkvo_{d}_{layer_idx}")
                w = W_qkvo.weight.data
                nn.init.normal_(w[:HS], std=std)          # Wq
                nn.init.normal_(w[HS:2*HS], std=std)      # Wk
                nn.init.normal_(w[2*HS:3*HS], std=std)    # Wv
                nn.init.xavier_normal_(w[3*HS:4*HS])      # Wo
                if W_qkvo.bias is not None:
                    nn.init.zeros_(W_qkvo.bias)

                # W_if: (2*Hh, in), rows: [Wi | Wf]
                W_if = getattr(self, f"W_if_{d}_{layer_idx}")
                wif = W_if.weight.data
                nn.init.normal_(wif[:Hh], std=1e-2)       # Wi
                nn.init.zeros_(wif[Hh:])                  # Wf
                if W_if.bias is not None:
                    nn.init.zeros_(W_if.bias)
                    # Forget gate bias = 3.0 for stability
                    W_if.bias.data[Hh:].fill_(3.0)

    def init_state(self, batch_size: int, device=None, dtype=None):
        states = []
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
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
        self, x: torch.Tensor, d: int, layer_idx: int, *, apply_activations: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """Project input sequence through fused weight matrices.

        Args:
            x: (B, T, in_features)
            d: direction index
            layer_idx: layer index
            apply_activations: if True, apply sigmoid to output gate and
                logsigmoid to forget gate.  Set False for triton backend
                which applies its own activations.

        Returns:
            q, k, v, o, i_tilde, f_or_logf
        """
        Hh = self.num_heads[layer_idx]
        Dh = self.hidden_size // Hh
        B, T, _ = x.shape
        sf = math.sqrt(Dh)

        W_qkvo = getattr(self, f"W_qkvo_{d}_{layer_idx}")
        qkvo = W_qkvo(x)
        q, k, v, o_raw = qkvo.view(B, T, 4, Hh, Dh).unbind(2)
        k = k / sf
        if apply_activations:
            o = torch.sigmoid(o_raw)
        else:
            o = o_raw

        W_if = getattr(self, f"W_if_{d}_{layer_idx}")
        iff = W_if(x)
        i_tilde, f_raw = iff.view(B, T, 2, Hh).unbind(2)
        if apply_activations:
            f_out = F.logsigmoid(f_raw)
        else:
            f_out = f_raw

        return q, k, v, o, i_tilde, f_out

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

        # Warn once if triton was requested but can't be used
        if self._use_triton_kernels:
            global _triton_fallback_warned
            if not _triton_fallback_warned:
                if not x.is_cuda:
                    msg = (
                        "mLSTM: triton kernels require CUDA tensors but input "
                        "is on CPU. Falling back to native chunked-parallel scan."
                    )
                elif x.size(1) % self._chunk_size != 0:
                    msg = (
                        f"mLSTM: triton kernels requested but seq_len={x.size(1)} "
                        f"is not divisible by chunk_size={self._chunk_size}. "
                        f"Falling back to native chunked-parallel scan. "
                        f"Pad to a multiple of {self._chunk_size} for triton acceleration."
                    )
                elif torch._dynamo.is_compiling():
                    msg = (
                        "mLSTM: triton kernels disabled under torch.compile tracing "
                        "(mlstm_kernels' triton_limit_chunk kernels crash inside "
                        "Inductor). Falling back to native chunked-parallel scan, "
                        "which is significantly slower. Do NOT wrap the model in "
                        "torch.compile if you want kernel acceleration."
                    )
                else:
                    msg = (
                        "mLSTM: triton kernels requested but unavailable. "
                        "Falling back to native chunked-parallel scan."
                    )
                warnings.warn(msg, stacklevel=3)
                _triton_fallback_warned = True

        return self._run_layer_native(x, d, layer_idx, C, n, m, boundaries=boundaries)

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
        q, k, v, o, i_tilde, log_f = self._project_sequence(
            x, d, layer_idx, apply_activations=True,
        )
        if boundaries is not None:
            # The native scan takes the ALREADY-LOG-SIGMOIDED forget gate;
            # override `log_f` so the recurrence sees an effectively-zero
            # cumulative forgetting factor at boundary positions.  The
            # constant must be large enough that the boundary reset is
            # unconditional even when the log-normalizer m has drifted
            # high (e.g. due to saturated forget biases at high LR).
            # With -1000 the reset holds for any m < i_tilde + 1000,
            # which covers all realistic scenarios.
            log_f = log_f.masked_fill(
                boundaries.to(device=log_f.device, dtype=torch.bool).unsqueeze(-1),
                _BOUNDARY_RESET_LOGF,
            )
        return _mlstm_recurrent_scan_parallel_chunked(
            q, k, v, o, i_tilde, log_f, C, n, m,
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

        q, k, v, o_raw, i_tilde, f_tilde = self._project_sequence(
            x, d, layer_idx, apply_activations=False,
        )

        # The triton kernels apply qk_scale = 1/sqrt(Dh) internally and store
        # the C/n states with UNSCALED keys (reference xLSTM math).  The native
        # scan pre-scales k instead.  Undo the pre-scale here and convert the
        # kernel's states back to the native convention at the boundary so that
        # both paths produce identical outputs and interchangeable states
        # (TBPTT can mix kernel and native segments freely).
        k = k * sf

        # Pack-aware state reset: override f_tilde at boundary positions
        # so logsigmoid(f_tilde) ≈ _BOUNDARY_RESET_LOGF contributes a
        # near-zero cumulative forgetting factor to the chunkwise
        # recurrence at those positions.  The constant is large enough
        # to guarantee an unconditional reset even when the log-normalizer
        # m has drifted high (e.g. due to saturated forget biases at
        # high LR).  See PACKED_FORGET_RESET_RESULTS.md for the math.
        if boundaries is not None:
            b = boundaries.to(device=f_tilde.device, dtype=torch.bool).unsqueeze(-1)
            f_tilde = f_tilde.masked_fill(b, _BOUNDARY_RESET_LOGF)

        # Permute (B,T,H,Dh) -> (B,H,T,Dh) for triton kernels.
        # Individual permute+contiguous avoids the intermediate stacked tensor
        # that torch.stack([q,k,v]).permute().contiguous() would create.
        q_k = q.permute(0, 2, 1, 3).contiguous()
        k_k = k.permute(0, 2, 1, 3).contiguous()
        v_k = v.permute(0, 2, 1, 3).contiguous()

        # Gates: (B, T, H) -> (B, H, T)
        i_k = i_tilde.permute(0, 2, 1).contiguous()
        f_k = f_tilde.permute(0, 2, 1).contiguous()

        m_k = m.unsqueeze(-1)

        h_k, (C_out, n_out, m_out_k) = self._mlstm_backend(
            q=q_k, k=k_k, v=v_k, i=i_k, f=f_k,
            c_initial=C * sf, n_initial=n * sf, m_initial=m_k,
            return_last_states=True,
        )

        h_out = h_k.permute(0, 2, 1, 3).reshape(B, T, -1)
        o = torch.sigmoid(o_raw.reshape(B, T, Hh * Dh))
        h_out = o * h_out

        m_out = m_out_k.squeeze(-1)
        C_out = C_out / sf
        n_out = n_out / sf

        return h_out, C_out, n_out, m_out

    def forward(
        self,
        input: torch.Tensor,
        state=None,
        boundaries: Optional[torch.Tensor] = None,
    ):
        """Run mLSTM recurrence over `input`.

        Args:
            input: (B, T, input_size) when batch_first=True.
            state: prior `mLSTMState`(s); `None` initialises at zero.
            boundaries: optional bool tensor of shape (B, T) marking the
                FIRST position of every packed document. At those
                positions the raw forget-gate is forced to
                _BOUNDARY_RESET_LOGF (-1000), killing the cumulative
                carry into the chunkwise recurrence from that point
                onward (effectively resetting the recurrent state).
                Enables sequence packing without padding pollution.

        See ``PACKED_FORGET_RESET_RESULTS.md`` for the math and the
        interplay with activation checkpointing.
        """
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)

        if not isinstance(state, tuple):
            state = (state,)

        # When the user passes `boundaries`, decide how to interact with
        # activation checkpointing globally. The chosen mode lives in
        # xlstm_cells._utils.PackedBoundariesMode.
        packed = boundaries is not None
        bounds_mode = get_packed_boundaries_override_mode()
        ckpt_active = bool(self.use_checkpoint and self.training)
        if packed and bounds_mode == PackedBoundariesMode.DISABLE_CKPT_IN_PACKED:
            ckpt_active = False
        ckpt_use_reentrant = False
        # PackedBoundariesMode.USE_REENTRANT_CKPT is no longer applied; the
        # boundaries override is compatible with use_reentrant=False on modern
        # PyTorch, and non-reentrant is more robust under detached-input /
        # frozen-embedding TBPTT (same VRAM envelope, no silent grad loss).
        # See tests/test_packed_non_reentrant.py.

        # Pre-allocate output state buffers (one per layer) to avoid
        # creating new tensors per layer per forward pass.  These are fresh
        # tensors that participate in autograd normally and are safe for
        # TBPTT (detach_states / zero_rows work identically).
        out_states: List[mLSTMState] = []
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = self.hidden_size // Hh
            D = self.num_directions
            out_states.append(mLSTMState(
                C=torch.empty(D, B, Hh, Dh, Dh, device=input.device, dtype=input.dtype),
                n=torch.empty(D, B, Hh, Dh, device=input.device, dtype=input.dtype),
                m=torch.empty(D, B, Hh, device=input.device, dtype=input.dtype),
            ))

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
                        use_reentrant=ckpt_use_reentrant,
                    )
                else:
                    out, C_out, n_out, m_out = self._run_layer(
                        layer_input, 0, layer_idx, C_dl, n_dl, m_dl,
                        boundaries=boundaries,
                    )

                layer_output = out
                out_states[layer_idx].C[0].copy_(C_out)
                out_states[layer_idx].n[0].copy_(n_out)
                out_states[layer_idx].m[0].copy_(m_out)
            else:
                dir_outputs: List[torch.Tensor] = []

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

                    C_dl = s_l.C[d]
                    n_dl = s_l.n[d]
                    m_dl = s_l.m[d]

                    if ckpt_active:
                        out, C_out, n_out, m_out = _torch_checkpoint(
                            self._run_layer, layer_input, d, layer_idx,
                            C_dl, n_dl, m_dl, b_d,
                            use_reentrant=ckpt_use_reentrant,
                        )
                    else:
                        out, C_out, n_out, m_out = self._run_layer(
                            layer_input, d, layer_idx, C_dl, n_dl, m_dl,
                            boundaries=b_d,
                        )

                    if d == 1:
                        out = torch.flip(out, [1])

                    dir_outputs.append(out)
                    out_states[layer_idx].C[d].copy_(C_out)
                    out_states[layer_idx].n[d].copy_(n_out)
                    out_states[layer_idx].m[d].copy_(m_out)

                layer_output = torch.cat(dir_outputs, dim=-1)

            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)

        result = tuple(out_states)
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
        for d in range(self.num_directions):
            for layer_idx in range(self.num_layers):
                Hh = self.num_heads[layer_idx]
                W_if = getattr(self, f"W_if_{d}_{layer_idx}")
                if W_if.bias is not None:
                    W_if.bias.data[Hh:].clamp_(-max_val, max_val)

    def __repr__(self) -> str:
        return (
            f"mLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}, "
            f"batch_first={self.batch_first}, use_checkpoint={self.use_checkpoint}, "
            f"use_triton_kernels={self._use_triton_kernels})"
        )
