r"""
xLSTM Blocks: Official paper-compliant residual blocks (Beck et al., 2024).

Architectures:
    mLSTMBlock (Pre Up-Projection, Figure 11):
        x -> LayerNorm -+-> up_proj -> split -> z -> SiLU(z) -------------------------+
                        |                                                             |
                        +-> x_mlstm -+-> Conv1d -> SiLU -> q_proj, k_proj             |
                                     |                    \                           |
                                     +------------------> v_proj                      |
                                                            \                         |
                                                             mLSTM -> MultiHeadLN -> (+) -> (*) -> down_proj + x
                                                                      (Token-wise)    ^      |
                                                                                      |      |
                                                              x_conv_act * LSkip -----+------+

    sLSTMBlock (Post Up-Projection, Figure 10):
        x -> LayerNorm -+-> Conv1d -> SiLU -> igate, fgate (convolved)
                        |
                        +-------------------> zgate, ogate (unconvolved)
                                                \
                                                 sLSTM -> MultiHeadLN + x -> LayerNorm -> GeGLU MLP + x
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from ._utils import (
    PackedBoundariesMode,
    get_packed_boundaries_override_mode,
)
from .components.conv import CausalConv1d
from .components.feedforward import GatedFeedForward
from .components.init import bias_linspace_init_, small_init_init_, wang_init_
from .components.ln import LayerNorm, MultiHeadLayerNorm
from .mlstm import _MAX_FORGET_BIAS, _MLSTM_CHUNK_SIZE, _BOUNDARY_RESET_LOGF, _EPS, mLSTMState
from .slstm import sLSTM, sLSTMState, _slstm_scan_sequential, _sLSTMCudaFunction

try:
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
    )
    _HAS_MLSTM_KERNELS = True
except ImportError:
    _HAS_MLSTM_KERNELS = False

_TRITON_CHUNKWISE_KERNELS = {
    "limit_chunk": "chunkwise--triton_limit_chunk",
    "xl_chunk": "chunkwise--triton_xl_chunk",
}


# ---------------------------------------------------------------------------
# mLSTM Block (Figure 11)
# ---------------------------------------------------------------------------

class mLSTMBlock(nn.Module):
    """Paper-compliant mLSTM block with pre up-projection (Figure 11).

    Args:
        d_model: Model hidden dimension.
        expand_factor: Expansion factor for inner projection (expanded = d_model * expand_factor). Default: 2.
        num_heads: Number of parallel matrix-memory heads. Default: 4.
        conv_kernel: 1D causal convolution kernel size. 0 disables convolution. Default: 4.
        dropout: Output dropout probability. Default: 0.0.
        bias: Whether linear projection layers use bias. Default: False.
        use_checkpoint: Whether to apply activation checkpointing. Default: False.
        use_triton_kernels: Whether to use mlstm_kernels Triton backend when available. Default: True.
        chunkwise_kernel: Triton chunk kernel name ("limit_chunk" or "xl_chunk"). Default: "xl_chunk".
        chunk_size: Sequence chunk size for chunked scan. Default: 128.
        eps: Denominator epsilon constant for stabilizer numerical stability. Default: 1e-6.
        num_blocks: Total number of stacked blocks in the model (for Wang init scaling). Default: 1.
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: Optional[float] = None,
        num_heads: int = 4,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
        use_checkpoint: bool = False,
        use_triton_kernels: bool = True,
        chunkwise_kernel: str = "xl_chunk",
        chunk_size: int = _MLSTM_CHUNK_SIZE,
        eps: Optional[float] = None,
        num_blocks: int = 1,
        hidden_size: Optional[int] = None,
    ):
        """Initializes paper-compliant mLSTMBlock."""
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError(f"mLSTMBlock: d_model must be a positive integer, got {d_model}")
        if expand_factor is not None and hidden_size is not None:
            raise ValueError(
                "mLSTMBlock: conflicting arguments: cannot specify both 'expand_factor' and 'hidden_size'. "
                "Specify only one."
            )
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"mLSTMBlock: num_heads must be a strictly positive integer, got {num_heads}")
        if hidden_size is not None:
            if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
                raise ValueError(f"mLSTMBlock: hidden_size must be a strictly positive integer, got {hidden_size}")
            expanded = hidden_size
        elif expand_factor is not None:
            if not isinstance(expand_factor, (int, float)) or isinstance(expand_factor, bool) or not math.isfinite(expand_factor) or expand_factor <= 0:
                raise ValueError(f"mLSTMBlock: expand_factor must be a positive finite number, got {expand_factor}")
            expanded = round(expand_factor * d_model)
        else:
            expanded = 2 * d_model  # Paper default (factor 2)

        if expanded <= 0:
            raise ValueError(f"mLSTMBlock: resolved hidden dimension must be strictly positive, got {expanded}")
        if expanded % num_heads != 0:
            raise ValueError(
                f"mLSTMBlock: hidden dimension ({expanded}) must be divisible by num_heads ({num_heads})"
            )
        if chunkwise_kernel not in _TRITON_CHUNKWISE_KERNELS:
            raise ValueError(
                f"mLSTMBlock: unknown chunkwise_kernel {chunkwise_kernel!r}, "
                f"expected one of {list(_TRITON_CHUNKWISE_KERNELS.keys())}"
            )
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a strictly positive integer, got {chunk_size}")

        if eps is None:
            eps = _EPS
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite float")

        self.d_model = d_model
        self.expanded = expanded
        self.num_heads = num_heads
        self.head_dim = expanded // num_heads
        self.conv_kernel = conv_kernel
        self.use_checkpoint = use_checkpoint
        self._use_triton_kernels = use_triton_kernels and _HAS_MLSTM_KERNELS
        self._chunkwise_kernel = chunkwise_kernel
        self._chunk_size = chunk_size
        self._eps = float(eps)

        self.ln = LayerNorm(d_model, bias=False)
        self.fused_proj = nn.Linear(d_model, 2 * expanded, bias=bias)

        if conv_kernel > 0:
            self.conv = CausalConv1d(expanded, kernel_size=conv_kernel, bias=bias)
        else:
            self.conv = None

        self.q_proj = nn.Linear(expanded, expanded, bias=bias)
        self.k_proj = nn.Linear(expanded, expanded, bias=bias)
        self.v_proj = nn.Linear(expanded, expanded, bias=bias)

        self.igate = nn.Linear(3 * expanded, num_heads, bias=True)
        self.fgate = nn.Linear(3 * expanded, num_heads, bias=True)

        self.gn = MultiHeadLayerNorm(ndim=expanded, weight=True, bias=False, eps=1e-5)
        self.learnable_skip = nn.Parameter(torch.ones(expanded))

        self.down_proj = nn.Linear(expanded, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._mlstm_backend = None
        if self._use_triton_kernels:
            self._init_triton_backend()

        self.reset_parameters(num_blocks=num_blocks)

    def _init_triton_backend(self) -> None:
        """Initializes Triton kernel backend configuration."""
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

    def reset_parameters(self, num_blocks: int = 1) -> None:
        """Initializes weights using official paper init schemes."""
        small_init_init_(self.fused_proj.weight, dim=self.d_model)
        if self.fused_proj.bias is not None: nn.init.zeros_(self.fused_proj.bias)

        small_init_init_(self.q_proj.weight, dim=self.expanded)
        small_init_init_(self.k_proj.weight, dim=self.expanded)
        small_init_init_(self.v_proj.weight, dim=self.expanded)
        if self.q_proj.bias is not None: nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None: nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None: nn.init.zeros_(self.v_proj.bias)

        nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.4, end=6.0)
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

        nn.init.ones_(self.learnable_skip)

        wang_init_(self.down_proj.weight, dim=self.expanded, num_blocks=num_blocks)
        if self.down_proj.bias is not None: nn.init.zeros_(self.down_proj.bias)

        self.ln.reset_parameters()
        self.gn.reset_parameters()
        if self.conv is not None:
            self.conv.reset_parameters()

    def init_state(self, batch_size: int, device=None, dtype=None) -> mLSTMState:
        """Initializes zero state object for mLSTM recurrence."""
        return mLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def _run_core_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i_tilde: torch.Tensor,
        f_raw: torch.Tensor,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes native PyTorch parallel chunked scan."""
        from .mlstm import _mlstm_recurrent_scan_parallel_chunked
        log_f = F.logsigmoid(f_raw)
        if boundaries is not None:
            b = boundaries.to(device=log_f.device, dtype=torch.bool).unsqueeze(-1)
            log_f = log_f.masked_fill(b, _BOUNDARY_RESET_LOGF)
        return _mlstm_recurrent_scan_parallel_chunked(
            q, k, v, i_tilde, log_f, C, n, m,
            chunk_size=self._chunk_size,
            eps=self._eps,
        )

    def _run_core_kernels(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i_tilde: torch.Tensor,
        f_raw: torch.Tensor,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes official Triton mLSTM kernel."""
        B, T, H, Dh = q.shape
        sf = math.sqrt(Dh)
        k_scaled = k * sf

        f_tilde = f_raw
        if boundaries is not None:
            b = boundaries.to(device=f_tilde.device, dtype=torch.bool).unsqueeze(-1)
            f_tilde = f_tilde.masked_fill(b, _BOUNDARY_RESET_LOGF)

        q_k = q.permute(0, 2, 1, 3).contiguous()
        k_k = k_scaled.permute(0, 2, 1, 3).contiguous()
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

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[mLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        """Forward pass through the mLSTM block.

        Args:
            x: Input tensor of shape (B, T, d_model).
            state: Optional previous mLSTMState. If None, initialized to zeros.
            boundaries: Optional boolean mask of shape (B, T) indicating document start boundaries
                in packed sequences. When True at position (b, t), resets the recurrent matrix
                memory (C, n, m) to prevent cross-document attention and gradient leakage.

        Returns:
            Tuple of (output_tensor, new_state).
        """
        B, T, _ = x.shape
        residual = x

        if state is None:
            state = self.init_state(B, device=x.device, dtype=x.dtype)

        x_norm = self.ln(x)
        x_mlstm, z = self.fused_proj(x_norm).split(self.expanded, dim=-1)

        # Causal Conv Branch
        if self.conv is not None:
            x_conv = self.conv(x_mlstm)
            x_conv_act = F.silu(x_conv)
        else:
            x_conv_act = x_mlstm

        # Projections: q, k from conv; v from unconvolved x_mlstm
        q_raw = self.q_proj(x_conv_act)
        k_raw = self.k_proj(x_conv_act)
        v_raw = self.v_proj(x_mlstm)

        # Gates from [q, k, v]
        if_input = torch.cat([q_raw, k_raw, v_raw], dim=-1)
        i_tilde = self.igate(if_input)
        f_raw = self.fgate(if_input)

        q = q_raw.view(B, T, self.num_heads, self.head_dim)
        k = (k_raw / math.sqrt(self.head_dim)).view(B, T, self.num_heads, self.head_dim)
        v = v_raw.view(B, T, self.num_heads, self.head_dim)

        # Recurrence
        C_in = state.C.squeeze(0) if state.C.dim() == 5 else state.C
        n_in = state.n.squeeze(0) if state.n.dim() == 4 else state.n
        m_in = state.m.squeeze(0) if state.m.dim() == 3 else state.m

        use_triton = (
            self._use_triton_kernels
            and x.is_cuda
            and T % self._chunk_size == 0
            and not torch._dynamo.is_compiling()
        )

        if use_triton:
            h_lstm, C_out, n_out, m_out = self._run_core_kernels(
                q, k, v, i_tilde, f_raw, C_in, n_in, m_in, boundaries=boundaries
            )
        else:
            h_lstm, C_out, n_out, m_out = self._run_core_native(
                q, k, v, i_tilde, f_raw, C_in, n_in, m_in, boundaries=boundaries
            )

        # Token-wise MultiHeadLayerNorm
        h_norm = self.gn(h_lstm, num_heads=self.num_heads)

        # Multiplicative Learnable Skip
        h_skip = h_norm + (self.learnable_skip * x_conv_act)

        # Outer Swish / SiLU Gating
        h_gated = h_skip * F.silu(z)

        out = self.dropout(self.down_proj(h_gated)) + residual
        new_state = mLSTMState(C_out, n_out, m_out)
        return out, new_state

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamps forget gate bias to prevent numerical saturation."""
        if self.fgate.bias is not None:
            self.fgate.bias.data.clamp_(-max_val, max_val)


# ---------------------------------------------------------------------------
# sLSTM Block (Figure 10)
# ---------------------------------------------------------------------------

class sLSTMBlock(nn.Module):
    """Paper-compliant sLSTM block with post up-projection (Figure 10).

    Args:
        d_model: Model hidden dimension.
        num_heads: Number of scalar-memory heads. Default: 4.
        conv_kernel: 1D causal convolution kernel size. 0 disables convolution. Default: 4.
        mlp_factor: Multiplier for feedforward hidden dimension. Default: 4.0 / 3.0.
        dropout: Output dropout probability. Default: 0.0.
        bias: Whether linear projection layers use bias. Default: True.
        backend: sLSTM backend ("vanilla" or "cuda"). Default: "vanilla".
        use_checkpoint: Whether to apply activation checkpointing. Default: False.
        fast_mode: Whether to use compiled chunking for vanilla backend. Default: False.
        fast_chunk_size: Chunk size for fast_mode compilation. Default: 32.
        num_blocks: Total number of stacked blocks in the model. Default: 1.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        conv_kernel: int = 4,
        mlp_factor: Optional[float] = None,
        dropout: float = 0.0,
        bias: bool = True,
        backend: str = "vanilla",
        use_checkpoint: bool = False,
        fast_mode: bool = False,
        fast_chunk_size: int = 32,
        num_blocks: int = 1,
        hidden_size: Optional[int] = None,
    ):
        """Initializes paper-compliant sLSTMBlock."""
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError(f"sLSTMBlock: d_model must be a positive integer, got {d_model}")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"sLSTMBlock: num_heads must be a positive integer, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"sLSTMBlock: d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if backend == "cuda" and fast_mode:
            raise ValueError("sLSTMBlock: backend='cuda' cannot be combined with fast_mode=True.")
        if backend not in ("vanilla", "cuda"):
            raise ValueError(f"sLSTMBlock: unknown backend '{backend}'. Must be 'vanilla' or 'cuda'.")
        if fast_mode:
            if not isinstance(fast_chunk_size, int) or isinstance(fast_chunk_size, bool) or fast_chunk_size <= 0:
                raise ValueError(
                    f"sLSTMBlock: fast_chunk_size must be a strictly positive integer when fast_mode=True, got {fast_chunk_size}."
                )

        if mlp_factor is not None and hidden_size is not None:
            raise ValueError(
                "sLSTMBlock: conflicting arguments: cannot specify both 'mlp_factor' and 'hidden_size'. "
                "Specify only one."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.backend = backend
        self.use_checkpoint = use_checkpoint
        self.fast_mode = fast_mode
        self.fast_chunk_size = fast_chunk_size

        self.ln = LayerNorm(d_model, bias=False)

        if conv_kernel > 0:
            self.conv = CausalConv1d(d_model, kernel_size=conv_kernel, bias=bias)
        else:
            self.conv = None

        # Decoupled Figure 10 projections: (i, f) from conv; (z, o) from unconvolved
        self.W_if = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.W_zo = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.R_fused = nn.Parameter(torch.empty(num_heads, self.head_dim, 4 * self.head_dim))

        self.gn = MultiHeadLayerNorm(ndim=d_model, weight=True, bias=False, eps=1e-5)

        # Post-sLSTM GeGLU Feedforward
        self.ffn_norm = LayerNorm(d_model, bias=False)
        self.ffn = GatedFeedForward(
            d_model=d_model,
            proj_factor=mlp_factor if hidden_size is None else None,
            proj_up_dim=hidden_size,
            act_fn="gelu",
            dropout=dropout,
            bias=bias,
            num_blocks=num_blocks,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._compiled_scans = {}
        self._cuda_kernel = None
        self._cuda_funcs = {}
        if backend == "cuda":
            self._init_cuda_backend()

        self.reset_parameters(num_blocks=num_blocks)

    def _init_cuda_backend(self) -> None:
        """Compiles and loads the official sLSTM CUDA C++ extension."""
        try:
            from .cuda.cuda_init import load, get_slstm_cuda_sources
            sources = get_slstm_cuda_sources()
            self._cuda_kernel = load(name="slstm_cuda", sources=sources)
        except (RuntimeError, OSError) as e:
            warnings.warn(f"sLSTMBlock: failed to compile CUDA kernel ({e}). Falling back to backend='vanilla'.", stacklevel=2)
            self.backend = "vanilla"

    def reset_parameters(self, num_blocks: int = 1) -> None:
        """Resets all block layer parameters."""
        Dh = self.head_dim
        HS = self.d_model
        NH = self.num_heads

        self.ln.reset_parameters()
        if self.conv is not None:
            self.conv.reset_parameters()

        w_if = self.W_if.weight.data
        small_init_init_(w_if[:HS], dim=HS)
        small_init_init_(w_if[HS:2*HS], dim=HS)
        if self.W_if.bias is not None:
            nn.init.zeros_(self.W_if.bias)
            nn.init.normal_(self.W_if.bias.data[:HS], mean=0.0, std=0.1)
            f_biases = torch.linspace(3.4, 6.0, NH).unsqueeze(-1).expand(NH, Dh).reshape(-1)
            self.W_if.bias.data[HS:2*HS].copy_(f_biases)

        w_zo = self.W_zo.weight.data
        small_init_init_(w_zo[:HS], dim=HS)
        small_init_init_(w_zo[HS:2*HS], dim=HS)
        if self.W_zo.bias is not None:
            nn.init.zeros_(self.W_zo.bias)

        R = self.R_fused.data
        for h in range(NH):
            for g in range(4):
                tmp = torch.empty_like(R[h, :, g*Dh:(g+1)*Dh])
                nn.init.orthogonal_(tmp)
                R[h, :, g*Dh:(g+1)*Dh] = tmp

        self.gn.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def init_state(self, batch_size: int, device=None, dtype=None) -> sLSTMState:
        """Initializes zero state object for sLSTM recurrence."""
        return sLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def _get_compiled_scan(self, chunk_size: int):
        """Returns or compiles torch.compile scan kernel for the specified chunk size."""
        if chunk_size not in self._compiled_scans:
            def scan_chunk(all_in, R, c, n, m, h, b=None):
                return _slstm_scan_sequential(all_in, R, c, n, m, h, b)
            self._compiled_scans[chunk_size] = torch.compile(scan_chunk, dynamic=False)
        return self._compiled_scans[chunk_size]

    def _run_core_vanilla(
        self,
        all_in: torch.Tensor,
        c: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        h: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs sLSTM recurrence using the vanilla PyTorch scan backend."""
        B, T, _, _ = all_in.shape
        if not self.fast_mode or T <= self.fast_chunk_size:
            return _slstm_scan_sequential(all_in, self.R_fused, c, n, m, h, boundaries)

        C = self.fast_chunk_size
        fn_c = self._get_compiled_scan(C)
        b_chunks = []
        if boundaries is not None:
            for i in range(0, T, C):
                b_chunks.append(boundaries[:, i:min(i+C, T)])

        outputs = []
        n_chunks = math.ceil(T / C)
        for idx in range(n_chunks):
            start = idx * C
            end = min(start + C, T)
            chunk_in = all_in[:, start:end]
            b_chunk = b_chunks[idx] if boundaries is not None else None

            if (end - start) == C:
                out_chunk, c, n, m, h = fn_c(chunk_in, self.R_fused, c, n, m, h, b_chunk)
            else:
                out_chunk, c, n, m, h = _slstm_scan_sequential(chunk_in, self.R_fused, c, n, m, h, b_chunk)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=1), c, n, m, h

    def _run_core_cuda(
        self,
        all_in: torch.Tensor,
        c: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs sLSTM recurrence using official CUDA backend."""
        B, T, Hh, _ = all_in.shape
        Dh = self.head_dim
        HS = self.d_model

        z_in, i_in, f_in, o_in = all_in.chunk(4, dim=-1)
        all_in_cuda = torch.cat([i_in, f_in, z_in, o_in], dim=-1)
        x_cuda = all_in_cuda.view(B, T, Hh, 4, Dh).permute(1, 0, 2, 3, 4).reshape(T, B, 4 * HS).contiguous()

        Rz, Ri, Rf, Ro = self.R_fused.chunk(4, dim=-1)
        R_cuda = torch.cat([Ri, Rf, Rz, Ro], dim=-1).contiguous()

        b_cuda = torch.zeros(4 * HS, device=all_in.device, dtype=all_in.dtype)
        s0_cuda = torch.stack([
            h.reshape(B, HS),
            c.reshape(B, HS),
            n.reshape(B, HS),
            m.reshape(B, HS)
        ], dim=0).contiguous()

        cache_key = (self.training, B, HS, Hh, all_in.device)
        if cache_key not in self._cuda_funcs:
            self._cuda_funcs[cache_key] = self._cuda_kernel.sLSTMFunc(self.training, B, HS, Hh)
        slstm_func = self._cuda_funcs[cache_key]

        states_cuda = _sLSTMCudaFunction.apply(slstm_func, self.training, x_cuda, s0_cuda, R_cuda, b_cuda)

        out_TBH = states_cuda[0, 1:]
        out = out_TBH.transpose(0, 1)

        h_out = states_cuda[0, -1].view(B, Hh, Dh)
        c_out = states_cuda[1, -1].view(B, Hh, Dh)
        n_out = states_cuda[2, -1].view(B, Hh, Dh)
        m_out = states_cuda[3, -1].view(B, Hh, Dh)

        return out, c_out, n_out, m_out, h_out

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[sLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, sLSTMState]:
        """Forward pass through the sLSTM block.

        Args:
            x: Input tensor of shape (B, T, d_model).
            state: Optional previous sLSTMState. If None, initialized to zeros.
            boundaries: Optional boolean mask of shape (B, T) indicating document start boundaries
                in packed sequences. When True at position (b, t), resets both recurrent cell memory
                (c, n, m) and recurrent hidden feedback (h) to prevent cross-document gradient leakage.

        Returns:
            Tuple of (output_tensor, new_state).
        """
        B, T, _ = x.shape
        residual = x
        x_norm = self.ln(x)

        if self.conv is not None:
            x_conv = F.silu(self.conv(x_norm))
        else:
            x_conv = x_norm

        if state is None:
            state = self.init_state(B, device=x.device, dtype=x.dtype)

        # Decoupled Figure 10 projections
        i_raw, f_raw = self.W_if(x_conv).split(self.d_model, dim=-1)
        z_raw, o_raw = self.W_zo(x_norm).split(self.d_model, dim=-1)

        H, Dh = self.num_heads, self.head_dim
        z_in = z_raw.view(B, T, H, Dh)
        i_in = i_raw.view(B, T, H, Dh)
        f_in = f_raw.view(B, T, H, Dh)
        o_in = o_raw.view(B, T, H, Dh)

        all_in = torch.cat([z_in, i_in, f_in, o_in], dim=-1)

        c_in = state.c.squeeze(0) if state.c.dim() == 4 else state.c
        n_in = state.n.squeeze(0) if state.n.dim() == 4 else state.n
        m_in = state.m.squeeze(0) if state.m.dim() == 4 else state.m
        h_in = state.h.squeeze(0) if state.h.dim() == 4 else state.h

        use_cuda = (
            self.backend == "cuda"
            and self._cuda_kernel is not None
            and x.is_cuda
            and boundaries is None
            and not torch._dynamo.is_compiling()
        )

        packed = boundaries is not None
        bounds_mode = get_packed_boundaries_override_mode()
        ckpt_active = bool(self.use_checkpoint and self.training)
        if packed and bounds_mode == PackedBoundariesMode.DISABLE_CKPT_IN_PACKED:
            ckpt_active = False

        if use_cuda:
            h_lstm, c_out, n_out, m_out, h_out = self._run_core_cuda(
                all_in, c_in, n_in, m_in, h_in
            )
        elif ckpt_active:
            h_lstm, c_out, n_out, m_out, h_out = _torch_checkpoint(
                self._run_core_vanilla, all_in, c_in, n_in, m_in, h_in, boundaries,
                use_reentrant=False,
            )
        else:
            h_lstm, c_out, n_out, m_out, h_out = self._run_core_vanilla(
                all_in, c_in, n_in, m_in, h_in, boundaries=boundaries
            )

        h_norm = self.gn(h_lstm, num_heads=self.num_heads)
        x_mid = residual + self.dropout(h_norm)

        # Post-sLSTM GeGLU MLP
        x_mlp = self.ffn(self.ffn_norm(x_mid))
        out = x_mid + x_mlp
        new_state = sLSTMState(c_out, n_out, m_out, h_out)
        return out, new_state

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamps forget gate bias to prevent numerical saturation."""
        HS = self.d_model
        if self.W_if.bias is not None:
            self.W_if.bias.data[HS:2*HS].clamp_(-max_val, max_val)
