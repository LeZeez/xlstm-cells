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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.conv import CausalConv1d
from .components.feedforward import GatedFeedForward
from .components.init import bias_linspace_init_, small_init_init_, wang_init_
from .components.ln import LayerNorm, MultiHeadLayerNorm
from .mlstm import _MAX_FORGET_BIAS, _MLSTM_CHUNK_SIZE, _BOUNDARY_RESET_LOGF, _EPS, mLSTMState
from .slstm import sLSTM, sLSTMState

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
        eps: Denominator epsilon constant for stabilizer numerical stability. Default: 1e-3.
        num_blocks: Total number of stacked blocks in the model (for Wang init scaling). Default: 1.
    """

    def __init__(
        self,
        d_model: int,
        expand_factor: int = 2,
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
    ):
        """Initializes paper-compliant mLSTMBlock."""
        super().__init__()
        expanded = d_model * expand_factor
        assert expanded % num_heads == 0, f"expanded ({expanded}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.expanded = expanded
        self.num_heads = num_heads
        self.head_dim = expanded // num_heads
        self.conv_kernel = conv_kernel
        self.use_checkpoint = use_checkpoint
        self._use_triton_kernels = use_triton_kernels and _HAS_MLSTM_KERNELS
        self._chunkwise_kernel = chunkwise_kernel
        self._chunk_size = chunk_size
        self._eps = float(eps) if eps is not None else _EPS

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
            boundaries: Optional boolean mask of shape (B, T) indicating document start boundaries.

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
        mlp_factor: float = 4.0 / 3.0,
        dropout: float = 0.0,
        bias: bool = True,
        backend: str = "vanilla",
        use_checkpoint: bool = False,
        fast_mode: bool = False,
        fast_chunk_size: int = 32,
        num_blocks: int = 1,
    ):
        """Initializes paper-compliant sLSTMBlock."""
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.ln = LayerNorm(d_model, bias=False)

        if conv_kernel > 0:
            self.conv = CausalConv1d(d_model, kernel_size=conv_kernel, bias=bias)
        else:
            self.conv = None

        self.lstm = sLSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            num_heads=num_heads,
            bias=bias,
            batch_first=True,
            backend=backend,
            use_checkpoint=use_checkpoint,
            fast_mode=fast_mode,
            fast_chunk_size=fast_chunk_size,
        )

        self.gn = MultiHeadLayerNorm(ndim=d_model, weight=True, bias=False, eps=1e-5)

        # Post-sLSTM GeGLU Feedforward
        self.ffn_norm = LayerNorm(d_model, bias=False)
        self.ffn = GatedFeedForward(
            d_model=d_model,
            proj_factor=mlp_factor,
            act_fn="gelu",
            dropout=dropout,
            bias=bias,
            num_blocks=num_blocks,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets all block layer parameters."""
        self.ln.reset_parameters()
        if self.conv is not None:
            self.conv.reset_parameters()
        self.lstm.reset_parameters()
        self.gn.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def init_state(self, batch_size: int, device=None, dtype=None) -> sLSTMState:
        """Initializes zero state object for sLSTM recurrence."""
        return self.lstm.init_state(batch_size, device, dtype)

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
            boundaries: Optional boolean mask of shape (B, T) indicating document start boundaries.

        Returns:
            Tuple of (output_tensor, new_state).
        """
        residual = x
        x_norm = self.ln(x)

        if self.conv is not None:
            c = F.silu(self.conv(x_norm))
            x_in = x_norm + c
        else:
            x_in = x_norm

        h, state = self.lstm(x_in, state, boundaries=boundaries)
        if isinstance(state, tuple) and len(state) == 1:
            state = state[0]
        h_norm = self.gn(h, num_heads=self.num_heads)
        x_mid = residual + self.dropout(h_norm)

        # Post-sLSTM GeGLU MLP
        x_mlp = self.ffn(self.ffn_norm(x_mid))
        out = x_mid + x_mlp
        return out, state

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        """Clamps forget gate bias to prevent numerical saturation."""
        self.lstm.clamp_forget_bias(max_val)
