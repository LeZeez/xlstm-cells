# Copyright (c) NXAI GmbH and its affiliates 2024
"""xLSTMLargeBlock: Official xLSTMLarge residual block and mLSTM layer."""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.conv import CausalConv1d
from .components.feedforward import SwiGLUFeedForward
from .components.ln import LayerNorm, MultiHeadLayerNorm, RMSNorm
from .components.utils import soft_cap
from .configs import xLSTMLargeBlockConfig
from .mlstm import (
    _BOUNDARY_RESET_LOGF,
    _EPS,
    _HAS_MLSTM_KERNELS,
    _TRITON_CHUNKWISE_KERNELS,
    mLSTMState,
    _mlstm_recurrent_scan_parallel_chunked,
)

try:
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackend,
        mLSTMBackendConfig,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# xLSTMLarge mLSTM Layer
# ---------------------------------------------------------------------------

class xLSTMLargeLayer(nn.Module):
    """Core mLSTM layer matching the official xLSTMLarge architecture.

    Features asymmetric QK/V dimensions, gate soft-capping, token-wise MultiHeadLayerNorm,
    and automatic fallback to native chunked parallel scan when Triton is unaligned or unavailable.
    """

    def __init__(self, config: xLSTMLargeBlockConfig):
        super().__init__()
        self.config = config

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)
        self.head_dim = self.qk_dim // self.num_heads
        self.v_head_dim = self.v_dim // self.num_heads
        self.gate_soft_cap = config.gate_soft_cap
        self.eps = config.eps
        self.chunk_size = config.chunk_size

        if self.qk_dim % self.num_heads != 0:
            raise ValueError(
                f"xLSTMLargeLayer: qk_dim ({self.qk_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.v_dim % self.num_heads != 0:
            raise ValueError(
                f"xLSTMLargeLayer: v_dim ({self.v_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if config.conv1d_kernel_size > 0:
            self.conv = CausalConv1d(config.embedding_dim, kernel_size=config.conv1d_kernel_size, bias=config.use_bias)
        else:
            self.conv = None

        if config.weight_mode == "single":
            self.q = nn.Linear(config.embedding_dim, self.qk_dim, bias=config.use_bias)
            self.k = nn.Linear(config.embedding_dim, self.qk_dim, bias=config.use_bias)
            self.v = nn.Linear(config.embedding_dim, self.v_dim, bias=config.use_bias)
            self.ogate_preact = nn.Linear(config.embedding_dim, self.v_dim, bias=config.use_bias)
            self.igate_preact = nn.Linear(config.embedding_dim, config.num_heads, bias=True)
            self.fgate_preact = nn.Linear(config.embedding_dim, config.num_heads, bias=True)
        else:
            self.qkv_opreact = nn.Linear(config.embedding_dim, 2 * self.qk_dim + 2 * self.v_dim, bias=config.use_bias)
            self.ifgate_preact = nn.Linear(config.embedding_dim, 2 * config.num_heads, bias=True)

        self.ogate_act_fn = nn.Sigmoid()

        self.multihead_norm = MultiHeadLayerNorm(
            ndim=self.v_dim,
            eps=config.norm_eps,
            weight=True,
            bias=config.use_bias,
        )

        self.out_proj = nn.Linear(self.v_dim, config.embedding_dim, bias=config.use_bias)

        self._mlstm_backend = None
        self._use_triton = _HAS_MLSTM_KERNELS
        if self._use_triton:
            self._init_triton_backend()

    def _init_triton_backend(self) -> None:
        if not _HAS_MLSTM_KERNELS:
            return
        kernel_name = _TRITON_CHUNKWISE_KERNELS.get(self.config.chunkwise_kernel, self.config.chunkwise_kernel)
        backend_cfg = mLSTMBackendConfig(
            chunkwise_kernel=kernel_name,
            sequence_kernel=self.config.sequence_kernel,
            step_kernel=self.config.step_kernel,
            chunk_size=self.chunk_size,
            return_last_states=True,
            autocast_kernel_dtype=self.config.autocast_kernel_dtype,
            eps=self.eps,
        )
        self._mlstm_backend = mLSTMBackend(config=backend_cfg)

    def init_state(self, batch_size: int, device=None, dtype=None) -> mLSTMState:
        """Initializes zero state object for mLSTM recurrence."""
        return mLSTMState.init(batch_size, self.num_heads, self.head_dim, self.v_head_dim, device, dtype)

    def _run_core_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: Optional[torch.Tensor],
        i_tilde: torch.Tensor,
        f_raw: torch.Tensor,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_f = F.logsigmoid(f_raw)
        if boundaries is not None:
            b = boundaries.to(device=log_f.device, dtype=torch.bool).unsqueeze(-1)
            log_f = log_f.masked_fill(b, _BOUNDARY_RESET_LOGF)
        h, C_out, n_out, m_out = _mlstm_recurrent_scan_parallel_chunked(
            q, k, v, o, i_tilde, log_f, C, n, m,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )
        return h, C_out, n_out, m_out

    def _run_core_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        i_tilde: torch.Tensor,
        f_raw: torch.Tensor,
        C: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, Dh_qk = q.shape
        sf = math.sqrt(Dh_qk)
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

        c_init = C * sf if C is not None else None
        n_init = n * sf if n is not None else None

        h_k, (C_out, n_out, m_out_k) = self._mlstm_backend(
            q=q_k, k=k_k, v=v_k, i=i_k, f=f_k,
            c_initial=c_init, n_initial=n_init, m_initial=m_k,
            return_last_states=True,
        )

        h_out = h_k.permute(0, 2, 1, 3)
        h_norm = self.multihead_norm(h_out, num_heads=H)
        h_gated = self.ogate_act_fn(o) * h_norm
        h_flat = h_gated.reshape(B, T, -1)

        m_out = m_out_k.squeeze(-1)
        C_out = C_out / sf
        n_out = n_out / sf

        return h_flat, C_out, n_out, m_out

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[mLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        B, S, _ = x.shape

        if self.conv is not None:
            x_in = F.silu(self.conv(x))
        else:
            x_in = x

        if self.config.weight_mode == "single":
            q = self.q(x_in)
            k = self.k(x_in)
            v = self.v(x_in)
            o_preact = self.ogate_preact(x_in)
            i_preact = soft_cap(self.igate_preact(x_in), self.gate_soft_cap)
            f_preact = soft_cap(self.fgate_preact(x_in), self.gate_soft_cap)
        else:
            qkv_opreact = self.qkv_opreact(x_in)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (self.qk_dim, 2 * self.qk_dim, 2 * self.qk_dim + self.v_dim),
                dim=-1,
            )
            if_preact = soft_cap(self.ifgate_preact(x_in), self.gate_soft_cap)
            i_preact, f_preact = torch.tensor_split(if_preact, (self.num_heads,), dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim)
        k = (k / math.sqrt(self.head_dim)).view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.v_head_dim)
        o = o_preact.view(B, S, self.num_heads, self.v_head_dim)

        if state is None:
            state = self.init_state(B, device=x.device, dtype=x.dtype)

        C_in = state.C.squeeze(0) if state.C.dim() == 5 else state.C
        n_in = state.n.squeeze(0) if state.n.dim() == 4 else state.n
        m_in = state.m.squeeze(0) if state.m.dim() == 3 else state.m

        # Triton kernel requires CUDA, alignment, and Dh >= 16
        is_triton_supported = (self.head_dim >= 16 and self.v_head_dim >= 16)
        use_triton = (
            self._use_triton
            and self._mlstm_backend is not None
            and x.is_cuda
            and S % self.chunk_size == 0
            and is_triton_supported
            and not torch._dynamo.is_compiling()
        )

        if use_triton:
            h_out, C_out, n_out, m_out = self._run_core_triton(
                q, k, v, o, i_preact, f_preact, C_in, n_in, m_in, boundaries=boundaries
            )
        else:
            h_lstm, C_out, n_out, m_out = self._run_core_native(
                q, k, v, None, i_preact, f_preact, C_in, n_in, m_in, boundaries=boundaries
            )
            h_norm = self.multihead_norm(h_lstm.view(B, S, self.num_heads, self.v_head_dim), num_heads=self.num_heads)
            h_out = self.ogate_act_fn(o) * h_norm
            h_out = h_out.reshape(B, S, -1)

        y = self.out_proj(h_out)
        return y, mLSTMState(C_out, n_out, m_out)


# ---------------------------------------------------------------------------
# xLSTMLarge Block
# ---------------------------------------------------------------------------

class xLSTMLargeBlock(nn.Module):
    """Residual block matching the official xLSTMLarge architecture.

    Components:
        x -> RMSNorm -> mLSTMLayer -> (+) -> RMSNorm -> SwiGLUFeedForward -> (+) -> out
    """

    def __init__(self, config: xLSTMLargeBlockConfig):
        super().__init__()
        self.config = config

        if config.norm_type == "rmsnorm":
            self.norm_mlstm = RMSNorm(
                config.embedding_dim,
                eps=config.norm_eps,
                weight=True,
                bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
            self.norm_ffn = RMSNorm(
                config.embedding_dim,
                eps=config.norm_eps,
                weight=True,
                bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
        else:
            self.norm_mlstm = LayerNorm(config.embedding_dim, eps=config.norm_eps, weight=True, bias=config.use_bias)
            self.norm_ffn = LayerNorm(config.embedding_dim, eps=config.norm_eps, weight=True, bias=config.use_bias)

        self.use_checkpoint = config.use_checkpoint
        self.mlstm_layer = xLSTMLargeLayer(config)

        self.ffn = SwiGLUFeedForward(
            d_model=config.embedding_dim,
            proj_factor=config.ffn_proj_factor,
            round_up_to_multiple_of=config.ffn_round_up_to_multiple_of,
            bias=config.use_bias,
            weight_mode=config.weight_mode,
            dropout=config.dropout,
            num_blocks=config.num_blocks,
        )

    def init_state(self, batch_size: int, device=None, dtype=None) -> mLSTMState:
        return self.mlstm_layer.init_state(batch_size, device=device, dtype=dtype)

    def _forward_body(
        self,
        x: torch.Tensor,
        state: Optional[mLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        x_norm = self.norm_mlstm(x)
        x_lstm, new_state = self.mlstm_layer(x_norm, state=state, boundaries=boundaries)
        x = x + x_lstm

        x_ffn = self.norm_ffn(x)
        x = x + self.ffn(x_ffn)
        return x, new_state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[mLSTMState] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, mLSTMState]:
        if self.use_checkpoint and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_body, x, state, boundaries, use_reentrant=False
            )
        return self._forward_body(x, state=state, boundaries=boundaries)
