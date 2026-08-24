# Copyright (c) NXAI GmbH and its affiliates 2024
"""Configuration dataclasses for xLSTM cells, sequence models, and blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class mLSTMCellConfig:
    """Configuration for single-step mLSTMCell."""
    input_size: int
    hidden_size: int
    num_heads: int = 4
    bias: bool = False
    eps: float = 1e-6
    qk_dim_factor: Optional[float] = None
    v_dim_factor: Optional[float] = None

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        if self.eps <= 0 or not math.isfinite(self.eps):
            raise ValueError(f"eps must be a positive finite float, got {self.eps}")


@dataclass
class sLSTMCellConfig:
    """Configuration for single-step sLSTMCell."""
    input_size: int
    hidden_size: int
    num_heads: int = 4
    bias: bool = True
    eps: float = 1e-6

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")


@dataclass
class mLSTMConfig:
    """Configuration for multi-layer mLSTM sequence model."""
    input_size: int
    hidden_size: int
    num_layers: int = 1
    num_heads: Union[int, List[int], Tuple[int, ...]] = 4
    bias: bool = False
    batch_first: bool = True
    dropout: float = 0.0
    bidirectional: bool = False
    pack_state: bool = True
    use_checkpoint: bool = False
    use_triton_kernels: bool = True
    chunkwise_kernel: str = "xl_chunk"
    chunk_size: int = 128
    eps: Optional[float] = None
    qk_dim_factor: Optional[float] = None
    v_dim_factor: Optional[float] = None

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError(f"mLSTMConfig: input_size must be positive, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"mLSTMConfig: hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"mLSTMConfig: num_layers must be positive, got {self.num_layers}")
        if self.chunk_size <= 0:
            raise ValueError(f"mLSTMConfig: chunk_size must be positive, got {self.chunk_size}")
        if self.eps is not None and (self.eps <= 0 or not math.isfinite(self.eps)):
            raise ValueError(f"mLSTMConfig: eps must be a positive finite float, got {self.eps}")


@dataclass
class sLSTMConfig:
    """Configuration for multi-layer sLSTM sequence model."""
    input_size: int
    hidden_size: int
    num_layers: int = 1
    num_heads: Union[int, List[int], Tuple[int, ...]] = 4
    bias: bool = True
    batch_first: bool = True
    dropout: float = 0.0
    bidirectional: bool = False
    pack_state: bool = True
    backend: str = "vanilla"
    use_checkpoint: bool = False
    fast_mode: bool = False
    fast_chunk_size: int = 32

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError(f"sLSTMConfig: input_size must be positive, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"sLSTMConfig: hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"sLSTMConfig: num_layers must be positive, got {self.num_layers}")
        if self.backend not in ("vanilla", "cuda"):
            raise ValueError(f"sLSTMConfig: unknown backend {self.backend!r}. Must be 'vanilla' or 'cuda'.")
        if self.fast_mode and self.fast_chunk_size <= 0:
            raise ValueError(f"sLSTMConfig: fast_chunk_size must be positive when fast_mode=True, got {self.fast_chunk_size}")


@dataclass
class mLSTMBlockConfig:
    """Configuration for paper-compliant mLSTMBlock (Figure 11)."""
    d_model: int
    expand_factor: Optional[float] = None
    hidden_size: Optional[int] = None
    num_heads: int = 4
    conv_kernel: int = 4
    dropout: float = 0.0
    bias: bool = False
    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    use_checkpoint: bool = False
    use_triton_kernels: bool = True
    chunkwise_kernel: str = "xl_chunk"
    chunk_size: int = 128
    eps: Optional[float] = None
    num_blocks: int = 1
    qk_dim_factor: Optional[float] = None
    v_dim_factor: Optional[float] = None

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError(f"mLSTMBlockConfig: d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"mLSTMBlockConfig: num_heads must be positive, got {self.num_heads}")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"mLSTMBlockConfig: unknown norm_type {self.norm_type!r}, expected 'layernorm' or 'rmsnorm'")


@dataclass
class sLSTMBlockConfig:
    """Configuration for paper-compliant sLSTMBlock (Figure 10)."""
    d_model: int
    num_heads: int = 4
    conv_kernel: int = 4
    mlp_factor: Optional[float] = None
    hidden_size: Optional[int] = None
    dropout: float = 0.0
    bias: bool = True
    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    backend: str = "vanilla"
    use_checkpoint: bool = False
    fast_mode: bool = False
    fast_chunk_size: int = 32
    num_blocks: int = 1

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError(f"sLSTMBlockConfig: d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"sLSTMBlockConfig: num_heads must be positive, got {self.num_heads}")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"sLSTMBlockConfig: unknown norm_type {self.norm_type!r}, expected 'layernorm' or 'rmsnorm'")
        if self.backend not in ("vanilla", "cuda"):
            raise ValueError(f"sLSTMBlockConfig: unknown backend {self.backend!r}. Must be 'vanilla' or 'cuda'.")


@dataclass
class xLSTMLargeBlockConfig:
    """Configuration for xLSTMLargeBlock residual block."""
    embedding_dim: int = 4096
    num_heads: int = 8
    num_blocks: int = 32
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True

    # Asymmetric matrix memory dimensions
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0

    # Backend and chunking
    chunkwise_kernel: str = "chunkwise--triton_limit_chunk"
    sequence_kernel: str = "native_sequence__triton"
    step_kernel: str = "triton"
    chunk_size: int = 64
    autocast_kernel_dtype: str = "bfloat16"
    eps: float = 1e-6

    # SwiGLU Feedforward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64

    # Gate soft-capping
    gate_soft_cap: Optional[float] = 15.0

    # Weight layout & custom features
    weight_mode: str = "single"  # "single" or "fused"
    conv1d_kernel_size: int = 0
    norm_type: str = "rmsnorm"   # "rmsnorm" or "layernorm"
    act_fn: str = "silu"
    use_checkpoint: bool = False
    dropout: float = 0.0

    # Aliases
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None

    def __post_init__(self):
        if self.hidden_size is not None:
            if self.embedding_dim != 4096 and self.hidden_size != self.embedding_dim:
                raise ValueError(
                    f"xLSTMLargeBlockConfig: conflicting values: embedding_dim={self.embedding_dim} and hidden_size={self.hidden_size}"
                )
            self.embedding_dim = self.hidden_size
        if self.num_hidden_layers is not None:
            if self.num_blocks != 32 and self.num_hidden_layers != self.num_blocks:
                raise ValueError(
                    f"xLSTMLargeBlockConfig: conflicting values: num_blocks={self.num_blocks} and num_hidden_layers={self.num_hidden_layers}"
                )
            self.num_blocks = self.num_hidden_layers

        if self.embedding_dim <= 0:
            raise ValueError(f"xLSTMLargeBlockConfig: embedding_dim must be positive, got {self.embedding_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"xLSTMLargeBlockConfig: num_heads must be positive, got {self.num_heads}")
        if self.num_blocks <= 0:
            raise ValueError(f"xLSTMLargeBlockConfig: num_blocks must be positive, got {self.num_blocks}")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"xLSTMLargeBlockConfig: unknown norm_type {self.norm_type!r}")
        if self.weight_mode not in ("single", "fused"):
            raise ValueError(f"xLSTMLargeBlockConfig: unknown weight_mode {self.weight_mode!r}")
