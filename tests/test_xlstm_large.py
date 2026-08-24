"""Comprehensive tests for xLSTMLargeBlock architecture, norms, SwiGLU, and asymmetric configurations."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm_cells import (
    xLSTMLargeBlock,
    xLSTMLargeBlockConfig,
    RMSNorm,
    MultiHeadRMSNorm,
    SwiGLUFeedForward,
    soft_cap,
    round_up_to_next_multiple_of,
    detach_states,
    zero_rows,
    mLSTMBlock,
    sLSTMBlock,
    mLSTMCell,
    mLSTM,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_rmsnorm_forward_backward():
    """Test RMSNorm and MultiHeadRMSNorm forward and backward passes."""
    B, T, D = 2, 16, 64
    norm = RMSNorm(D).to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)
    out = norm(x)
    assert out.shape == (B, T, D)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # MultiHeadRMSNorm 4D
    NH, DH = 4, 16
    mhn = MultiHeadRMSNorm(ndim=D, num_heads=NH, head_dim=DH).to(DEV)
    x_4d = torch.randn(B, NH, T, DH, device=DEV, requires_grad=True)
    out_4d = mhn(x_4d)
    assert out_4d.shape == (B, NH, T, DH)
    out_4d.sum().backward()
    assert x_4d.grad is not None and torch.isfinite(x_4d.grad).all()


def test_soft_cap_bounds():
    """Test soft_cap bounds values to [-cap, cap] smoothly."""
    x = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0], device=DEV)
    capped = soft_cap(x, 15.0)
    assert (capped <= 15.0).all()
    assert (capped >= -15.0).all()
    assert torch.isclose(capped[2], torch.tensor(0.0, device=DEV))
    assert soft_cap(x, None) is x


def test_swiglu_feedforward():
    """Test SwiGLUFeedForward under single and fused weight modes."""
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)

    ffn_single = SwiGLUFeedForward(d_model=D, proj_factor=2.6667, weight_mode="single").to(DEV)
    out_s = ffn_single(x)
    assert out_s.shape == (B, T, D)
    out_s.sum().backward()
    assert x.grad is not None

    x.grad.zero_()
    ffn_fused = SwiGLUFeedForward(d_model=D, proj_factor=2.6667, weight_mode="fused").to(DEV)
    out_f = ffn_fused(x)
    assert out_f.shape == (B, T, D)
    out_f.sum().backward()
    assert x.grad is not None


def test_xlstm_large_block_asymmetric_qk_v():
    """Test xLSTMLargeBlock with asymmetric qk_dim (0.5) and v_dim (1.0)."""
    B, T, D = 2, 64, 128
    cfg = xLSTMLargeBlockConfig(
        embedding_dim=D,
        num_heads=4,
        num_blocks=1,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0,
    )
    block = xLSTMLargeBlock(cfg).to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)
    out, state = block(x)
    assert out.shape == (B, T, D)

    # State dimensions: C is (Dh_qk, Dh_v) = (16, 32)
    Dh_qk = int(D * 0.5) // 4  # 16
    Dh_v = int(D * 1.0) // 4   # 32
    assert state.C.shape == (B, 4, Dh_qk, Dh_v)
    assert state.n.shape == (B, 4, Dh_qk)

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_xlstm_large_block_boundaries_reset():
    """Test document boundary resets in xLSTMLargeBlock."""
    B, T, D = 2, 64, 64
    cfg = xLSTMLargeBlockConfig(embedding_dim=D, num_heads=4, num_blocks=1)
    block = xLSTMLargeBlock(cfg).to(DEV)

    x = torch.randn(B, T, D, device=DEV)
    b = torch.zeros(B, T, dtype=torch.bool, device=DEV)
    b[:, 32] = True

    out, state = block(x, boundaries=b)
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()


def test_mlstm_and_slstm_block_norm_type():
    """Test mLSTMBlock and sLSTMBlock with norm_type='rmsnorm'."""
    B, T, D = 2, 16, 64
    mb_rms = mLSTMBlock(d_model=D, num_heads=4, norm_type="rmsnorm").to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)
    out_m, _ = mb_rms(x)
    assert out_m.shape == (B, T, D)
    out_m.sum().backward()
    assert x.grad is not None

    x.grad.zero_()
    sb_rms = sLSTMBlock(d_model=D, num_heads=4, norm_type="rmsnorm").to(DEV)
    out_s, _ = sb_rms(x)
    assert out_s.shape == (B, T, D)
    out_s.sum().backward()
    assert x.grad is not None


def test_xlstm_large_with_conv1d():
    """Test xLSTMLargeBlock with optional CausalConv1d."""
    B, T, D = 2, 32, 64
    cfg = xLSTMLargeBlockConfig(
        embedding_dim=D,
        num_heads=4,
        num_blocks=1,
        conv1d_kernel_size=4,
    )
    block = xLSTMLargeBlock(cfg).to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)
    out, st = block(x)
    assert out.shape == (B, T, D)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mlstm_cell_and_layer_asymmetric_qk_v():
    """Test mLSTMCell and mLSTM sequence layer with asymmetric qk_dim_factor and v_dim_factor."""
    B, T = 2, 16
    in_sz = 32
    hidden_sz = 64
    cell = mLSTMCell(input_size=in_sz, hidden_size=hidden_sz, num_heads=4, qk_dim_factor=0.5, v_dim_factor=1.0).to(DEV)
    st = cell.init_state(B, device=DEV)
    assert st.C.shape == (B, 4, 8, 16)  # (B, H, Dh_qk, Dh_v)
    assert st.n.shape == (B, 4, 8)

    x_t = torch.randn(B, in_sz, device=DEV)
    out_t, st = cell(x_t, st)
    assert out_t.shape == (B, 64)

    # Sequence layer
    layer = mLSTM(input_size=in_sz, hidden_size=hidden_sz, num_layers=2, num_heads=4, qk_dim_factor=0.5, v_dim_factor=1.0).to(DEV)
    x_seq = torch.randn(B, T, in_sz, device=DEV, requires_grad=True)
    out_seq, states = layer(x_seq)
    assert out_seq.shape == (B, T, 64)
    out_seq.sum().backward()
    assert x_seq.grad is not None and torch.isfinite(x_seq.grad).all()


def test_mlstm_block_asymmetric_learnable_skip_grad():
    """Test mLSTMBlock with asymmetric v_dim_factor != 1.0 ensures learnable_skip receives finite non-zero grad."""
    B, T, D = 2, 16, 64
    block = mLSTMBlock(
        d_model=D,
        num_heads=4,
        expand_factor=2,
        qk_dim_factor=0.5,
        v_dim_factor=0.75,
    ).to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)
    out, _ = block(x)
    assert out.shape == (B, T, D)
    loss = out.sum()
    loss.backward()

    assert block.learnable_skip.grad is not None
    assert torch.isfinite(block.learnable_skip.grad).all()
    assert (block.learnable_skip.grad.abs() > 0).any()
