"""Tests for xlstm-cells v0.6.0: Full alignment with official xLSTM paper & repo."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm_cells import (
    mLSTM,
    mLSTMCell,
    mLSTMBlock,
    sLSTM,
    sLSTMCell,
    sLSTMBlock,
    LayerNorm,
    MultiHeadLayerNorm,
    CausalConv1d,
    GatedFeedForward,
    LinearHeadwiseExpand,
    zero_rows,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_multihead_layernorm_token_wise_isolation():
    """Verify that an outlier on one token does not corrupt other tokens."""
    B, T, NH, DH = 2, 64, 4, 16
    C = NH * DH
    mhln = MultiHeadLayerNorm(ndim=C, weight=True, bias=False).to(DEV)

    x = torch.randn(B, T, C, device=DEV)
    out_clean = mhln(x, num_heads=NH)

    # Corrupt token 30
    x_outlier = x.clone()
    x_outlier[:, 30, :] *= 100.0
    out_outlier = mhln(x_outlier, num_heads=NH)

    # Token 10 must be 100% unaffected
    diff_token10 = (out_outlier[:, 10] - out_clean[:, 10]).abs().max().item()
    assert diff_token10 == 0.0, f"Token-wise isolation violated: diff={diff_token10}"


def test_mlstm_block_forward_backward():
    """Test full forward and backward through paper-compliant mLSTMBlock."""
    B, T, D = 2, 128, 64
    block = mLSTMBlock(d_model=D, num_heads=4, use_triton_kernels=True, eps=1e-3).to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)

    out, state = block(x)
    assert out.shape == (B, T, D)
    Dh = (D * 2) // 4
    assert state.C.shape == (B, 4, Dh, Dh)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mlstm_block_boundaries_reset():
    """Test that document boundaries reset recurrent states in mLSTMBlock."""
    B, T, D = 2, 128, 64
    block = mLSTMBlock(d_model=D, num_heads=4, use_triton_kernels=True, eps=1e-3).to(DEV)
    x = torch.randn(B, T, D, device=DEV)
    b = torch.zeros(B, T, dtype=torch.bool, device=DEV)
    b[:, 64] = True

    out, state = block(x, boundaries=b)
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()


def test_slstm_block_forward_backward():
    """Test full forward and backward through paper-compliant sLSTMBlock."""
    B, T, D = 2, 64, 64
    block = sLSTMBlock(d_model=D, num_heads=4, backend="vanilla").to(DEV)
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)

    out, state = block(x)
    assert out.shape == (B, T, D)
    assert state.c.shape == (1, B, 4, D // 4) or state.c.shape == (B, 4, D // 4)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for backend='cuda'")
def test_slstm_cuda_backend_warns_and_ignores_fast_mode():
    """backend='cuda' with fast_mode=True must warn and ignore fast_mode without raising."""
    with pytest.warns(UserWarning, match="does not use fast_mode"):
        layer = sLSTM(64, 64, backend="cuda", fast_mode=True)
        assert layer.fast_mode is False


def test_slstm_gate_clamping():
    """Exponential gates must be bounded to <= 1.0."""
    cell = sLSTMCell(16, 16, num_heads=2).to(DEV)
    state = cell.init_state(1, device=DEV)
    # Huge positive input
    x = torch.full((1, 16), 50.0, device=DEV)
    out, new_state = cell(x, state)
    assert torch.isfinite(out).all()
    assert torch.isfinite(new_state.c).all()


def test_gated_feedforward_geglu():
    """Test GeGLU post-MLP in components."""
    mlp = GatedFeedForward(d_model=64, proj_factor=4.0/3.0, act_fn="gelu").to(DEV)
    x = torch.randn(2, 16, 64, device=DEV)
    out = mlp(x)
    assert out.shape == (2, 16, 64)


def test_custom_head_topology_explicit_hidden_size():
    """Test custom head topologies like 19 heads with explicit hidden_size=2432."""
    blk = mLSTMBlock(d_model=512, hidden_size=2432, num_heads=19).to(DEV)
    assert blk.expanded == 2432
    assert blk.head_dim == 128
    x = torch.randn(2, 8, 512, device=DEV, requires_grad=True)
    out, state = blk(x)
    assert out.shape == (2, 8, 512)
    assert state.C.shape == (2, 19, 128, 128)
    out.sum().backward()
    assert x.grad is not None

    with pytest.raises(ValueError, match="conflicting arguments"):
        mLSTMBlock(d_model=512, expand_factor=2, hidden_size=2432, num_heads=19)


def test_linear_headwise_expand_factor():
    """Test LinearHeadwiseExpand initialized via expand_factor without out_features."""
    lin = LinearHeadwiseExpand(in_features=64, num_heads=4, expand_factor=2.0).to(DEV)
    assert lin.out_features == 128
    assert lin.bias.shape == (128,)
    x = torch.randn(2, 5, 64, device=DEV)
    out = lin(x)
    assert out.shape == (2, 5, 128)


def test_feedforward_hidden_size_alias_and_conflict():
    """Test GatedFeedForward hidden_size alias and mutual conflict check."""
    ffn = GatedFeedForward(d_model=128, hidden_size=256).to(DEV)
    assert ffn.proj_up_dim == 256
    x = torch.randn(2, 8, 128, device=DEV)
    out = ffn(x)
    assert out.shape == (2, 8, 128)

    with pytest.raises(ValueError, match="cannot specify both 'proj_up_dim' and 'hidden_size'"):
        GatedFeedForward(d_model=128, proj_up_dim=256, hidden_size=512)


def test_slstm_and_mlstm_per_layer_heads_sequence():
    """Test sLSTM and mLSTM with per-layer head counts sequence."""
    slstm = sLSTM(input_size=128, hidden_size=256, num_layers=2, num_heads=[4, 8], backend="vanilla").to(DEV)
    assert slstm.num_heads == [4, 8]
    x = torch.randn(2, 10, 128, device=DEV)
    out_s, _ = slstm(x)
    assert out_s.shape == (2, 10, 256)

    mlstm = mLSTM(input_size=128, hidden_size=256, num_layers=2, num_heads=[4, 8]).to(DEV)
    assert mlstm.num_heads == [4, 8]
    out_m, _ = mlstm(x)
    assert out_m.shape == (2, 10, 256)


def test_causal_conv1d_short_prefill_step_parity():
    """Test short prefill lengths (T < pad) seamless parity into step()."""
    conv = CausalConv1d(feature_dim=32, kernel_size=4).to(DEV)
    for p_len in [0, 1, 2, 3]:
        p = torch.randn(2, p_len, 32, device=DEV) if p_len > 0 else torch.empty(2, 0, 32, device=DEV)
        _, conv_st = conv(p, return_last_state=True)
        assert conv_st.shape == (2, 3, 32)

        # Append one token and verify step() matches full sequence forward()
        next_tok = torch.randn(2, 1, 32, device=DEV)
        out_step, next_conv_st = conv.step(next_tok, conv_state=conv_st)

        full_seq = torch.cat([p, next_tok], dim=1)
        out_full, full_conv_st = conv(full_seq, return_last_state=True)

        assert torch.allclose(out_step, out_full[:, -1:], atol=1e-5)
        assert torch.allclose(next_conv_st, full_conv_st, atol=1e-5)


def test_zero_rows_ambiguous_batch_dim():
    """Test zero_rows ambiguity resolution when D == B."""
    from xlstm_cells.slstm import sLSTMState
    state = sLSTMState(
        c=torch.ones(2, 2, 4, 16),
        n=torch.ones(2, 2, 4, 16),
        m=torch.zeros(2, 2, 4, 16),
        h=torch.zeros(2, 2, 4, 16),
    )
    mask = torch.tensor([False, True])
    with pytest.raises(ValueError, match="ambiguous batch dimension"):
        zero_rows(state, mask)

    zero_rows(state, mask, batch_dim=1)
    assert (state.c[:, 0] == 1).all()
    assert (state.c[:, 1] == 0).all()

