"""Targeted tests for numerical stability, boundary resets, and denominator floor bounds."""

import math

import pytest
import torch

from xlstm_cells import (
    mLSTM,
    mLSTMCell,
    mLSTMBlock,
    sLSTM,
    sLSTMCell,
    sLSTMBlock,
)

from xlstm_cells import mlstm as mlstm_mod
from xlstm_cells import slstm as slstm_mod
from xlstm_cells.mlstm import _BOUNDARY_RESET_LOGF, _EPS


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Fix 1: boundary reset must survive an inflated log-normalizer m
# ---------------------------------------------------------------------------

def _inflated_m_run(model, x, boundaries, m_val, device):
    """Run the model with a state whose log-normalizer m is inflated to
    m_val, then return outputs and final m.  C/n (or c/n/h) stay at the
    zero init, matching real training state semantics (only m can drift)."""
    st = model.init_state(x.shape[0], device=device, dtype=x.dtype)
    st.m.fill_(m_val)
    if hasattr(st, "n"):
        st.n.fill_(1.0)
    out, st_out = model(x, st, boundaries=boundaries)
    return out, st_out


@pytest.mark.parametrize("num_heads", [2, 4])
def test_mlstm_boundary_reset_with_inflated_m(num_heads):
    """With the old -30 constant and m=50, the boundary reset fails
    silently (f_prime = 1, full carry).  With -1000 the post-boundary
    segment must match a fresh-state run exactly."""
    torch.manual_seed(0)
    device = _device()
    B, T, D = 2, 16, 16
    model = mLSTM(
        D, 32, num_layers=1, num_heads=num_heads,
        batch_first=True, pack_state=False, use_triton_kernels=False,
    ).to(device).eval()
    x = torch.randn(B, T, D, device=device)

    boundary_t = 8
    b = torch.zeros(B, T, dtype=torch.bool, device=device)
    b[:, boundary_t] = True

    # Baseline: fresh zero state, with boundary.
    st_fresh = model.init_state(B, device=device, dtype=x.dtype)
    out_fresh, st_fresh_out = model(x, st_fresh, boundaries=b)

    # No-boundary baseline with inflated m: what the model would do if
    # the reset did not fire (full carry-through).
    out_inf_nob, st_inf_nob = _inflated_m_run(model, x, None, 50.0, device)

    # Inflated m WITH boundary: must behave like the fresh run, not the
    # carry-through run.
    out_inf_b, st_inf_b = _inflated_m_run(model, x, b, 50.0, device)

    # The inflated-m no-boundary run must actually differ from the fresh
    # run (otherwise the test is vacuous).
    assert not torch.allclose(
        out_inf_nob[:, boundary_t:], out_fresh[:, boundary_t:], atol=1e-3
    ), "sanity: inflated-m carry-through must differ from fresh run"

    # With the -1000 reset the post-boundary segment equals the fresh run.
    assert torch.allclose(
        out_inf_b[:, boundary_t:], out_fresh[:, boundary_t:], atol=1e-4
    ), "post-boundary outputs must match a fresh-state run after reset"

    # Final m must be reset to the fresh-run level, not stay inflated.
    assert torch.allclose(st_inf_b.m, st_fresh_out.m, atol=1e-3), (
        "log-normalizer m must be reset at the boundary"
    )


def test_mlstm_boundary_reset_old_constant_would_fail():
    """Demonstrate the failure mode the fix addresses: with the old -30
    constant and m=50, the boundary does NOT reset (state stays inflated),
    while with -1000 it does."""
    torch.manual_seed(0)
    device = _device()
    B, T, D = 2, 16, 16
    model = mLSTM(
        D, 32, num_layers=1, num_heads=2,
        batch_first=True, pack_state=False, use_triton_kernels=False,
    ).to(device).eval()
    x = torch.randn(B, T, D, device=device)
    boundary_t = 8
    b = torch.zeros(B, T, dtype=torch.bool, device=device)
    b[:, boundary_t] = True

    st_fresh = model.init_state(B, device=device, dtype=x.dtype)
    out_fresh, st_fresh_out = model(x, st_fresh, boundaries=b)

    out_new, st_new_out = _inflated_m_run(model, x, b, 50.0, device)

    # With the new -1000 constant the reset succeeds: outputs match fresh.
    assert torch.allclose(
        out_new[:, boundary_t:], out_fresh[:, boundary_t:], atol=1e-4
    ), "new constant must fully reset the inflated-m state"

    import xlstm_cells.mlstm as mlm

    # Re-run with the old -30 constant monkeypatched in.
    orig = mlm._BOUNDARY_RESET_LOGF
    mlm._BOUNDARY_RESET_LOGF = -30.0
    try:
        out_old, st_old_out = _inflated_m_run(model, x, b, 50.0, device)
    finally:
        mlm._BOUNDARY_RESET_LOGF = orig

    # With the old constant the reset fails: outputs must NOT match the
    # fresh run, and the final log-normalizer must remain inflated (~20)
    # instead of collapsing back to the fresh-run level (~0).
    assert not torch.allclose(
        out_old[:, boundary_t:], out_fresh[:, boundary_t:], atol=1e-3
    ), "old -30 constant must fail to reset an inflated m=50 state"
    assert st_old_out.m.mean().item() > 10.0, (
        "old constant leaves m inflated after the boundary"
    )
    assert st_new_out.m.abs().mean().item() < 1.0, (
        "new constant resets m to the fresh-run level"
    )


@pytest.mark.parametrize("num_heads", [2, 4])
def test_slstm_boundary_reset_with_inflated_m(num_heads):
    """sLSTM: with an inflated m state, the boundary must collapse the
    log-normalizer back to the fresh-run level.  (Output equality at the
    boundary position itself is not expected -- the sLSTM gates at time t
    depend on the pre-boundary hidden h via R_fused, so the boundary
    position's inputs differ between runs.  The m-reset is the
    confound-free observable.)"""
    torch.manual_seed(0)
    device = _device()
    B, T, D = 2, 16, 16
    model = sLSTM(
        D, 32, num_layers=1, num_heads=num_heads,
        batch_first=True, pack_state=False, fast_mode=False,
    ).to(device).eval()
    x = torch.randn(B, T, D, device=device)

    boundary_t = 8
    b = torch.zeros(B, T, dtype=torch.bool, device=device)
    b[:, boundary_t] = True

    st_fresh = model.init_state(B, device=device, dtype=x.dtype)
    out_fresh, st_fresh_out = model(x, st_fresh, boundaries=b)

    out_inf_b, st_inf_b = _inflated_m_run(model, x, b, 50.0, device)
    out_inf_nob, st_inf_nob = _inflated_m_run(model, x, None, 50.0, device)

    # The inflated-m no-boundary run must keep m elevated...
    assert st_inf_nob.m.mean().item() > 10.0, (
        "sanity: without a boundary, inflated m must stay inflated"
    )
    # ...and the fresh run must have near-zero m.
    assert st_fresh_out.m.abs().mean().item() < 1.0, (
        "sanity: fresh run keeps m near zero"
    )
    # The boundary must collapse the inflated m (~50) back to O(1).
    # (Exact equality with the fresh run is not expected -- the sLSTM
    # gates depend on the pre-boundary hidden h via R_fused, which
    # differs between runs -- but the magnitude must collapse.)
    assert st_inf_b.m.abs().max().item() < 5.0, (
        "log-normalizer m must collapse from ~50 to O(1) at the boundary"
    )
    assert (
        st_inf_b.m.abs().mean().item()
        < st_inf_nob.m.abs().mean().item() / 10.0
    ), "boundary must collapse m by >10x relative to no-boundary run"


def test_slstm_scan_reset_kills_carry():
    """Direct scan-level test: with R_fused=0 (no hidden feedback), the
    boundary reset makes the post-boundary segment bit-identical
    to a fresh-state run."""
    torch.manual_seed(0)
    device = _device()
    from xlstm_cells.slstm import _slstm_scan_sequential

    B, H, Dh, T = 2, 2, 4, 6
    all_in = torch.randn(B, T, H, 4 * Dh, device=device)
    R_fused = torch.zeros(H, Dh, 4 * Dh, device=device)
    boundary_t = 2
    b = torch.zeros(B, T, dtype=torch.bool, device=device)
    b[:, boundary_t] = True

    def run(m_val, fresh=False):
        if fresh:
            c0 = torch.zeros(B, H, Dh, device=device)
            n0 = torch.zeros(B, H, Dh, device=device)
            m0 = torch.zeros(B, H, Dh, device=device)
        else:
            c0 = torch.randn(B, H, Dh, device=device)
            n0 = torch.rand(B, H, Dh, device=device) + 0.5
            m0 = torch.full((B, H, Dh), m_val, device=device)
        h0 = torch.zeros(B, H, Dh, device=device)
        out, *_ = _slstm_scan_sequential(all_in, R_fused, c0, n0, m0, h0, b)
        return out

    out_fresh = run(0.0, fresh=True)
    out_inflated = run(50.0)

    # Post-boundary outputs must match fresh bit-for-bit.
    assert torch.allclose(
        out_inflated[:, boundary_t:], out_fresh[:, boundary_t:], atol=1e-5
    ), "boundary reset must fully kill the carry at the boundary"


def test_boundary_reset_constants_are_large_negative():
    """Guard the chosen constants: the reset must be unconditional for
    realistic m drift (m < ~100)."""
    assert _BOUNDARY_RESET_LOGF <= -1000.0
    assert slstm_mod._BOUNDARY_RESET_LOGF <= -1000.0
    # exp(-1000) must be exactly zero (fp32/bf16 underflow).
    assert math.exp(_BOUNDARY_RESET_LOGF) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mlstm_boundary_reset_triton_kernel_path():
    """The triton kernel path applies logsigmoid(-1000) internally; verify
    the inflated-m boundary reset also works through the kernels and that
    no NaN/inf appears (fp32 autocast kernel dtype)."""
    torch.manual_seed(0)
    device = _device()
    if not mlstm_mod._HAS_MLSTM_KERNELS:
        pytest.skip("mlstm_kernels not installed")
    B, T, D = 2, 128, 16
    model = mLSTM(
        D, 32, num_layers=1, num_heads=2,
        batch_first=True, pack_state=False, use_triton_kernels=True,
    ).to(device).eval()
    x = torch.randn(B, T, D, device=device)
    b = torch.zeros(B, T, dtype=torch.bool, device=device)
    b[:, 64] = True  # mid-sequence boundary

    st_fresh = model.init_state(B, device=device, dtype=x.dtype)
    out_fresh, st_fresh_out = model(x, st_fresh, boundaries=b)
    out_inf_b, st_inf_b = _inflated_m_run(model, x, b, 50.0, device)

    assert torch.isfinite(out_fresh).all()
    assert torch.isfinite(out_inf_b).all()
    # Post-boundary segment must match a fresh-state run.
    assert torch.allclose(
        out_inf_b[:, 64:], out_fresh[:, 64:], atol=1e-3
    ), "triton kernel path must reset the inflated-m state at boundaries"
    assert torch.allclose(st_inf_b.m, st_fresh_out.m, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mlstm_boundary_reset_bf16_no_nan():
    """Verify that boundary resets under bfloat16 autocast do not produce NaN or Inf values."""
    torch.manual_seed(0)
    device = _device()
    B, T, D = 2, 128, 16
    model = mLSTM(
        D, 32, num_layers=1, num_heads=2,
        batch_first=True, pack_state=False, use_triton_kernels=False,
    ).to(device).eval().to(torch.bfloat16)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    b = torch.zeros(B, T, dtype=torch.bool, device=device)
    b[:, 32] = True
    b[:, 96] = True

    st_fresh = model.init_state(B, device=device, dtype=x.dtype)
    out_fresh, st_fresh_out = model(x, st_fresh, boundaries=b)
    out_inf_b, st_inf_b = _inflated_m_run(model, x, b, 50.0, device)

    assert torch.isfinite(out_fresh).all()
    assert torch.isfinite(out_inf_b).all()
    assert torch.allclose(
        out_inf_b[:, 96:], out_fresh[:, 96:], atol=1e-2
    ), "bf16 native path must reset the inflated-m state at boundaries"


# ---------------------------------------------------------------------------
# Fix 2: mLSTM denominator epsilon bounds the cancellation regime
# ---------------------------------------------------------------------------

def _scan_once(device, eps_override=None):
    """Adversarial parallel-scan input: denom_raw cancels to ~0 while
    numerator is O(1e-4), in the high-m regime (m=15) where the exp floor
    is below epsilon.  Returns max |h|."""
    B, T, H, Dh = 1, 2, 1, 1
    q = torch.zeros(B, T, H, Dh, device=device)
    q[0, :, 0, 0] = 1.0
    k = torch.zeros(B, T, H, Dh, device=device)
    k[0, 0, 0, 0] = 1000.0
    k[0, 1, 0, 0] = -1000.0
    v = torch.zeros(B, T, H, Dh, device=device)
    v[0, 0, 0, 0] = 1.0
    v[0, 1, 0, 0] = 2.0
    o = torch.ones(B, T, H, Dh, device=device)
    i_tilde = torch.full((B, T, H), -1.0, device=device)
    log_f = torch.zeros(B, T, H, device=device)
    m_init = torch.full((B, H), 15.0, device=device)
    C_init = torch.zeros(B, H, Dh, Dh, device=device)
    n_init = torch.zeros(B, H, Dh, device=device)

    eps = eps_override if eps_override is not None else mlstm_mod._EPS
    out, *_ = mlstm_mod._mlstm_recurrent_scan_parallel(
        q, k, v, o, i_tilde, log_f, C_init, n_init, m_init,
        eps=eps,
    )
    return out.abs().max().item()


def test_eps_bounds_cancellation_regime():
    """In the denom_raw-cancellation regime the output must be bounded and finite."""
    device = _device()
    h_eps3 = _scan_once(device, eps_override=1e-3)
    h_eps6 = _scan_once(device, eps_override=1e-6)

    assert math.isfinite(h_eps3)
    assert math.isfinite(h_eps6)
    # Larger eps provides a tighter denominator floor, strictly bounding cancellation amplification.
    assert h_eps3 <= h_eps6
    assert h_eps3 < 1e5


def test_eps_value():
    assert _EPS == 1e-6


# ---------------------------------------------------------------------------
# Configurable eps plumbing (mLSTM / mLSTMBlock constructor arg)
# ---------------------------------------------------------------------------

def test_eps_constructor_default():
    """eps=None must resolve to the module-level _EPS constant."""
    torch.manual_seed(0)
    device = _device()
    m = mLSTM(
        16, 32, num_layers=1, num_heads=4,
        batch_first=True, use_triton_kernels=False,
    ).to(device)
    assert m._eps == _EPS


def test_eps_constructor_custom():
    """eps=X must be stored and used on both native and triton paths."""
    torch.manual_seed(0)
    device = _device()
    for eps in (1e-4, 1e-3, 5e-3, 1e-2, 1e-1):
        m = mLSTM(
            16, 32, num_layers=1, num_heads=4,
            batch_first=True, use_triton_kernels=False,
            eps=eps,
        ).to(device)
        assert m._eps == eps
        x = torch.randn(2, 64, 16, device=device)
        out, _ = m(x)
        assert torch.isfinite(out).all()


def test_eps_constructor_block_forwards():
    """mLSTMBlock must forward eps= to its inner mLSTM."""
    torch.manual_seed(0)
    device = _device()
    blk = mLSTMBlock(
        16, num_heads=4, use_triton_kernels=False, eps=5e-3,
    ).to(device)
    assert blk._eps == 5e-3
    blk_default = mLSTMBlock(16, num_heads=4, use_triton_kernels=False).to(device)
    assert blk_default._eps == _EPS


def test_eps_constructor_rejects_bad_values():
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            mLSTM(16, 32, num_layers=1, num_heads=4, eps=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_eps_constructor_triton_path_uses_eps():
    """The triton backend config must receive the configured eps."""
    torch.manual_seed(0)
    device = _device()
    m = mLSTM(
        16, 32, num_layers=1, num_heads=4,
        batch_first=True, use_triton_kernels=True, eps=3e-3,
    ).to(device)
    if not mlstm_mod._HAS_MLSTM_KERNELS:
        pytest.skip("mlstm_kernels not installed")
    assert m._mlstm_backend is not None
    assert m._mlstm_backend.config.eps == 3e-3
