"""Tests for activation checkpointing, document boundary resets, and TBPTT training under non-reentrant checkpointing.

The packed+ckpt path verifies:
  1. Forward+backward completes without raising on Triton and native scan paths.
  2. Per-parameter gradients match the no-checkpoint baseline within tolerance across detached and frozen inputs.
  3. TBPTT training: detached states between chunks + frozen embedding + boundaries + use_checkpoint=True
     produces non-zero gradients on every trainable block parameter.
"""

import pytest
import torch
import torch.nn as nn

from xlstm_cells import mLSTMBlock
from xlstm_cells._utils import detach_states, PackedBoundariesMode

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
TOL_GRAD_ABS = 5e-2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return DEV


def _build(use_ckpt: bool, **overrides) -> mLSTMBlock:
    kw = dict(
        d_model=64, expand_factor=2, num_heads=4, conv_kernel=4,
        use_checkpoint=use_ckpt, use_triton_kernels=False, chunk_size=64,
    )
    kw.update(overrides)
    return mLSTMBlock(**kw).to(DEV).to(DT)


def _build_triton(use_ckpt: bool) -> mLSTMBlock:
    return mLSTMBlock(
        d_model=64, expand_factor=2, num_heads=4, conv_kernel=4,
        use_checkpoint=use_ckpt, use_triton_kernels=True,
        chunkwise_kernel="limit_chunk", chunk_size=64,
    ).to(DEV).to(DT)


def _grads(model: nn.Module) -> dict:
    return {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }


def _grad_diff(a: dict, b: dict, tol: float = TOL_GRAD_ABS) -> list:
    out = []
    for n, ga in a.items():
        gb = b.get(n)
        if ga is None or gb is None:
            out.append((n, "None"))
            continue
        d = (ga.float() - gb.float()).abs().max().item()
        if d > tol:
            out.append((n, d))
    return out


# ---------------------------------------------------------------------------
# 1. Forward+backward must not raise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["native", "triton"])
def test_packed_ckpt_forward_backward_does_not_raise(kernel):
    blk = _build_triton(True) if kernel == "triton" else _build(True)
    x = torch.randn(2, 128, 64, device=DEV, dtype=DT, requires_grad=True)
    b = torch.zeros(2, 128, dtype=torch.bool, device=DEV)
    b[:, [32, 96]] = True
    out, _ = blk(x, boundaries=b)
    out.float().pow(2).mean().backward()
    n_total = sum(1 for _ in blk.parameters())
    n_nz = sum(1 for p in blk.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert n_nz == n_total, f"only {n_nz}/{n_total} params got non-zero grads"


def test_packed_ckpt_supports_DISABLE_CKPT_IN_PACKED():
    """The legacy PASSIVE knob still silences ckpt when set."""
    from xlstm_cells._utils import set_packed_boundaries_override_mode
    prev = set_packed_boundaries_override_mode(PackedBoundariesMode.DISABLE_CKPT_IN_PACKED)
    try:
        # We can't easily observe that ckpt was skipped without monkeypatching,
        # but we can confirm the path completes without raising.
        blk = _build(True)
        x = torch.randn(2, 128, 64, device=DEV, dtype=DT, requires_grad=True)
        b = torch.zeros(2, 128, dtype=torch.bool, device=DEV)
        b[:, [32, 96]] = True
        out, _ = blk(x, boundaries=b)
        out.float().pow(2).mean().backward()
        assert all(p.grad is not None for p in blk.parameters())
    finally:
        set_packed_boundaries_override_mode(prev)


# ---------------------------------------------------------------------------
# 2. Per-parameter gradient equivalence vs no-ckpt baseline (no F1 trap).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["native", "triton"])
@pytest.mark.parametrize(
    "x_factory_name",
    ["leaf_randr", "requires_grad_True", "frozen_embedding"],
)
def test_packed_ckpt_grad_matches_no_ckpt(kernel, x_factory_name, monkeypatch):
    """The non-reentrant ckpt path must produce per-parameter grads
    identical (within bf16) to the no-ckpt baseline, regardless of
    whether the input tensor requires grad."""
    if kernel == "triton":
        factory_ckpt = lambda: _build_triton(True)
        factory_ref  = lambda: _build_triton(False)
    else:
        factory_ckpt = lambda: _build(True)
        factory_ref  = lambda: _build(False)
    torch.manual_seed(0)
    a = factory_ref()
    b = factory_ckpt()
    b.load_state_dict(a.state_dict(), strict=True)

    boundaries = torch.zeros(2, 256, dtype=torch.bool, device=DEV)
    boundaries[:, [32, 96, 160, 224]] = True
    if x_factory_name == "leaf_randr":
        x_a = torch.randn(2, 256, 64, device=DEV, dtype=DT)
        x_b = x_a
    elif x_factory_name == "requires_grad_True":
        x_a = torch.randn(2, 256, 64, device=DEV, dtype=DT).requires_grad_(True)
        x_b = x_a
    else:
        emb = nn.Embedding(64, 64).to(DEV).to(DT)
        emb.weight.requires_grad_(False)
        idx = torch.randint(0, 64, (2, 256), device=DEV)
        x_a = emb(idx)
        x_b = x_a

    out_a, _ = a(x_a, boundaries=boundaries)
    a.zero_grad()
    out_a.float().pow(2).mean().backward()
    out_b, _ = b(x_b, boundaries=boundaries)
    b.zero_grad()
    out_b.float().pow(2).mean().backward()

    fwd_diff = (out_a.float() - out_b.float()).abs().max().item()
    assert fwd_diff < 5e-2, f"forward parity violated: max diff {fwd_diff:.3e}"
    failed = _grad_diff(_grads(a), _grads(b))
    assert not failed, f"per-parameter gradients diverge: {failed[:3]}"


# ---------------------------------------------------------------------------
# 3. TBPTT training loop: detached states + frozen embedding still trains.
# ---------------------------------------------------------------------------

def test_tbptt_frozen_embedding_loop_produces_non_zero_grads():
    """Verify that a two-step TBPTT training loop with frozen embeddings,
    document boundaries, and activation checkpointing produces non-zero
    gradients on all trainable block parameters."""
    torch.manual_seed(0)
    emb = nn.Embedding(64, 64, device=DEV).to(DT)
    emb.weight.requires_grad_(False)
    head = nn.Linear(64, 64, bias=False).to(DEV).to(DT)
    blk = _build_triton(True)

    boundaries = torch.zeros(2, 128, dtype=torch.bool, device=DEV)
    boundaries[:, [32, 96]] = True

    n_total_block_params = sum(1 for _ in blk.parameters())
    n_nz = 0
    for step in range(2):
        idx = torch.randint(0, 64, (2, 128), device=DEV)
        labels = torch.randint(0, 64, (2, 128), device=DEV)
        x = emb(idx)
        out, states = blk(x, boundaries=boundaries)
        states = detach_states(states)
        out, states = blk(x, states, boundaries=boundaries)
        logits = head(out)
        loss = nn.functional.cross_entropy(
            logits.float().reshape(-1, 64), labels.reshape(-1),
        )
        emb.zero_grad(); head.zero_grad(); blk.zero_grad()
        loss.backward()
        n_nz = sum(1 for p in blk.parameters()
                   if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert n_nz == n_total_block_params, (
        f"frozen-emb TBPTT loop: only {n_nz}/{n_total_block_params} block "
        f"params received non-zero grads (silent grad-loss regression)"
    )


# ---------------------------------------------------------------------------
# 4. End-to-end: packed docs ending exactly at T-1 work correctly through
#    the full mLSTMBlock forward (sanity check the conv fix is wired in).
# ---------------------------------------------------------------------------

def test_block_forward_handles_doc_ending_at_last_position():
    """Verify that document boundaries at the last position (T-1) are handled correctly."""
    torch.manual_seed(1234)
    blk = _build_triton(True)
    x_ref = torch.randn(1, 16, 64, device=DEV, dtype=DT, requires_grad=True)
    # No boundary
    out_ref, _ = blk(x_ref, boundaries=None)
    # Boundary at T-1
    b = torch.zeros(1, 16, dtype=torch.bool, device=DEV)
    b[:, 15] = True
    out_with_boundary, _ = blk(x_ref, boundaries=b)
    assert not torch.allclose(out_ref.float(), out_with_boundary.float(), atol=1e-3), (
        "boundary at T-1 produced no measurable effect on block output"
    )
