"""Tests for xlstm_cells: mLSTM and sLSTM."""

import pytest
import torch
import torch.nn.functional as F

from xlstm_cells import (
    mLSTMCell,
    mLSTMState,
    mLSTM,
    mLSTMBlock,
    sLSTMCell,
    sLSTMState,
    sLSTM,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 4
T = 32
D = 64


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def x_seq(device):
    return torch.randn(B, T, D, device=device)


@pytest.fixture(scope="module")
def x_step(device):
    return torch.randn(B, D, device=device)


# helper: detach a tuple of states (the new per-layer return format)
def _detach(states):
    return tuple(s.detach() for s in states)


# ---------------------------------------------------------------------------
# mLSTMCell
# ---------------------------------------------------------------------------

class TestmLSTMCell:
    def test_forward_shape(self, device, x_step):
        cell = mLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)
        h, new_state = cell(x_step, state)
        assert h.shape == (B, D)
        assert new_state.C.shape == (B, 4, D // 4, D // 4)
        assert new_state.n.shape == (B, 4, D // 4)
        assert new_state.m.shape == (B, 4)

    def test_gradient_flow(self, device, x_step):
        cell = mLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)
        h, _ = cell(x_step, state)
        loss = h.sum()
        loss.backward()
        for name, p in cell.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert not torch.isnan(p.grad).any(), f"{name} grad is NaN"

    def test_stateful_step(self, device):
        cell = mLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)

        for _ in range(4):
            x_t = torch.randn(B, D, device=device)
            h, state = cell(x_t, state)

        assert not torch.allclose(h, torch.zeros_like(h))

    def test_detach(self, device):
        cell = mLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)
        x_t = torch.randn(B, D, device=device)
        h, state = cell(x_t, state)
        detached = state.detach()
        assert not detached.C.requires_grad
        assert not detached.n.requires_grad
        assert not detached.m.requires_grad

    def test_different_heads(self, device):
        for h in [2, 4, 8]:
            cell = mLSTMCell(32, 64, num_heads=h).to(device)
            state = cell.init_state(2, device=device)
            x_t = torch.randn(2, 32, device=device)
            ht, _ = cell(x_t, state)
            assert ht.shape == (2, 64)


# ---------------------------------------------------------------------------
# mLSTM (multi-layer, bidirectional)
# ---------------------------------------------------------------------------

class TestmLSTM:
    def test_single_layer_output(self, device, x_seq):
        layer = mLSTM(D, D, num_layers=1, batch_first=True).to(device)
        out, states = layer(x_seq)
        assert out.shape == (B, T, D)
        assert len(states) == 1
        # state[0].C shape: (num_directions=1, B, H=4, Dh=16, Dh=16)
        assert states[0].C.shape == (1, B, 4, D // 4, D // 4)

    def test_multi_layer_output(self, device, x_seq):
        layer = mLSTM(D, D, num_layers=3, batch_first=True).to(device)
        out, states = layer(x_seq)
        assert out.shape == (B, T, D)
        assert len(states) == 3
        for s in states:
            assert s.C.shape[0] == 1  # num_directions

    def test_bidirectional_shape(self, device, x_seq):
        layer = mLSTM(D, D, num_layers=2, bidirectional=True, batch_first=True).to(device)
        out, states = layer(x_seq)
        assert out.shape == (B, T, D * 2)
        assert len(states) == 2
        # each per-layer state has 2 directions
        for s in states:
            assert s.C.shape[0] == 2

    def test_batch_first_false(self, device):
        layer = mLSTM(D, D, batch_first=False).to(device)
        x = torch.randn(T, B, D, device=device)
        out, states = layer(x)
        assert out.shape == (T, B, D)

    def test_dropout(self, device, x_seq):
        layer = mLSTM(D, D, num_layers=3, dropout=0.5).to(device)
        layer.train()
        out_train, _ = layer(x_seq)
        layer.eval()
        out_eval, _ = layer(x_seq)
        assert out_train.shape == out_eval.shape

    def test_tbptt_chunks(self, device):
        layer = mLSTM(D, D, num_layers=2, batch_first=True).to(device)
        full_seq = torch.randn(B, T, D, device=device)
        target = torch.randn(B, T, D, device=device)

        chunk_size = 8
        state = None
        losses = []
        for start in range(0, T, chunk_size):
            chunk = full_seq[:, start:start + chunk_size]
            y = target[:, start:start + chunk_size]
            out, state = layer(chunk, state)
            loss = F.mse_loss(out, y)
            losses.append(loss.item())
            loss.backward()
            state = _detach(state)
        assert len(losses) == T // chunk_size

    def test_tbptt_gradients_flow(self, device):
        layer = mLSTM(D, D, batch_first=True).to(device)
        x = torch.randn(B, 8, D, device=device)
        target = torch.randn(B, 8, D, device=device)
        out, _ = layer(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_initial_state_carries(self, device):
        layer = mLSTM(D, D, batch_first=True).to(device)
        x = torch.randn(B, T, D, device=device)

        out1, s1_tuple = layer(x)

        out_a, sa_tuple = layer(x[:, :T // 2])
        out_b, sb_tuple = layer(x[:, T // 2:], sa_tuple)

        combined = torch.cat([out_a, out_b], dim=1)

        for s1, sb in zip(s1_tuple, sb_tuple):
            assert torch.allclose(s1.C, sb.C, atol=1e-4)
            assert torch.allclose(s1.n, sb.n, atol=1e-4)
            assert torch.allclose(s1.m, sb.m, atol=1e-4)

    def test_flatten_parameters(self, device):
        layer = mLSTM(D, D, batch_first=True).to(device)
        layer.flatten_parameters()

    def test_reset_parameters(self, device):
        layer = mLSTM(D, D, num_layers=2, batch_first=True).to(device)
        layer.reset_parameters()
        x = torch.randn(B, T, D, device=device)
        layer(x)


# ---------------------------------------------------------------------------
# sLSTMCell
# ---------------------------------------------------------------------------

class TestsLSTMCell:
    def test_forward_shape(self, device, x_step):
        cell = sLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)
        h, new_state = cell(x_step, state)
        assert h.shape == (B, D)
        assert new_state.c.shape == (B, 4, D // 4)
        assert new_state.n.shape == (B, 4, D // 4)
        assert new_state.m.shape == (B, 4, D // 4)
        assert new_state.h.shape == (B, 4, D // 4)

    def test_gradient_flow(self, device, x_step):
        cell = sLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)
        h, _ = cell(x_step, state)
        loss = h.sum()
        loss.backward()
        for name, p in cell.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert not torch.isnan(p.grad).any(), f"{name} grad is NaN"

    def test_recurrence_changes_output(self, device):
        cell = sLSTMCell(D, D, num_heads=4).to(device)
        state = cell.init_state(B, device=device)

        x_seed = torch.randn(B, D, device=device)
        h_seed, state = cell(x_seed, state)
        x_zeros = torch.zeros(B, D, device=device)
        h0, state = cell(x_zeros, state)
        h1, state = cell(x_zeros, state)
        assert not torch.allclose(h0, h1)

    def test_different_heads(self, device):
        for h in [2, 4, 8]:
            cell = sLSTMCell(32, 64, num_heads=h).to(device)
            state = cell.init_state(2, device=device)
            x_t = torch.randn(2, 32, device=device)
            ht, _ = cell(x_t, state)
            assert ht.shape == (2, 64)


# ---------------------------------------------------------------------------
# sLSTM (multi-layer, bidirectional)
# ---------------------------------------------------------------------------

class TestsLSTM:
    def test_single_layer_output(self, device, x_seq):
        layer = sLSTM(D, D, num_layers=1, batch_first=True).to(device)
        out, states = layer(x_seq)
        assert out.shape == (B, T, D)
        assert len(states) == 1
        assert states[0].c.shape == (1, B, 4, D // 4)

    def test_multi_layer_output(self, device, x_seq):
        layer = sLSTM(D, D, num_layers=3, batch_first=True).to(device)
        out, states = layer(x_seq)
        assert out.shape == (B, T, D)
        assert len(states) == 3
        for s in states:
            assert s.c.shape[0] == 1

    def test_bidirectional_shape(self, device, x_seq):
        layer = sLSTM(D, D, num_layers=2, bidirectional=True, batch_first=True).to(device)
        out, states = layer(x_seq)
        assert out.shape == (B, T, D * 2)
        assert len(states) == 2
        for s in states:
            assert s.c.shape[0] == 2

    def test_batch_first_false(self, device):
        layer = sLSTM(D, D, batch_first=False).to(device)
        x = torch.randn(T, B, D, device=device)
        out, states = layer(x)
        assert out.shape == (T, B, D)

    def test_dropout(self, device, x_seq):
        layer = sLSTM(D, D, num_layers=3, dropout=0.5).to(device)
        layer.train()
        out_train, _ = layer(x_seq)
        layer.eval()
        out_eval, _ = layer(x_seq)
        assert out_train.shape == out_eval.shape

    def test_tbptt_chunks(self, device):
        layer = sLSTM(D, D, num_layers=2, batch_first=True).to(device)
        full_seq = torch.randn(B, T, D, device=device)
        target = torch.randn(B, T, D, device=device)

        chunk_size = 8
        state = None
        losses = []
        for start in range(0, T, chunk_size):
            chunk = full_seq[:, start:start + chunk_size]
            y = target[:, start:start + chunk_size]
            out, state = layer(chunk, state)
            loss = F.mse_loss(out, y)
            losses.append(loss.item())
            loss.backward()
            state = _detach(state)
        assert len(losses) == T // chunk_size

    def test_tbptt_gradients_flow(self, device):
        layer = sLSTM(D, D, batch_first=True).to(device)
        x = torch.randn(B, 8, D, device=device)
        target = torch.randn(B, 8, D, device=device)
        out, _ = layer(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_initial_state_carries(self, device):
        layer = sLSTM(D, D, batch_first=True).to(device)
        x = torch.randn(B, T, D, device=device)
        out1, s1_tuple = layer(x)
        out_a, sa_tuple = layer(x[:, :T // 2])
        out_b, sb_tuple = layer(x[:, T // 2:], sa_tuple)
        for s1, sb in zip(s1_tuple, sb_tuple):
            assert torch.allclose(s1.c, sb.c, atol=1e-4)
            assert torch.allclose(s1.n, sb.n, atol=1e-4)
            assert torch.allclose(s1.m, sb.m, atol=1e-4)
            assert torch.allclose(s1.h, sb.h, atol=1e-4)

    def test_flatten_parameters(self, device):
        layer = sLSTM(D, D, batch_first=True).to(device)
        layer.flatten_parameters()

    def test_reset_parameters(self, device):
        layer = sLSTM(D, D, num_layers=2, batch_first=True).to(device)
        layer.reset_parameters()
        x = torch.randn(B, T, D, device=device)
        layer(x)


# ---------------------------------------------------------------------------
# nn.LSTM interface parity
# ---------------------------------------------------------------------------

class TestInterfaceParity:
    def test_mlstm_like_nn_lstm(self, device):
        """Test that mLSTM can be dropped in where nn.LSTM expects."""
        x = torch.randn(B, T, D, device=device)

        torch_lstm = torch.nn.LSTM(D, D, num_layers=2, batch_first=True, bidirectional=True).to(device)
        torch_out, (torch_h, torch_c) = torch_lstm(x)

        mlstm = mLSTM(D, D, num_layers=2, batch_first=True, bidirectional=True).to(device)
        mlstm_out, mlstm_states = mlstm(x)

        assert torch_out.shape == mlstm_out.shape
        assert len(mlstm_states) == 2
        # torch_h: (D*L, B, Hs) = (4, B, D)
        assert mlstm_states[0].C.shape == (2, B, 4, D // 4, D // 4)

    def test_slstm_like_nn_lstm(self, device):
        """Test that sLSTM conforms to the same shapes."""
        x = torch.randn(B, T, D, device=device)

        torch_lstm = torch.nn.LSTM(D, D, num_layers=1, batch_first=True).to(device)
        torch_out, _ = torch_lstm(x)

        slstm = sLSTM(D, D, num_layers=1, batch_first=True).to(device)
        slstm_out, _ = slstm(x)

        assert torch_out.shape == slstm_out.shape


# ---------------------------------------------------------------------------
# Correctness: cell stepping vs layer forward
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_mlstm_cell_vs_layer(self, device):
        """Cell stepping and layer forward should produce identical outputs."""
        torch.manual_seed(42)
        HS = D
        NH = 4
        Dh = HS // NH

        cell = mLSTMCell(D, HS, num_heads=NH).to(device)
        state = cell.init_state(B, device=device)

        x = torch.randn(B, 10, D, device=device)

        # Manual cell stepping
        state_man = mLSTMState(
            C=torch.zeros_like(state.C),
            n=torch.zeros_like(state.n),
            m=torch.zeros_like(state.m),
        )
        manual_outs = []
        for t in range(10):
            h, state_man = cell(x[:, t], state_man)
            manual_outs.append(h)
        manual_outs = torch.stack(manual_outs, dim=1)

        # Layer with same weights
        layer = mLSTM(D, HS, num_layers=1, batch_first=True).to(device)

        # Build layer state dict from cell parameters
        cell_sd = cell.state_dict()
        layer_state = {}
        for name, param in cell_sd.items():
            layer_state[f"{name.split('.')[0]}_0_0.{name.split('.')[1]}"] = param
        layer.load_state_dict(layer_state)

        opt_out, _ = layer(x)

        assert torch.allclose(manual_outs, opt_out, atol=1e-4), "Optimised mLSTM and manual differ"

    def test_slstm_cell_vs_layer(self, device):
        """Cell stepping and layer forward should produce identical outputs."""
        torch.manual_seed(42)
        HS = D
        NH = 4
        Dh = HS // NH

        cell = sLSTMCell(D, HS, num_heads=NH).to(device)
        state = cell.init_state(B, device=device)

        x = torch.randn(B, 10, D, device=device)

        # Manual cell stepping
        state_man = sLSTMState(
            c=torch.zeros_like(state.c),
            n=torch.zeros_like(state.n),
            m=torch.zeros_like(state.m),
            h=torch.zeros_like(state.h),
        )
        manual_outs = []
        for t in range(10):
            h, state_man = cell(x[:, t], state_man)
            manual_outs.append(h)
        manual_outs = torch.stack(manual_outs, dim=1)

        # Layer with same weights
        layer = sLSTM(D, HS, num_layers=1, batch_first=True).to(device)

        # Build layer state dict from cell's fused parameters
        cell_sd = cell.state_dict()
        layer_state = {}
        for name, param in cell_sd.items():
            # cell has W_all, R_fused -- layer has W_all_0_0, R_fused_0_0
            layer_state[name.replace("W_all", "W_all_0_0")
                            .replace("R_fused", "R_fused_0_0")] = param
        layer.load_state_dict(layer_state)

        opt_out, _ = layer(x)

        assert torch.allclose(manual_outs, opt_out, atol=1e-4), "Optimised sLSTM and manual differ"


# ---------------------------------------------------------------------------
# torch.compile smoketest
# ---------------------------------------------------------------------------

class TestCompile:
    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
    def test_mlstm_compile(self, device):
        layer = mLSTM(D, D, batch_first=True).to(device)
        compiled = torch.compile(layer)
        x = torch.randn(B, T, D, device=device)
        out, states = compiled(x)
        assert out.shape == (B, T, D)

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
    def test_slstm_compile(self, device):
        layer = sLSTM(D, D, batch_first=True).to(device)
        compiled = torch.compile(layer)
        x = torch.randn(B, T, D, device=device)
        out, states = compiled(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Parallel / optimized scan correctness
# ---------------------------------------------------------------------------

class TestParallelCorrectness:
    def test_mlstm_parallel_vs_sequential(self, device):
        from xlstm_cells.mlstm import _mlstm_recurrent_scan, _mlstm_recurrent_scan_parallel

        torch.manual_seed(123)
        B, T, H, Dh = 2, 16, 4, 8
        q = torch.randn(B, T, H, Dh, device=device)
        k = torch.randn(B, T, H, Dh, device=device) / Dh ** 0.5
        v = torch.randn(B, T, H, Dh, device=device)
        o = torch.sigmoid(torch.randn(B, T, H, Dh, device=device))
        i_tilde = torch.randn(B, T, H, device=device)
        log_f = F.logsigmoid(torch.randn(B, T, H, device=device))

        C0 = torch.zeros(B, H, Dh, Dh, device=device)
        n0 = torch.zeros(B, H, Dh, device=device)
        m0 = torch.zeros(B, H, device=device)

        out_seq, c_seq, n_seq, m_seq = _mlstm_recurrent_scan(
            q, k, v, o, i_tilde, log_f, C0.clone(), n0.clone(), m0.clone())
        out_par, c_par, n_par, m_par = _mlstm_recurrent_scan_parallel(
            q, k, v, o, i_tilde, log_f, C0.clone(), n0.clone(), m0.clone())

        assert torch.allclose(out_seq, out_par, atol=1e-4), "Parallel mLSTM outputs differ"
        assert torch.allclose(c_seq, c_par, atol=1e-4), "Parallel mLSTM final C differs"
        assert torch.allclose(n_seq, n_par, atol=1e-4), "Parallel mLSTM final n differs"
        assert torch.allclose(m_seq, m_par, atol=1e-4), "Parallel mLSTM final m differs"

    def test_mlstm_parallel_with_nonzero_state(self, device):
        from xlstm_cells.mlstm import _mlstm_recurrent_scan, _mlstm_recurrent_scan_parallel

        torch.manual_seed(42)
        B, T, H, Dh = 2, 16, 4, 8
        q = torch.randn(B, T, H, Dh, device=device)
        k = torch.randn(B, T, H, Dh, device=device) / Dh ** 0.5
        v = torch.randn(B, T, H, Dh, device=device)
        o = torch.sigmoid(torch.randn(B, T, H, Dh, device=device))
        i_tilde = torch.randn(B, T, H, device=device)
        log_f = F.logsigmoid(torch.randn(B, T, H, device=device))

        C_init = torch.randn(B, H, Dh, Dh, device=device) * 0.1
        n_init = torch.randn(B, H, Dh, device=device) * 0.1
        m_init = torch.randn(B, H, device=device) * 0.1

        out_seq, c_seq, n_seq, m_seq = _mlstm_recurrent_scan(
            q, k, v, o, i_tilde, log_f, C_init.clone(), n_init.clone(), m_init.clone())
        out_par, c_par, n_par, m_par = _mlstm_recurrent_scan_parallel(
            q, k, v, o, i_tilde, log_f, C_init.clone(), n_init.clone(), m_init.clone())

        assert torch.allclose(out_seq, out_par, atol=1e-4), "Parallel mLSTM (nonzero state) outputs differ"

    def test_slstm_optimized_vs_sequential(self, device):
        from xlstm_cells.slstm import _slstm_recurrent_scan, _slstm_scan_sequential

        torch.manual_seed(42)
        B, T, H, Dh = 2, 16, 4, 8
        z_in = torch.randn(B, T, H, Dh, device=device)
        i_in = torch.randn(B, T, H, Dh, device=device)
        f_in = torch.randn(B, T, H, Dh, device=device)
        o_in = torch.randn(B, T, H, Dh, device=device)
        Rz = torch.randn(H, Dh, Dh, device=device) * 0.1
        Ri = torch.randn(H, Dh, Dh, device=device) * 0.1
        Rf = torch.randn(H, Dh, Dh, device=device) * 0.1
        Ro = torch.randn(H, Dh, Dh, device=device) * 0.1

        c0 = torch.zeros(B, H, Dh, device=device)
        n0 = torch.zeros(B, H, Dh, device=device)
        m0 = torch.zeros(B, H, Dh, device=device)
        h0 = torch.zeros(B, H, Dh, device=device)

        out_seq, cs_s, ns_s, ms_s, hs_s = _slstm_recurrent_scan(
            z_in, i_in, f_in, o_in, Rz, Ri, Rf, Ro,
            c0.clone(), n0.clone(), m0.clone(), h0.clone())

        all_in = torch.cat([z_in, i_in, f_in, o_in], dim=-1)
        R_fused = torch.cat([Rz, Ri, Rf, Ro], dim=-1)
        out_opt, cs_o, ns_o, ms_o, hs_o = _slstm_scan_sequential(
            all_in, R_fused,
            c0.clone(), n0.clone(), m0.clone(), h0.clone())

        assert torch.allclose(out_seq, out_opt, atol=1e-4), "Optimized sLSTM outputs differ"


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_mlstm_checkpoint_forward(self, device):
        layer = mLSTM(D, D, batch_first=True, use_checkpoint=True).to(device)
        layer.train()
        x = torch.randn(B, T, D, device=device)
        out, states = layer(x)
        assert out.shape == (B, T, D)

    def test_mlstm_checkpoint_gradient(self, device):
        layer = mLSTM(D, D, batch_first=True, use_checkpoint=True).to(device)
        layer.train()
        x = torch.randn(B, T, D, device=device)
        out, _ = layer(x)
        loss = out.sum()
        loss.backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert not torch.isnan(p.grad).any(), f"{name} grad is NaN"

    def test_slstm_checkpoint_forward(self, device):
        layer = sLSTM(D, D, batch_first=True, use_checkpoint=True).to(device)
        layer.train()
        x = torch.randn(B, T, D, device=device)
        out, states = layer(x)
        assert out.shape == (B, T, D)

    def test_slstm_checkpoint_gradient(self, device):
        layer = sLSTM(D, D, batch_first=True, use_checkpoint=True).to(device)
        layer.train()
        x = torch.randn(B, T, D, device=device)
        out, _ = layer(x)
        loss = out.sum()
        loss.backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_mlstm_checkpoint_vs_no_checkpoint(self, device):
        """Checkpointing should produce identical forward pass."""
        torch.manual_seed(42)
        layer_ckpt = mLSTM(D, D, batch_first=True, use_checkpoint=True).to(device)
        layer_no = mLSTM(D, D, batch_first=True).to(device)
        layer_no.load_state_dict(layer_ckpt.state_dict())

        layer_ckpt.train()
        layer_no.eval()  # eval to avoid dropout differences; checkpoint runs in train mode

        x = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out_ckpt, _ = layer_ckpt(x)
            out_no, _ = layer_no(x)
        assert torch.allclose(out_ckpt, out_no, atol=1e-4), "Checkpoint forward differs"

    def test_mlstm_checkpoint_triton_uses_kernel_path(self, device):
        """Checkpointing must NOT force fallback to the native scan."""
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("Triton kernels require CUDA device")

        from xlstm_cells.mlstm import _HAS_MLSTM_KERNELS

        layer = mLSTM(D, D, batch_first=True, use_checkpoint=True,
                      use_triton_kernels=True).to(device)
        layer.train()
        layer._use_triton_kernels = layer._use_triton_kernels and _HAS_MLSTM_KERNELS
        if not layer._use_triton_kernels:
            pytest.skip("mlstm_kernels not installed")

        assert layer._use_triton_kernels, "triton should be requested and available"

        x = torch.randn(B, T * 4, D, device=device)  # T*4=128, divisible by chunk_size
        assert x.size(1) % layer._chunk_size == 0

        path = []

        def patched_kernels(*args, **kwargs):
            path.append("triton")
            return orig_kernels(*args, **kwargs)

        def patched_native(*args, **kwargs):
            path.append("native")
            return orig_native(*args, **kwargs)

        orig_kernels = layer._run_layer_kernels
        orig_native = layer._run_layer_native
        layer._run_layer_kernels = patched_kernels
        layer._run_layer_native = patched_native

        out, _ = layer(x)
        out.sum().backward()

        layer._run_layer_kernels = orig_kernels
        layer._run_layer_native = orig_native

        assert "triton" in path, f"triton kernel path never used (calls: {path})"
        assert "native" not in path, f"native scan used under checkpoint (calls: {path})"

    def test_mlstm_checkpoint_triton_matches_no_checkpoint(self, device):
        """checkpoint+triton must be numerically identical to triton alone."""
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("Triton kernels require CUDA device")

        from xlstm_cells.mlstm import _HAS_MLSTM_KERNELS

        if not _HAS_MLSTM_KERNELS:
            pytest.skip("mlstm_kernels not installed")

        torch.manual_seed(42)
        layer_ckpt = mLSTM(D, D, batch_first=True, use_checkpoint=True,
                           use_triton_kernels=True).to(device)
        layer_no = mLSTM(D, D, batch_first=True, use_triton_kernels=True).to(device)
        layer_no.load_state_dict(layer_ckpt.state_dict())

        layer_ckpt.train()
        layer_no.eval()

        x = torch.randn(B, T * 4, D, device=device)  # T*4=128, divisible by chunk_size
        with torch.no_grad():
            out_ckpt, _ = layer_ckpt(x)
            out_no, _ = layer_no(x)
        assert torch.allclose(out_ckpt, out_no, atol=1e-4), "checkpoint+triton differs from triton-only"


class TestTritonKernelSelection:
    def test_default_kernel_is_xl_chunk(self):
        layer = mLSTM(D, D, batch_first=True)
        assert layer._chunkwise_kernel == "xl_chunk"
        assert layer._chunk_size == 128

    def test_short_names_map_to_full_kernel_names(self):
        from xlstm_cells.mlstm import _TRITON_CHUNKWISE_KERNELS

        assert _TRITON_CHUNKWISE_KERNELS == {
            "limit_chunk": "chunkwise--triton_limit_chunk",
            "xl_chunk": "chunkwise--triton_xl_chunk",
        }

    def test_custom_kernel_and_chunk_size(self):
        layer = mLSTM(D, D, batch_first=True, use_triton_kernels=True,
                      chunkwise_kernel="xl_chunk", chunk_size=128)
        assert layer._chunkwise_kernel == "xl_chunk"
        assert layer._chunk_size == 128

    def test_invalid_kernel_name_raises(self):
        with pytest.raises(ValueError, match="xl_chunk"):
            mLSTM(D, D, batch_first=True, chunkwise_kernel="bogus")

    def test_block_passes_kernel_args_to_lstm(self):
        block = mLSTMBlock(D, num_heads=4, use_triton_kernels=True,
                           chunkwise_kernel="xl_chunk", chunk_size=128)
        assert block._chunkwise_kernel == "xl_chunk"
        assert block._chunk_size == 128

    def test_cpu_falls_back_to_native(self):
        """CPU tensors must not crash — fall back to native scan."""
        import warnings
        from xlstm_cells.mlstm import _HAS_MLSTM_KERNELS

        if not _HAS_MLSTM_KERNELS:
            pytest.skip("mlstm_kernels not installed")

        # Reset the module-level warning flag so we can capture it
        import xlstm_cells.mlstm as _mod
        _prev = _mod._triton_fallback_warned
        _mod._triton_fallback_warned = False

        try:
            layer = mLSTM(D, D, batch_first=True, use_triton_kernels=True)
            layer.train()
            x = torch.randn(B, T, D)  # CPU tensor

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out, _ = layer(x)

            assert out.shape == (B, T, D), "CPU forward must produce correct shape"
            msgs = [str(wi.message) for wi in w]
            assert any("CPU" in m or "CUDA" in m for m in msgs), (
                f"Expected a CPU fallback warning, got: {msgs}"
            )
        finally:
            _mod._triton_fallback_warned = _prev

    def test_siging_rejected(self):
        """xl_chunk_siging is not a supported kernel."""
        with pytest.raises(ValueError, match="xl_chunk"):
            mLSTM(D, D, batch_first=True, chunkwise_kernel="xl_chunk_siging")


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_mlstm_repr(self):
        layer = mLSTM(128, 256, num_layers=2, bidirectional=True)
        r = repr(layer)
        assert "mLSTM" in r
        assert "input_size=128" in r
        assert "hidden_size=256" in r
        assert "bidirectional=True" in r

    def test_slstm_repr(self):
        layer = sLSTM(128, 256, use_checkpoint=True)
        r = repr(layer)
        assert "sLSTM" in r
        assert "use_checkpoint=True" in r
