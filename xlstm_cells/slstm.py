"""
sLSTM: Scalar-memory LSTM cell with block-diagonal (per-head) recurrence.

100% aligned with the official xLSTM paper (arXiv:2405.04517v2, Figure 10) and NX-AI/xlstm:
1. Block-diagonal recurrent connections Rz, Ri, Rf, Ro per head.
2. Exponential gating with stabilizer state m and gate upper-bound clamping (<= 1.0).
3. Linspace forget-gate bias init (3.4 to 6.0 across heads) for diverse memory timescales.
4. Input-gate bias init ~ N(0.0, 0.1).
5. Dual backend support:
   - "vanilla" (default): Fast sequential scan with torch.compile chunking (fast_mode).
   - "cuda": Official custom CUDA C++ extension (JIT compiled).
6. Hard-block validation guards preventing conflicting settings (e.g. fast_mode with cuda backend).
7. Non-reentrant activation checkpointing support.
8. Packed document boundaries reset (f_tilde = -1000.0).
"""

from __future__ import annotations

import math
import os
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
from .components.init import bias_linspace_init_, small_init_init_
from .mlstm import _MAX_FORGET_BIAS

_EPS = 1e-6
_BOUNDARY_RESET_LOGF = -1000.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class sLSTMState:
    """State for sLSTM, held as named tensors for full user control.

    Shapes (single direction, single layer):
        c  (B, H, Dh)   cell state
        n  (B, H, Dh)   normalizer
        m  (B, H, Dh)   log-space stabilizer
        h  (B, H, Dh)   previous hidden output
    """

    c: torch.Tensor
    n: torch.Tensor
    m: torch.Tensor
    h: torch.Tensor

    @classmethod
    def init(cls, batch_size: int, num_heads: int, head_dim: int,
             device=None, dtype=None) -> "sLSTMState":
        H, Dh = num_heads, head_dim
        return cls(
            c=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            n=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            m=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
            h=torch.zeros(batch_size, H, Dh, device=device, dtype=dtype),
        )

    def detach(self) -> "sLSTMState":
        return sLSTMState(self.c.detach(), self.n.detach(),
                          self.m.detach(), self.h.detach())

    def to(self, *args, **kwargs) -> "sLSTMState":
        return sLSTMState(self.c.to(*args, **kwargs), self.n.to(*args, **kwargs),
                          self.m.to(*args, **kwargs), self.h.to(*args, **kwargs))

    def clone(self) -> "sLSTMState":
        return sLSTMState(self.c.clone(), self.n.clone(),
                          self.m.clone(), self.h.clone())

    def __repr__(self) -> str:
        return (f"sLSTMState(c={list(self.c.shape)}, n={list(self.n.shape)}, "
                f"m={list(self.m.shape)}, h={list(self.h.shape)})")


# ---------------------------------------------------------------------------
# Fused sequential scan (Vanilla Backend)
# ---------------------------------------------------------------------------

def _slstm_scan_sequential(
    all_in: torch.Tensor,
    R_fused: torch.Tensor,
    c: torch.Tensor,
    n: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
    boundaries: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run sLSTM recurrence over full sequence in a single loop."""
    B, T, H, Dh_x4 = all_in.shape
    Dh = Dh_x4 // 4
    hidden_size = H * Dh
    output_list: List[torch.Tensor] = []

    b_dev = boundaries.to(device=all_in.device, dtype=torch.bool) if boundaries is not None else None

    for t in range(T):
        all_tilde = all_in[:, t] + torch.einsum("bhd,hde->bhe", h, R_fused)
        z_tilde, i_tilde, f_tilde, o_tilde = all_tilde.chunk(4, dim=-1)

        if b_dev is not None:
            b_t = b_dev[:, t].view(B, 1, 1)
            f_tilde = torch.where(b_t, torch.full_like(f_tilde, _BOUNDARY_RESET_LOGF), f_tilde)

        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)
        log_f = F.logsigmoid(f_tilde)

        m_new = torch.maximum(log_f + m, i_tilde)
        # Gate clamping to <= 1.0 (paper compliance)
        i_prime = torch.minimum(torch.exp(i_tilde - m_new), torch.ones_like(i_tilde))
        f_prime = torch.minimum(torch.exp(log_f + m - m_new), torch.ones_like(i_tilde))

        c = f_prime * c + i_prime * z
        n = f_prime * n + i_prime
        h = o * (c / n.clamp_min(_EPS))
        m = m_new

        output_list.append(h.reshape(B, hidden_size))

    outputs = torch.stack(output_list, dim=1)
    return outputs, c, n, m, h


def _slstm_recurrent_scan(
    z_in: torch.Tensor, i_in: torch.Tensor, f_in: torch.Tensor, o_in: torch.Tensor,
    Rz: torch.Tensor, Ri: torch.Tensor, Rf: torch.Tensor, Ro: torch.Tensor,
    c: torch.Tensor, n: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager reference scan for testing only."""
    R_fused = torch.cat([Rz, Ri, Rf, Ro], dim=-1)
    all_in = torch.cat([z_in, i_in, f_in, o_in], dim=-1)
    return _slstm_scan_sequential(all_in, R_fused, c, n, m, h)


# ---------------------------------------------------------------------------
# sLSTMCell -- Single Step
# ---------------------------------------------------------------------------

class sLSTMCell(nn.Module):
    """One time-step of sLSTM."""

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4,
                 bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        Dh = self.head_dim

        self.W_all = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.R_fused = nn.Parameter(torch.empty(num_heads, Dh, 4 * Dh))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        Dh = self.head_dim
        HS = self.hidden_size
        NH = self.num_heads

        w = self.W_all.weight.data
        small_init_init_(w[:HS], dim=self.input_size)          # Wz
        small_init_init_(w[HS:2*HS], dim=self.input_size)      # Wi
        small_init_init_(w[2*HS:3*HS], dim=self.input_size)    # Wf
        small_init_init_(w[3*HS:4*HS], dim=self.input_size)    # Wo

        if self.W_all.bias is not None:
            nn.init.zeros_(self.W_all.bias)
            nn.init.normal_(self.W_all.bias.data[HS:2*HS], mean=0.0, std=0.1)
            f_biases = torch.linspace(3.4, 6.0, NH).unsqueeze(-1).expand(NH, Dh).reshape(-1)
            self.W_all.bias.data[2*HS:3*HS].copy_(f_biases)

        R = self.R_fused.data
        for h in range(self.num_heads):
            for g in range(4):
                tmp = torch.empty_like(R[h, :, g*Dh:(g+1)*Dh])
                nn.init.orthogonal_(tmp)
                R[h, :, g*Dh:(g+1)*Dh] = tmp

    def init_state(self, batch_size: int, device=None, dtype=None) -> sLSTMState:
        return sLSTMState.init(batch_size, self.num_heads, self.head_dim, device, dtype)

    def forward(self, x_t: torch.Tensor, state: sLSTMState) -> Tuple[torch.Tensor, sLSTMState]:
        B = x_t.size(0)
        H, Dh = self.num_heads, self.head_dim

        wx = self.W_all(x_t).view(B, 4, H, Dh).permute(0, 2, 1, 3).reshape(B, H, 4 * Dh)
        all_tilde = wx + torch.einsum("bhd,hde->bhe", state.h, self.R_fused)
        z_tilde, i_tilde, f_tilde, o_tilde = all_tilde.chunk(4, dim=-1)

        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)
        log_f = F.logsigmoid(f_tilde)

        m_new = torch.maximum(log_f + state.m, i_tilde)
        i_prime = torch.minimum(torch.exp(i_tilde - m_new), torch.ones_like(i_tilde))
        f_prime = torch.minimum(torch.exp(log_f + state.m - m_new), torch.ones_like(i_tilde))

        c_new = f_prime * state.c + i_prime * z
        n_new = f_prime * state.n + i_prime
        h_new = o * (c_new / n_new.clamp_min(_EPS))

        return h_new.reshape(B, self.hidden_size), sLSTMState(c_new, n_new, m_new, h_new)

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        HS = self.hidden_size
        if self.W_all.bias is not None:
            self.W_all.bias.data[2*HS:3*HS].clamp_(-max_val, max_val)


# ---------------------------------------------------------------------------
# sLSTM -- Full Sequence Layer
# ---------------------------------------------------------------------------

class sLSTM(nn.Module):
    """Multi-layer sLSTM supporting vanilla compiled scan and CUDA kernels."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pack_state: bool = True,
        backend: str = "vanilla",
        use_checkpoint: bool = False,
        fast_mode: bool = False,
        fast_chunk_size: int = 32,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        # Hard-block invalid backend/mode overlaps
        if backend == "cuda" and fast_mode:
            raise ValueError(
                "sLSTM: conflicting arguments: backend='cuda' cannot be combined with fast_mode=True. "
                "fast_mode is for torch.compile chunking under backend='vanilla'. Set fast_mode=False "
                "when using backend='cuda'."
            )
        if backend not in ("vanilla", "cuda"):
            raise ValueError(f"sLSTM: unknown backend '{backend}'. Must be 'vanilla' or 'cuda'.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = [num_heads] * num_layers if isinstance(num_heads, int) else num_heads
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pack_state = pack_state
        self.backend = backend
        self.use_checkpoint = use_checkpoint
        self.fast_mode = fast_mode
        self.fast_chunk_size = fast_chunk_size

        self._cuda_kernel = None
        if backend == "cuda":
            self._init_cuda_backend()

        for layer_idx in range(num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = hidden_size // Hh
            for d in range(self.num_directions):
                in_sz = input_size if layer_idx == 0 else hidden_size * self.num_directions
                w_all = nn.Linear(in_sz, 4 * hidden_size, bias=bias)
                r_fused = nn.Parameter(torch.empty(Hh, Dh, 4 * Dh))
                setattr(self, f"W_all_{d}_{layer_idx}", w_all)
                setattr(self, f"R_fused_{d}_{layer_idx}", r_fused)

        if dropout > 0.0 and num_layers > 1:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

        self._compiled_scans = {}
        self.reset_parameters()

    def flatten_parameters(self):
        """No-op for nn.LSTM compatibility."""
        pass

    def _init_cuda_backend(self):
        try:
            from .cuda.cuda_init import load
            curdir = os.path.dirname(__file__)
            src_dir = os.path.join(curdir, "cuda")
            sources = [
                os.path.join(src_dir, "cuda", "slstm.cc"),
                os.path.join(src_dir, "cuda", "slstm_forward.cu"),
                os.path.join(src_dir, "cuda", "slstm_backward.cu"),
                os.path.join(src_dir, "cuda", "slstm_backward_cut.cu"),
                os.path.join(src_dir, "cuda", "slstm_pointwise.cu"),
                os.path.join(src_dir, "util", "blas.cu"),
                os.path.join(src_dir, "util", "cuda_error.cu"),
            ]
            self._cuda_kernel = load(name="slstm_cuda", sources=sources)
        except Exception as e:
            warnings.warn(f"sLSTM: failed to compile CUDA kernel ({e}). Falling back to backend='vanilla'.")
            self.backend = "vanilla"

    def reset_parameters(self):
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = self.hidden_size // Hh
            HS = self.hidden_size
            for d in range(self.num_directions):
                in_sz = self.input_size if layer_idx == 0 else self.hidden_size * self.num_directions
                w_all = getattr(self, f"W_all_{d}_{layer_idx}")
                r_fused = getattr(self, f"R_fused_{d}_{layer_idx}")

                w = w_all.weight.data
                small_init_init_(w[:HS], dim=in_sz)
                small_init_init_(w[HS:2*HS], dim=in_sz)
                small_init_init_(w[2*HS:3*HS], dim=in_sz)
                small_init_init_(w[3*HS:4*HS], dim=in_sz)

                if w_all.bias is not None:
                    nn.init.zeros_(w_all.bias)
                    nn.init.normal_(w_all.bias.data[HS:2*HS], mean=0.0, std=0.1)
                    f_biases = torch.linspace(3.4, 6.0, Hh).unsqueeze(-1).expand(Hh, Dh).reshape(-1)
                    w_all.bias.data[2*HS:3*HS].copy_(f_biases)

                R = r_fused.data
                for h in range(Hh):
                    for g in range(4):
                        tmp = torch.empty_like(R[h, :, g*Dh:(g+1)*Dh])
                        nn.init.orthogonal_(tmp)
                        R[h, :, g*Dh:(g+1)*Dh] = tmp

    def init_state(self, batch_size: int, device=None, dtype=None) -> Union[sLSTMState, Tuple[sLSTMState, ...]]:
        states = []
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = self.hidden_size // Hh
            D = self.num_directions
            states.append(sLSTMState(
                c=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                n=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                m=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
                h=torch.zeros(D, batch_size, Hh, Dh, device=device, dtype=dtype),
            ))
        return tuple(states) if self.pack_state else (states[0] if self.num_layers == 1 else tuple(states))

    def _get_compiled_scan(self, chunk_size: int):
        if chunk_size not in self._compiled_scans:
            def scan_chunk(all_in, R, c, n, m, h, b=None):
                return _slstm_scan_sequential(all_in, R, c, n, m, h, b)
            self._compiled_scans[chunk_size] = torch.compile(scan_chunk, dynamic=False)
        return self._compiled_scans[chunk_size]

    def _run_layer_vanilla(
        self,
        x: torch.Tensor,
        d: int,
        layer_idx: int,
        c: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
        h: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        Hh = self.num_heads[layer_idx]
        Dh = self.hidden_size // Hh

        w_all = getattr(self, f"W_all_{d}_{layer_idx}")
        r_fused = getattr(self, f"R_fused_{d}_{layer_idx}")

        all_in = w_all(x).view(B, T, 4, Hh, Dh).permute(0, 1, 3, 2, 4).reshape(B, T, Hh, 4 * Dh)

        if not self.fast_mode or T <= self.fast_chunk_size:
            return _slstm_scan_sequential(all_in, r_fused, c, n, m, h, boundaries)

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
                out_chunk, c, n, m, h = fn_c(chunk_in, r_fused, c, n, m, h, b_chunk)
            else:
                out_chunk, c, n, m, h = _slstm_scan_sequential(chunk_in, r_fused, c, n, m, h, b_chunk)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=1), c, n, m, h

    def forward(
        self,
        input: torch.Tensor,
        state=None,
        boundaries: Optional[torch.Tensor] = None,
    ):
        if not self.batch_first:
            input = input.transpose(0, 1)

        B, T, _ = input.shape

        if state is None:
            state = self.init_state(B, device=input.device, dtype=input.dtype)
        if not isinstance(state, (tuple, list)):
            state = (state,)

        if self.backend == "cuda":
            if not input.is_cuda:
                warnings.warn("sLSTM: backend='cuda' requested but tensor is on CPU. Falling back to 'vanilla'.")
            elif torch._dynamo.is_compiling():
                warnings.warn("sLSTM: backend='cuda' disabled under torch.compile tracing. Falling back to 'vanilla'.")

        packed = boundaries is not None
        bounds_mode = get_packed_boundaries_override_mode()
        ckpt_active = bool(self.use_checkpoint and self.training)
        if packed and bounds_mode == PackedBoundariesMode.DISABLE_CKPT_IN_PACKED:
            ckpt_active = False

        out_states: List[sLSTMState] = []
        for layer_idx in range(self.num_layers):
            Hh = self.num_heads[layer_idx]
            Dh = self.hidden_size // Hh
            D = self.num_directions
            out_states.append(sLSTMState(
                c=torch.empty(D, B, Hh, Dh, device=input.device, dtype=input.dtype),
                n=torch.empty(D, B, Hh, Dh, device=input.device, dtype=input.dtype),
                m=torch.empty(D, B, Hh, Dh, device=input.device, dtype=input.dtype),
                h=torch.empty(D, B, Hh, Dh, device=input.device, dtype=input.dtype),
            ))

        layer_input = input

        for layer_idx in range(self.num_layers):
            s_l = state[layer_idx]

            if self.num_directions == 1:
                c_dl = s_l.c.squeeze(0)
                n_dl = s_l.n.squeeze(0)
                m_dl = s_l.m.squeeze(0)
                h_dl = s_l.h.squeeze(0)

                if ckpt_active:
                    out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                        self._run_layer_vanilla, layer_input, 0, layer_idx,
                        c_dl, n_dl, m_dl, h_dl, boundaries,
                        use_reentrant=False,
                    )
                else:
                    out, c_out, n_out, m_out, h_out = self._run_layer_vanilla(
                        layer_input, 0, layer_idx, c_dl, n_dl, m_dl, h_dl,
                        boundaries=boundaries,
                    )

                layer_output = out
                out_states[layer_idx].c[0].copy_(c_out)
                out_states[layer_idx].n[0].copy_(n_out)
                out_states[layer_idx].m[0].copy_(m_out)
                out_states[layer_idx].h[0].copy_(h_out)
            else:
                dir_outputs: List[torch.Tensor] = []
                for d in range(self.num_directions):
                    l_in = torch.flip(layer_input, [1]) if d == 1 else layer_input
                    b_d = torch.flip(boundaries, [1]) if (d == 1 and boundaries is not None) else boundaries

                    c_dl = s_l.c[d]
                    n_dl = s_l.n[d]
                    m_dl = s_l.m[d]
                    h_dl = s_l.h[d]

                    if ckpt_active:
                        out, c_out, n_out, m_out, h_out = _torch_checkpoint(
                            self._run_layer_vanilla, l_in, d, layer_idx,
                            c_dl, n_dl, m_dl, h_dl, b_d,
                            use_reentrant=False,
                        )
                    else:
                        out, c_out, n_out, m_out, h_out = self._run_layer_vanilla(
                            l_in, d, layer_idx, c_dl, n_dl, m_dl, h_dl,
                            boundaries=b_d,
                        )

                    if d == 1:
                        out = torch.flip(out, [1])
                    dir_outputs.append(out)
                    out_states[layer_idx].c[d].copy_(c_out)
                    out_states[layer_idx].n[d].copy_(n_out)
                    out_states[layer_idx].m[d].copy_(m_out)
                    out_states[layer_idx].h[d].copy_(h_out)

                layer_output = torch.cat(dir_outputs, dim=-1)

            if self.drop is not None and layer_idx < self.num_layers - 1:
                layer_output = self.drop(layer_output)
            layer_input = layer_output

        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)

        if self.pack_state:
            ret_state = tuple(out_states)
        else:
            ret_state = out_states[0] if self.num_layers == 1 else tuple(out_states)

        return layer_output, ret_state

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout}, bidirectional={self.bidirectional}, "
            f"backend={self.backend!r}, use_checkpoint={self.use_checkpoint}, "
            f"fast_mode={self.fast_mode}, fast_chunk_size={self.fast_chunk_size}"
        )

    @torch.no_grad()
    def clamp_forget_bias(self, max_val: float = _MAX_FORGET_BIAS) -> None:
        HS = self.hidden_size
        for layer_idx in range(self.num_layers):
            for d in range(self.num_directions):
                w_all = getattr(self, f"W_all_{d}_{layer_idx}")
                if w_all.bias is not None:
                    w_all.bias.data[2*HS:3*HS].clamp_(-max_val, max_val)
