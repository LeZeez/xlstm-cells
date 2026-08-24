"""Tests for xlstm_cells configuration dataclasses."""

from __future__ import annotations

import pytest
import torch

from xlstm_cells.configs import (
    mLSTMCellConfig,
    sLSTMCellConfig,
    mLSTMConfig,
    sLSTMConfig,
    mLSTMBlockConfig,
    sLSTMBlockConfig,
    xLSTMLargeBlockConfig,
)
from xlstm_cells import (
    mLSTMCell,
    sLSTMCell,
    mLSTM,
    sLSTM,
    mLSTMBlock,
    sLSTMBlock,
    xLSTMLargeBlock,
)


def test_mlstm_cell_config():
    cfg = mLSTMCellConfig(input_size=64, hidden_size=128, num_heads=4)
    cell = mLSTMCell(config=cfg)
    assert cell.input_size == 64
    assert cell.hidden_size == 128
    assert cell.num_heads == 4
    x = torch.randn(2, 64)
    st = cell.init_state(2)
    out, new_st = cell(x, st)
    assert out.shape == (2, 128)

    with pytest.raises(ValueError):
        mLSTMCellConfig(input_size=64, hidden_size=128, num_heads=5)  # not divisible


def test_slstm_cell_config():
    cfg = sLSTMCellConfig(input_size=64, hidden_size=128, num_heads=4)
    cell = sLSTMCell(config=cfg)
    assert cell.input_size == 64
    assert cell.hidden_size == 128
    assert cell.num_heads == 4
    x = torch.randn(2, 64)
    st = cell.init_state(2)
    out, new_st = cell(x, st)
    assert out.shape == (2, 128)


def test_mlstm_and_slstm_sequence_configs():
    m_cfg = mLSTMConfig(input_size=32, hidden_size=64, num_layers=2, num_heads=4)
    layer_m = mLSTM(config=m_cfg)
    x = torch.randn(2, 16, 32)
    out_m, states_m = layer_m(x)
    assert out_m.shape == (2, 16, 64)
    assert len(states_m) == 2

    s_cfg = sLSTMConfig(input_size=32, hidden_size=64, num_layers=2, num_heads=4, backend="vanilla")
    layer_s = sLSTM(config=s_cfg)
    out_s, states_s = layer_s(x)
    assert out_s.shape == (2, 16, 64)
    assert len(states_s) == 2

    # Validation checks
    with pytest.raises(ValueError):
        mLSTMConfig(input_size=-1, hidden_size=64)
    with pytest.raises(ValueError):
        sLSTMConfig(input_size=32, hidden_size=64, backend="invalid_backend")


def test_block_configs():
    mb_cfg = mLSTMBlockConfig(d_model=64, num_heads=4, norm_type="rmsnorm")
    blk_m = mLSTMBlock(config=mb_cfg)
    x = torch.randn(2, 8, 64)
    out_m, st_m = blk_m(x)
    assert out_m.shape == (2, 8, 64)

    sb_cfg = sLSTMBlockConfig(d_model=64, num_heads=4, norm_type="rmsnorm")
    blk_s = sLSTMBlock(config=sb_cfg)
    out_s, st_s = blk_s(x)
    assert out_s.shape == (2, 8, 64)

    with pytest.raises(ValueError, match="unknown norm_type"):
        mLSTMBlockConfig(d_model=64, norm_type="invalid_norm")


def test_xlstm_large_block_config():
    cfg = xLSTMLargeBlockConfig(
        embedding_dim=512,
        num_heads=8,
        num_blocks=6,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0,
    )
    assert cfg.embedding_dim == 512
    assert cfg.num_blocks == 6
    assert cfg.qk_dim_factor == 0.5
    assert cfg.gate_soft_cap == 15.0

    # Aliases
    cfg_alias = xLSTMLargeBlockConfig(hidden_size=256, num_hidden_layers=4, num_heads=4)
    assert cfg_alias.embedding_dim == 256
    assert cfg_alias.num_blocks == 4

    block = xLSTMLargeBlock(cfg_alias)
    x = torch.randn(2, 16, 256)
    out, st = block(x)
    assert out.shape == (2, 16, 256)

    with pytest.raises(ValueError, match="conflicting values"):
        xLSTMLargeBlockConfig(embedding_dim=128, hidden_size=256)
