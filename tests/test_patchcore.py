"""Offline tests for the PatchCore core: coreset, memory bank, score-map normalization.

No backbone download happens here — scoring is exercised through MemoryBank
directly with synthetic embeddings.
"""
import numpy as np
import torch

from src.heatmap import _norm_score_map
from src.patchcore import INPUT_SIZE, MemoryBank, coreset_subsample


def test_coreset_selects_expected_count_and_preserves_shape():
    g = torch.Generator().manual_seed(0)
    feats = torch.rand(50, 8, generator=g)
    out = coreset_subsample(feats, ratio=0.1, seed=123)
    assert out.shape == (5, 8)
    # Every selected row must be a member of the input set.
    matches = (feats.unsqueeze(0) == out.unsqueeze(1)).all(dim=2).any(dim=1)
    assert matches.all()


def test_coreset_keeps_at_least_one_row_for_tiny_ratios():
    feats = torch.rand(4, 3)
    out = coreset_subsample(feats, ratio=0.05)
    assert out.shape == (1, 3)


def test_coreset_is_deterministic_for_fixed_seed():
    g = torch.Generator().manual_seed(42)
    feats = torch.rand(30, 16, generator=g)
    a = coreset_subsample(feats, ratio=0.2, seed=7)
    b = coreset_subsample(feats, ratio=0.2, seed=7)
    assert torch.equal(a, b)


def test_memory_bank_score_shapes_and_finiteness():
    g = torch.Generator().manual_seed(1)
    bank = MemoryBank(torch.rand(64, 12, generator=g), torch.device('cpu'))
    queries = torch.rand(784, 12, generator=g)  # 784 patches per standard frame
    scores = bank.score(queries)
    assert scores.shape == (784,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_memory_bank_matches_exact_nearest_neighbour_distance():
    bank_feats = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    bank = MemoryBank(bank_feats, torch.device('cpu'))
    queries = torch.tensor([[1.0, 0.0], [9.0, 9.0], [10.0, 10.0]])
    scores = bank.score(queries)
    expected = torch.tensor([1.0, torch.sqrt(torch.tensor(2.0)), 0.0])
    assert torch.allclose(scores, expected, atol=1e-6)


def test_memory_bank_chunking_covers_more_queries_than_chunk_size():
    g = torch.Generator().manual_seed(3)
    bank_feats = torch.rand(32, 8, generator=g)
    bank = MemoryBank(bank_feats, torch.device('cpu'))
    queries = torch.rand(1200, 8, generator=g)  # score() chunks at 512
    scores = bank.score(queries)
    brute_force = torch.cdist(queries, bank_feats).min(dim=1).values
    assert scores.shape == (1200,)
    assert torch.allclose(scores, brute_force, atol=1e-5)


def test_memory_bank_save_load_round_trip_with_meta(tmp_path):
    g = torch.Generator().manual_seed(5)
    features = torch.rand(16, 4, generator=g)
    bank = MemoryBank(features, torch.device('cpu'))
    path = tmp_path / 'bank.pt'
    bank.save(str(path), meta={'threshold': 1.23, 'good_max': 0.6})

    loaded, meta = MemoryBank.load(str(path), torch.device('cpu'))
    assert loaded.features.shape == (16, 4)
    assert torch.allclose(loaded.features, features)
    assert meta['threshold'] == 1.23
    assert meta['good_max'] == 0.6


def test_norm_score_map_uses_frame_max_when_vmax_is_none():
    score_map = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    out = _norm_score_map(score_map, None)
    assert out.dtype == np.uint8
    assert out.shape == score_map.shape
    assert out.max() == 255
    assert out[0, 0] == 0


def test_norm_score_map_pins_vmax_anchor_and_clips_above():
    score_map = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    out = _norm_score_map(score_map, vmax=2.0)
    assert out[0, 0] == 0
    assert out[0, 1] == int(1.0 / 2.0 * 255.0)      # truncated to uint8
    assert out[1, 0] == 255                          # exactly at the anchor
    assert out[1, 1] == 255                          # above vmax clips


def test_norm_score_map_zero_map_returns_zeros():
    score_map = np.zeros((7, 7), dtype=np.float32)
    out = _norm_score_map(score_map, None)
    assert (out == 0).all()


def test_norm_score_map_degenerate_vmax_falls_back_to_frame_max():
    score_map = np.array([[0.25, 0.5]], dtype=np.float32)
    fallback = _norm_score_map(score_map, None)
    degenerate = _norm_score_map(score_map, vmax=0.0)  # <= 1e-6 -> ignored
    assert np.array_equal(fallback, degenerate)


def test_input_size_constant_is_standard_patchcore_grid():
    assert INPUT_SIZE == 224
