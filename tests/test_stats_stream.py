"""Unit tests for the streaming statistics (StreamingStats, StreamingHistogram).

Each streaming result is compared with the same statistic computed in one
batch with NumPy, after feeding the data chunk by chunk.
"""
import numpy as np
import pytest

from numcompute_stream.stats import (
    StreamingHistogram,
    StreamingStats,
    chunk_moments,
    merge_moments,
)


def _stream(stats, X, n_chunks):
    for chunk in np.array_split(X, n_chunks):
        stats.update_stats(chunk)
    return stats


def test_streaming_mean_var_match_batch():
    X = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    stats = _stream(StreamingStats(), X, 7)
    assert stats.n_samples_seen_ == 1000
    assert np.allclose(stats.mean_, X.mean(axis=0))
    assert np.allclose(stats.var_, X.var(axis=0))
    assert np.allclose(stats.std_, X.std(axis=0))
    assert np.allclose(stats.min_, X.min(axis=0))
    assert np.allclose(stats.max_, X.max(axis=0))


def test_streaming_ddof():
    X = np.arange(20.0).reshape(10, 2)
    stats = _stream(StreamingStats(ddof=1), X, 3)
    assert np.allclose(stats.var_, X.var(axis=0, ddof=1))


def test_streaming_result_independent_of_chunking():
    X = np.random.default_rng(1).random((300, 3))
    one = _stream(StreamingStats(), X, 1)
    many = _stream(StreamingStats(), X, 300)
    assert np.allclose(one.mean_, many.mean_)
    assert np.allclose(one.var_, many.var_)


def test_streaming_1d_returns_floats():
    stats = _stream(StreamingStats(), np.arange(10.0), 3)
    assert isinstance(stats.mean_, float)
    assert stats.mean_ == 4.5
    assert stats.quantiles(0.5) == 4.5


def test_streaming_ignores_nan_per_feature():
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 6.0]])
    stats = _stream(StreamingStats(), X, 3)
    assert np.allclose(stats.count_, [2, 2])
    assert np.allclose(stats.mean_, [2.0, 5.0])
    assert np.allclose(stats.min_, [1.0, 4.0])


def test_streaming_all_nan_feature_stays_nan():
    X = np.array([[1.0, np.nan], [2.0, np.nan]])
    with np.errstate(all="raise"):
        stats = _stream(StreamingStats(), X, 2)
    assert np.isnan(stats.mean_[1]) and np.isnan(stats.var_[1])
    assert np.isnan(stats.max_[1])
    assert np.isnan(stats.quantiles(0.5)[1])


def test_streaming_zero_variance_chunks():
    stats = _stream(StreamingStats(), np.full((50, 2), 7.0), 5)
    assert np.allclose(stats.var_, 0.0)
    assert np.allclose(stats.mean_, 7.0)


def test_streaming_single_sample_variance_with_ddof_is_nan():
    stats = StreamingStats(ddof=1).update_stats([[5.0]])
    assert np.isnan(stats.var_[0])


def test_streaming_numerically_stable_with_large_offset():
    # E[x^2] - E[x]^2 loses all precision here; merged moments do not.
    X = 1e9 + np.random.default_rng(2).random(10000)
    stats = _stream(StreamingStats(), X, 10)
    assert np.isclose(stats.var_, np.var(X - 1e9), rtol=1e-6)


def test_streaming_empty_chunk_is_ignored():
    stats = StreamingStats().update_stats([[1.0, 2.0]])
    stats.update_stats(np.empty((0, 2)))
    assert stats.n_samples_seen_ == 1


def test_streaming_rejects_feature_mismatch():
    stats = StreamingStats().update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError):
        stats.update_stats(np.ones((3, 3)))


def test_streaming_rejects_non_numeric_and_3d():
    with pytest.raises(TypeError):
        StreamingStats().update_stats(["a", "b"])
    with pytest.raises(ValueError):
        StreamingStats().update_stats(np.ones((2, 2, 2)))


def test_streaming_requires_data():
    with pytest.raises(ValueError):
        StreamingStats().mean_
    with pytest.raises(ValueError):
        StreamingStats().quantiles(0.5)


def test_streaming_reset():
    stats = _stream(StreamingStats(), np.arange(10.0), 2).reset()
    assert stats.n_samples_seen_ == 0
    stats.update_stats([1.0, 3.0])
    assert stats.mean_ == 2.0


def test_streaming_quantiles_exact_below_capacity():
    X = np.random.default_rng(3).random((500, 2))
    stats = _stream(StreamingStats(max_samples=1000), X, 4)
    q = [0.1, 0.5, 0.9]
    assert np.allclose(stats.quantiles(q), np.quantile(X, q, axis=0))


def test_streaming_quantiles_reservoir_approximation():
    X = np.random.default_rng(4).random(50000)
    stats = _stream(StreamingStats(max_samples=2000, seed=0), X, 25)
    assert stats._reservoir.shape == (2000, 1)
    assert abs(stats.quantiles(0.5) - 0.5) < 0.05


def test_streaming_quantiles_reject_invalid_q():
    stats = StreamingStats().update_stats([1.0, 2.0])
    with pytest.raises(ValueError):
        stats.quantiles(1.5)


def test_merge_moments_with_empty_group():
    X = np.array([[1.0], [3.0]])
    empty = (np.zeros(1), np.full(1, np.nan), np.zeros(1))
    count, mean, m2 = merge_moments(*empty, *chunk_moments(X))
    assert count[0] == 2 and mean[0] == 2.0 and m2[0] == 2.0


def test_histogram_matches_batch():
    X = np.random.default_rng(5).normal(size=1000)
    hist = StreamingHistogram(bins=8, range=(-4, 4))
    _stream(hist, X, 6)
    counts, edges = hist.result()
    expected, expected_edges = np.histogram(X, bins=8, range=(-4, 4))
    assert np.array_equal(counts, expected)
    assert np.allclose(edges, expected_edges)


def test_histogram_infers_range_and_counts_outside():
    hist = StreamingHistogram(bins=2)
    hist.update_stats([0.0, 1.0, 2.0])
    hist.update_stats([5.0, 1.5, np.nan])
    counts, edges = hist.result()
    assert np.allclose(edges, [0.0, 1.0, 2.0])
    assert counts.tolist() == [1, 3]
    assert hist.n_outside_ == 1


def test_histogram_sliding_window():
    hist = StreamingHistogram(bins=2, range=(0, 2), window=2)
    hist.update_stats([0.5, 0.5])
    hist.update_stats([1.5])
    hist.update_stats([1.5, 1.5])
    counts, _ = hist.result()
    assert counts.tolist() == [0, 3]


def test_histogram_constant_values():
    hist = StreamingHistogram(bins=1).update_stats([3.0, 3.0])
    counts, edges = hist.result()
    assert counts.tolist() == [2]
    assert np.allclose(edges, [2.5, 3.5])


def test_histogram_invalid_parameters():
    with pytest.raises(ValueError):
        StreamingHistogram(bins=0)
    with pytest.raises(ValueError):
        StreamingHistogram(window=0)
    with pytest.raises(ValueError):
        StreamingHistogram(range=(2, 1))
    with pytest.raises(ValueError):
        StreamingHistogram().result()
