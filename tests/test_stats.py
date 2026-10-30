import numpy as np

from numcompute_stream.stats import (
    mean,
    median,
    std,
    minimum,
    maximum,
    histogram,
    quantiles,
)


def test_mean():
    assert mean([1, 2, 3]) == 2


def test_median():
    assert median([1, 2, 3, 4]) == 2.5


def test_std():
    assert np.isclose(std([1, 2, 3]), np.std([1, 2, 3]))


def test_minimum_and_maximum():
    values = [4, 1, 7, 2]
    assert minimum(values) == 1
    assert maximum(values) == 7


def test_statistics_ignore_nan():
    values = [1.0, np.nan, 3.0]
    assert mean(values) == 2.0
    assert median(values) == 2.0
    assert minimum(values) == 1.0
    assert maximum(values) == 3.0


def test_statistics_with_axis():
    values = np.array([[1, 2], [3, 4]])
    assert np.array_equal(mean(values, axis=0), [2, 3])
    assert np.array_equal(minimum(values, axis=1), [1, 3])


def test_histogram():
    counts, edges = histogram([1, 2, 3, 4], bins=2)
    assert counts.tolist() == [2, 2]
    assert len(edges) == 3


def test_quantiles():
    result = quantiles([1, 2, 3, 4], [0.25, 0.75])
    assert np.allclose(result, [1.75, 3.25])


def test_quantiles_reject_invalid_values():
    with np.testing.assert_raises(ValueError):
        quantiles([1, 2, 3], 1.5)


def test_statistics_all_equal():
    values = [5, 5, 5]
    assert mean(values) == 5
    assert std(values) == 0


def test_statistics_non_contiguous():
    values = np.arange(10.0)[::2]
    assert mean(values) == 4
    assert quantiles(values, 0.5) == 4
