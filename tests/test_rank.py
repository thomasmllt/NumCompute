import numpy as np
import pytest

from numcompute_stream.rank import rank, percentile


# ---------- rank ----------

@pytest.mark.parametrize("method, expected", [
    ("average", [1, 2.5, 2.5, 4]),
    ("dense", [1, 2, 2, 3]),
    ("ordinal", [1, 2, 3, 4]),
])
def test_rank_methods_with_ties(method, expected):
    assert rank([10, 20, 20, 30], method=method).tolist() == expected


def test_rank_unsorted_input():
    assert rank([30, 10, 20, 10]).tolist() == [4, 1.5, 3, 1.5]


def test_rank_all_equal():
    assert rank([7, 7, 7], method="average").tolist() == [2, 2, 2]
    assert rank([7, 7, 7], method="dense").tolist() == [1, 1, 1]


def test_rank_empty():
    assert rank([]).shape == (0,)


def test_rank_nan_left_unranked():
    r = rank([3.0, np.nan, 1.0], method="dense")
    assert r[0] == 2 and np.isnan(r[1]) and r[2] == 1


def test_rank_along_axis():
    X = np.array([[3, 1, 2], [1, 1, 5]])
    assert rank(X, method="ordinal", axis=1).tolist() == [[3, 1, 2], [1, 2, 3]]
    assert rank(X, method="dense", axis=0).tolist() == [[2, 1, 1], [1, 1, 2]]


def test_rank_invalid_method():
    with pytest.raises(ValueError):
        rank([1, 2], method="max")


# ---------- percentile ----------

@pytest.mark.parametrize("interp", ["linear", "lower", "higher", "midpoint"])
def test_percentile_matches_numpy(interp):
    rng = np.random.default_rng(42)
    a = rng.normal(size=101)
    q = [0, 10, 33.3, 50, 99, 100]
    assert np.allclose(percentile(a, q, interpolation=interp),
                       np.percentile(a, q, method=interp))


def test_percentile_scalar_and_known_values():
    a = [1, 2, 3, 4]
    assert percentile(a, 50) == 2.5
    assert percentile(a, 50, interpolation="lower") == 2
    assert percentile(a, 50, interpolation="higher") == 3


def test_percentile_ignores_nan():
    a = [1.0, np.nan, 3.0]
    assert percentile(a, 50) == 2.0


def test_percentile_all_nan_slice():
    X = np.array([[np.nan, np.nan], [1.0, 3.0]])
    out = percentile(X, 50, axis=1)
    assert np.isnan(out[0]) and out[1] == 2.0


def test_percentile_axis_shape():
    X = np.arange(12.0).reshape(3, 4)
    out = percentile(X, [25, 75], axis=0)
    assert out.shape == (2, 4)
    assert np.allclose(out, np.percentile(X, [25, 75], axis=0))


def test_percentile_single_value_and_inf():
    assert percentile([5.0], 30) == 5.0
    assert percentile([1.0, np.inf], 100) == np.inf


def test_percentile_non_contiguous():
    a = np.arange(10.0)[::3]              # 0, 3, 6, 9
    assert percentile(a, 50) == 4.5


def test_percentile_errors():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
    with pytest.raises(ValueError):
        percentile([1, 2], 50, interpolation="nearest")


# ---------- added: input validation and extreme values ----------

def test_rank_rejects_non_numeric():
    with pytest.raises(TypeError):
        rank(["a", "b"])


def test_rank_integer_dtype_and_nan_along_axis():
    assert rank([3, 1, 2], method="dense").dtype.kind == "i"
    X = np.array([[1.0, np.nan], [2.0, 1.0]])
    r = rank(X, method="dense", axis=1)
    assert r[0, 0] == 1 and np.isnan(r[0, 1]) and r[1].tolist() == [2, 1]


def test_percentile_invalid_q():
    with pytest.raises(ValueError):
        percentile([1, 2], np.nan)
    with pytest.raises(ValueError):
        percentile([1, 2], [[50]])


def test_percentile_negative_axis_and_opposite_infinities():
    X = np.arange(6.0).reshape(2, 3)
    assert percentile(X, 50, axis=-1).tolist() == [1.0, 4.0]
    for interp in ["linear", "midpoint"]:
        assert np.isnan(percentile([-np.inf, np.inf], 50, interpolation=interp))


def test_percentile_no_overflow_with_huge_values():
    a = [-1e308, 1e308]
    assert percentile(a, 50) == 0.0
    assert percentile(a, 50, interpolation="midpoint") == 0.0
    assert percentile([1e308, 1.5e308], 50) == 1.25e308


def test_percentile_interpolation_towards_infinity():
    # NumPy returns NaN here; the mathematical limit is +inf.
    assert percentile([0.0, np.inf, np.inf], 75) == np.inf
    assert percentile([-np.inf, 0.0], 50) == -np.inf
