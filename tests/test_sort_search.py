import numpy as np
import pytest

from numcompute_stream.sort_search import (stable_sort, stable_argsort, multikey_sort,
                                    topk, quickselect, binary_search)


# ---------- stable sort ----------

def test_stable_sort_matches_numpy():
    a = np.array([3.0, -1.0, 2.0, 2.0, 0.5])
    assert np.array_equal(stable_sort(a), np.sort(a))


def test_stable_argsort_descending_keeps_tie_order():
    a = np.array([1, 3, 2, 3, 1])
    # Ties (3 at idx 1 and 3; 1 at idx 0 and 4) keep original order.
    assert stable_argsort(a, descending=True).tolist() == [1, 3, 2, 0, 4]


def test_stable_sort_descending_nan_last():
    a = np.array([1.0, np.nan, 3.0])
    out = stable_sort(a, descending=True)
    assert out[:2].tolist() == [3.0, 1.0] and np.isnan(out[2])


def test_stable_sort_rejects_strings():
    with pytest.raises(TypeError):
        stable_sort(np.array(["b", "a"]))


# ---------- multi-key sort ----------

def test_multikey_sort_primary_then_secondary():
    X = np.array([[2, 9], [1, 5], [2, 1], [1, 7]])
    out = multikey_sort(X, keys=[0, 1])
    assert out.tolist() == [[1, 5], [1, 7], [2, 1], [2, 9]]


def test_multikey_sort_mixed_direction():
    X = np.array([[2, 9], [1, 5], [2, 1], [1, 7]])
    out, order = multikey_sort(X, keys=[0, 1], ascending=[True, False],
                               return_indices=True)
    assert out.tolist() == [[1, 7], [1, 5], [2, 9], [2, 1]]
    assert order.tolist() == [3, 1, 0, 2]


def test_multikey_sort_invalid_key():
    with pytest.raises(ValueError):
        multikey_sort(np.zeros((3, 2)), keys=[5])


# ---------- top-k ----------

def test_topk_largest_and_smallest():
    a = np.array([5, 1, 9, 3, 7])
    vals, idx = topk(a, 3)
    assert vals.tolist() == [9, 7, 5] and idx.tolist() == [2, 4, 0]
    assert topk(a, 2, largest=False, return_indices=False).tolist() == [1, 3]


def test_topk_extreme_k():
    a = np.array([4, 2, 8])
    assert topk(a, 0, return_indices=False).shape == (0,)
    assert topk(a, 3, return_indices=False).tolist() == [8, 4, 2]
    with pytest.raises(ValueError):
        topk(a, 4)
    with pytest.raises(ValueError):
        topk(a, -1)


def test_topk_ties_ordered_by_index():
    a = np.array([5, 5, 5, 5])
    vals, idx = topk(a, 4)
    assert idx.tolist() == [0, 1, 2, 3]


def test_topk_nan_selected_last():
    a = np.array([np.nan, 2.0, 1.0])
    vals, idx = topk(a, 2)
    assert vals.tolist() == [2.0, 1.0]
    assert np.isnan(topk(a, 3, return_indices=False)[-1])


def test_topk_2d_along_axis():
    X = np.array([[1, 4, 2], [9, 0, 3]])
    assert topk(X, 2, axis=1, return_indices=False).tolist() == [[4, 2], [9, 3]]
    assert topk(X, 1, axis=0, return_indices=False).tolist() == [[9, 4, 3]]


def test_topk_non_integer_k():
    with pytest.raises(TypeError):
        topk([1, 2, 3], 1.5)


# ---------- quickselect ----------

def test_quickselect_matches_sort_random():
    rng = np.random.default_rng(0)
    a = rng.integers(-50, 50, size=200)   # many duplicates
    s = np.sort(a)
    for k in [0, 1, 57, 100, 199, -1]:
        assert quickselect(a, k, seed=1) == s[k]


def test_quickselect_all_equal_and_single():
    assert quickselect(np.full(10, 7.0), 4) == 7.0
    assert quickselect([3], 0) == 3


def test_quickselect_errors():
    with pytest.raises(ValueError):
        quickselect([], 0)
    with pytest.raises(ValueError):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(ValueError):
        quickselect([1, 2, 3], 3)


def test_quickselect_non_contiguous():
    a = np.arange(20)[::-2]               # strided view: 19, 17, ..., 1
    assert not a.flags["C_CONTIGUOUS"]
    assert quickselect(a, 0) == 1


# ---------- binary search ----------

def test_binary_search_found_and_missing():
    arr = np.array([1, 3, 5, 7])
    assert binary_search(arr, 5) == (2, True)
    assert binary_search(arr, 4) == (2, False)
    assert binary_search(arr, 0) == (0, False)
    assert binary_search(arr, 9) == (4, False)


def test_binary_search_vectorised_with_duplicates():
    arr = np.array([1, 2, 2, 2, 5])
    idx, found = binary_search(arr, [2, 3, 5], side="right")
    assert idx.tolist() == [4, 4, 5]
    assert found.tolist() == [True, False, True]


def test_binary_search_empty_and_nan():
    assert binary_search(np.array([]), 1.0) == (0, False)
    assert binary_search(np.array([1.0, 2.0]), np.nan)[1] is False


def test_binary_search_unsorted_check():
    with pytest.raises(ValueError):
        binary_search([3, 1, 2], 1, check_sorted=True)


# ---------- added: precision, dtypes and input validation ----------

def test_sort_large_integers_exact():
    # 2**62 and 2**62 + 1 are equal as float64; integer keys must stay exact.
    a = np.array([2**62 + 1, 2**62])
    assert stable_argsort(a).tolist() == [1, 0]
    assert stable_sort(a, descending=True).tolist() == [2**62 + 1, 2**62]
    assert multikey_sort(a[:, None], keys=[0]).ravel().tolist() == [2**62, 2**62 + 1]
    assert topk(a, 1, largest=False, return_indices=False).tolist() == [2**62]


def test_descending_integer_extremes_and_unsigned():
    lo = np.iinfo(np.int64).min
    assert stable_sort(np.array([lo, 0, 5]), descending=True).tolist() == [5, 0, lo]
    u = np.array([0, 255, 7], dtype=np.uint8)
    assert stable_sort(u, descending=True).tolist() == [255, 7, 0]


def test_stable_sort_2d_axis0_descending():
    X = np.array([[3, 1], [1, 2]])
    assert stable_sort(X, axis=0, descending=True).tolist() == [[3, 2], [1, 1]]


def test_multikey_sort_nan_last_and_errors():
    X = np.array([[1, np.nan], [1, 2.0], [0, 5.0]])
    out = multikey_sort(X, keys=[0, 1], ascending=[True, False])
    assert out[:2].tolist() == [[0, 5], [1, 2]] and np.isnan(out[2, 1])
    with pytest.raises(ValueError):
        multikey_sort([1, 2, 3], keys=[0])                       # not 2-D
    with pytest.raises(ValueError):
        multikey_sort(np.zeros((2, 2)), keys=[0, 1], ascending=[True])


def test_topk_non_contiguous_and_empty():
    a = np.arange(10)[::-3]                                     # 9, 6, 3, 0
    assert not a.flags["C_CONTIGUOUS"]
    assert topk(a, 2, return_indices=False).tolist() == [9, 6]
    vals, idx = topk(np.array([]), 0)
    assert vals.shape == (0,) and idx.shape == (0,)


def test_quickselect_type_and_shape_errors():
    with pytest.raises(TypeError):
        quickselect([1, 2, 3], 0.5)
    with pytest.raises(ValueError):
        quickselect(np.zeros((2, 2)), 0)


def test_binary_search_2d_queries_and_bad_side():
    idx, found = binary_search([1, 2, 3], [[1, 4], [2, 3]])
    assert idx.tolist() == [[0, 3], [1, 2]]
    assert found.tolist() == [[True, False], [True, True]]
    with pytest.raises(ValueError):
        binary_search([1, 2], 1, side="middle")


def test_topk_boundary_ties_pick_smallest_index():
    # Three 0s compete for the last two places: indices 1 and 4 must win.
    a = np.array([255, 0, 255, 254, 0, 254, 0, 255, 1, 254], dtype=np.uint8)
    _, idx = topk(a, 9)
    assert sorted(idx.tolist()) == [0, 1, 2, 3, 4, 5, 7, 8, 9]


def test_topk_nan_after_infinity():
    a = np.array([np.nan, -np.inf, 1.0])
    _, idx = topk(a, 2)                        # largest: 1.0, then -inf (not NaN)
    assert idx.tolist() == [2, 1]
