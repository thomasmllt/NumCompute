"""Randomised comparison tests for sort_search and rank.

Each test generates many random inputs (mixed dtypes, NaN, +-inf, ties,
integer extremes) and compares the result with a simple brute-force
reference written in plain Python. Seeds are fixed, so runs are
reproducible. Increase N_CASES for a longer, more thorough run.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from numcompute_stream.sort_search import (stable_argsort, stable_sort, multikey_sort,
                                    topk, quickselect, binary_search)
from numcompute_stream.rank import rank, percentile

N_CASES = 300
pytestmark = pytest.mark.filterwarnings("error")   # any RuntimeWarning fails a test

DTYPES = ["int8", "uint8", "int64", "float16", "float32", "float64", "bool", "special"]


# ---------- helpers ----------

def random_array(rng, n):
    """Random 1-D array of a random dtype, with ties, NaN, inf and extremes."""
    kind = rng.choice(DTYPES)
    if kind == "bool":
        return rng.integers(0, 2, n).astype(bool)
    if kind in ("int8", "uint8", "int64"):
        info = np.iinfo(kind)
        if rng.random() < 0.3:
            pool = np.array([info.min, info.max, info.max - 1, 0, 1], dtype=kind)
            return rng.choice(pool, n)
        return rng.integers(max(info.min, -5), min(info.max, 5), n, endpoint=True).astype(kind)
    if kind == "special":
        pool = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e308, -1e308, 5e-324])
        return rng.choice(pool, n)
    a = rng.integers(-3, 4, n).astype(kind)
    if n and rng.random() < 0.5:
        a[rng.random(n) < 0.3] = np.nan
    return a


def py(v):
    """Exact Python value (integers stay integers)."""
    return v.item() if hasattr(v, "item") else v


def is_nan(v):
    return v != v


def sort_key(v, descending=False):
    """Key for Python's sorted(): NaN last in both directions."""
    v = py(v)
    if is_nan(v):
        return (1, 0)
    return (0, -v if descending else v)


def ref_order(a, descending=False):
    """Stable reference order: by value, then by original index."""
    return sorted(range(len(a)), key=lambda i: (sort_key(a[i], descending), i))


def ref_percentile(a, q, interpolation):
    """Exact reference using fractions, with the mathematical limits for inf."""
    v = sorted(float(x) for x in a.astype(float) if not math.isnan(float(x)))
    if not v:
        return math.nan
    pos = (q / 100.0) * (len(v) - 1)
    i, j = math.floor(pos), math.ceil(pos)
    lo, hi = v[i], v[j]
    if interpolation == "lower":
        return lo
    if interpolation == "higher":
        return hi
    t = 0.5 if interpolation == "midpoint" else pos - i
    if i == j or lo == hi:
        return lo
    if math.isinf(lo) or math.isinf(hi):
        if lo == -math.inf and hi == math.inf:
            return math.nan
        return lo if math.isinf(lo) else hi
    return float(Fraction(lo) * (1 - Fraction(t)) + Fraction(hi) * Fraction(t))


def ref_rank(a, method):
    """Brute-force ranks; NaN stays unranked."""
    valid = [i for i in range(len(a)) if not is_nan(py(a[i]))]
    values = sorted(py(a[i]) for i in valid)
    unique = sorted(set(values))
    order = sorted(valid, key=lambda i: (py(a[i]), i))
    out = [math.nan] * len(a)
    for pos, i in enumerate(order):
        v = py(a[i])
        if method == "ordinal":
            out[i] = pos + 1
        elif method == "dense":
            out[i] = unique.index(v) + 1
        else:
            first, last = values.index(v), len(values) - 1 - values[::-1].index(v)
            out[i] = (first + last + 2) / 2
    return np.array(out, dtype=float)


# ---------- tests ----------

def test_random_stable_sort():
    rng = np.random.default_rng(1)
    for _ in range(N_CASES):
        a = random_array(rng, int(rng.integers(0, 12)))
        for desc in (False, True):
            ref = ref_order(a, desc)
            assert stable_argsort(a, descending=desc).tolist() == ref
            out = stable_sort(a, descending=desc)
            assert out.dtype == a.dtype
            assert np.array_equal(out, a[ref], equal_nan=a.dtype.kind == "f")


def test_random_topk():
    rng = np.random.default_rng(2)
    for _ in range(N_CASES):
        a = random_array(rng, int(rng.integers(0, 12)))
        for largest in (True, False):
            k = int(rng.integers(0, a.size + 1))
            vals, idx = topk(a, k, largest=largest)
            assert idx.tolist() == ref_order(a, largest)[:k]
            assert np.array_equal(vals, a[idx], equal_nan=a.dtype.kind == "f")


def test_random_quickselect():
    rng = np.random.default_rng(3)
    for case in range(N_CASES):
        a = random_array(rng, int(rng.integers(1, 15)))
        if a.dtype.kind == "f":
            a = a[~np.isnan(a)]
            if a.size == 0:
                continue
        k = int(rng.integers(-a.size, a.size))
        assert quickselect(a, k, seed=case) == np.sort(a)[k]


def test_random_binary_search():
    rng = np.random.default_rng(4)
    for _ in range(N_CASES):
        s = np.sort(random_array(rng, int(rng.integers(0, 12))))
        queries = rng.choice(s, 3) if s.size else s[:0]
        if s.dtype.kind == "f":
            queries = np.r_[queries, np.array([0.5, np.nan], dtype=s.dtype)]
        for side in ("left", "right"):
            idx, found = binary_search(s, queries, side=side)
            assert idx.tolist() == np.searchsorted(s, queries, side=side).tolist()
            ref_found = [any(py(x) == py(y) for y in s) for x in queries]
            assert found.tolist() == ref_found


def test_random_multikey_sort():
    rng = np.random.default_rng(5)
    for _ in range(N_CASES):
        rows, cols = int(rng.integers(0, 8)), int(rng.integers(1, 4))
        X = random_array(rng, rows * cols).reshape(rows, cols)
        keys = [int(k) for k in rng.choice(cols, int(rng.integers(1, cols + 1)), replace=False)]
        asc = [bool(b) for b in rng.integers(0, 2, len(keys))]

        def row_key(i):
            return tuple(sort_key(X[i, k], not up) for k, up in zip(keys, asc)) + (i,)

        _, order = multikey_sort(X, keys, ascending=asc, return_indices=True)
        assert order.tolist() == sorted(range(rows), key=row_key)


@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
def test_random_rank(method):
    rng = np.random.default_rng(6)
    for _ in range(N_CASES):
        a = random_array(rng, int(rng.integers(0, 12)))
        got = rank(a, method=method)
        assert got.shape == a.shape
        assert np.allclose(got.astype(float), ref_rank(a, method), equal_nan=True)


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint"])
def test_random_percentile(interpolation):
    rng = np.random.default_rng(7)
    for _ in range(N_CASES):
        a = random_array(rng, int(rng.integers(1, 12)))
        if a.dtype == bool:
            continue
        qs = rng.uniform(0, 100, 3).round(1)
        qs[0] = rng.choice([0.0, 100.0, qs[0]])
        got = percentile(a, qs, interpolation=interpolation)
        ref = [ref_percentile(a, q, interpolation) for q in qs]
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-300, equal_nan=True)


def test_random_axis_consistency():
    """Results along an axis must equal the results of each 1-D slice."""
    rng = np.random.default_rng(8)
    for _ in range(N_CASES):
        shape = tuple(int(s) for s in rng.integers(1, 5, int(rng.integers(2, 4))))
        X = rng.integers(-3, 4, shape).astype(float)
        X[rng.random(shape) < 0.2] = np.nan
        axis = int(rng.integers(-X.ndim, X.ndim))
        slices = np.moveaxis(X, axis, -1).reshape(-1, X.shape[axis])

        r = np.moveaxis(rank(X, method="average", axis=axis), axis, -1).reshape(slices.shape)
        # percentile puts q first; the other axes keep their order, like `slices`.
        p = np.moveaxis(percentile(X, [10, 50, 90], axis=axis), 0, -1).reshape(-1, 3)
        k = int(rng.integers(0, X.shape[axis] + 1))
        _, i = topk(X, k, axis=axis)
        i = np.moveaxis(i, axis, -1).reshape(slices.shape[0], k)

        for row in range(slices.shape[0]):
            s = slices[row]
            assert np.allclose(r[row], rank(s), equal_nan=True)
            assert np.allclose(p[row], percentile(s, [10, 50, 90]), equal_nan=True)
            assert i[row].tolist() == topk(s, k)[1].tolist()
