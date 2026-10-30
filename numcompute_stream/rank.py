"""Ranking with tie handling, and percentiles with interpolation.

Both functions treat NaN as missing: NaNs receive no rank and are
excluded from percentile computation.
"""

import numpy as np

__all__ = ["rank", "percentile"]

_RANK_METHODS = ("average", "dense", "ordinal")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


def _check_numeric(arr, name="data"):
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")


def _rank_1d(x, method):
    """Rank a 1-D array with no NaNs. Fully vectorised."""
    n = x.size
    sorter = np.argsort(x, kind="stable")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)             # position of each element after sorting

    if method == "ordinal":
        return inv + 1

    xs = x[sorter]
    is_new = np.r_[True, xs[1:] != xs[:-1]]  # True at the start of each tie group
    dense = np.cumsum(is_new)[inv]
    if method == "dense":
        return dense

    # Group g occupies sorted positions [starts[g-1], starts[g]); the
    # average 1-based rank of that block is (start + end + 1) / 2.
    starts = np.r_[np.flatnonzero(is_new), n]
    return 0.5 * (starts[dense] + starts[dense - 1] + 1)


def rank(data, method="average", axis=None):
    """Assign 1-based ranks to data, handling ties.

    Parameters
    ----------
    data : array_like
        Numeric input. NaN values are left unranked (rank NaN).
    method : {'average', 'dense', 'ordinal'}, default 'average'
        Tie handling, e.g. for ``[10, 20, 20, 30]``:

        - 'average': tied values share their mean rank  -> [1, 2.5, 2.5, 4]
        - 'dense'  : tied values share a rank, no gaps   -> [1, 2, 2, 3]
        - 'ordinal': unique ranks, ties by original order -> [1, 2, 3, 4]
    axis : int or None, default None
        Axis to rank along. None ranks the flattened array.

    Returns
    -------
    ranks : ndarray
        Shape ``(data.size,)`` if ``axis`` is None, otherwise
        ``data.shape``. dtype is float for 'average' or when NaNs are
        present, int otherwise.

    Raises
    ------
    TypeError
        If ``data`` is not numeric.
    ValueError
        If ``method`` is not one of the supported options.

    Complexity
    ----------
    Time O(n log n) per slice, space O(n).
    Ranking along an axis uses ``np.apply_along_axis`` (a loop over slices).
    """
    arr = np.asarray(data)
    _check_numeric(arr)
    if method not in _RANK_METHODS:
        raise ValueError(f"method must be one of {_RANK_METHODS}, got {method!r}")

    def _rank_with_nan(x):
        nan_mask = np.isnan(x) if np.issubdtype(x.dtype, np.floating) else None
        if nan_mask is None or not nan_mask.any():
            return _rank_1d(x, method)
        out = np.full(x.size, np.nan)
        out[~nan_mask] = _rank_1d(x[~nan_mask], method)
        return out

    if axis is None:
        return _rank_with_nan(arr.ravel())
    if arr.ndim == 0:
        raise ValueError("axis must be None for 0-d input")
    # Rank every slice as float so slices with/without NaN share a dtype.
    ranks = np.apply_along_axis(lambda s: _rank_with_nan(s).astype(float), axis, arr)
    if method != "average" and not np.isnan(ranks).any():
        ranks = ranks.astype(np.intp)
    return ranks


def percentile(data, q, interpolation="linear", axis=None):
    """Compute the q-th percentile(s) of data, ignoring NaNs.

    For n valid values sorted ascending, the target position is
    ``pos = q / 100 * (n - 1)``, between ``i = floor(pos)`` and
    ``j = ceil(pos)``:

    - 'linear'  : (1 - t) * v[i] + t * v[j], with t = pos - i
    - 'lower'   : v[i]
    - 'higher'  : v[j]
    - 'midpoint': (v[i] + v[j]) / 2

    The weighted form never overflows for finite data and handles +-inf
    (e.g. any interpolation towards +inf is +inf); only an interpolation
    between -inf and +inf is undefined and returns NaN.

    Parameters
    ----------
    data : array_like
        Numeric input. NaNs are ignored.
    q : float or array_like of float, shape (m,)
        Percentile(s) in ``[0, 100]``.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default 'linear'
        Rule used when ``pos`` falls between two data points.
    axis : int or None, default None
        Axis to compute along. None uses the flattened array.

    Returns
    -------
    result : float or ndarray
        Scalar ``q`` and ``axis=None`` give a float. Otherwise the shape is
        ``q.shape + data.shape`` with ``axis`` removed (NumPy convention).
        A slice with no valid values gives NaN.

    Raises
    ------
    TypeError
        If ``data`` is not numeric.
    ValueError
        If ``data`` is empty, ``q`` is outside ``[0, 100]`` or NaN, or
        ``interpolation`` is not supported.

    Complexity
    ----------
    Time O(n log n) per slice (sort), space O(n).
    """
    arr = np.asarray(data)
    _check_numeric(arr)
    if arr.size == 0:
        raise ValueError("percentile of an empty array is undefined")
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            f"interpolation must be one of {_INTERPOLATIONS}, got {interpolation!r}")
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim > 1:
        raise ValueError("q must be a scalar or a 1-D array")
    if np.isnan(q_arr).any() or np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError(f"q must be in [0, 100], got {q}")

    arr = arr.astype(float).ravel() if axis is None else np.moveaxis(arr.astype(float), axis, -1)
    s = np.sort(arr, axis=-1)                          # NaNs sorted to the end
    n_valid = np.count_nonzero(~np.isnan(s), axis=-1)  # shape: batch
    qs = np.atleast_1d(q_arr)                          # shape: (m,)

    # pos has shape batch + (m,) so it can index the last axis of s.
    pos = (qs / 100.0) * np.maximum(n_valid - 1, 0)[..., None]
    lo = np.floor(pos).astype(np.intp)
    hi = np.ceil(pos).astype(np.intp)
    v_lo = np.take_along_axis(s, lo, axis=-1)
    v_hi = np.take_along_axis(s, hi, axis=-1)

    if interpolation == "lower":
        res = v_lo
    elif interpolation == "higher":
        res = v_hi
    else:
        t = 0.5 if interpolation == "midpoint" else pos - lo
        # (1 - t) * a + t * b avoids the overflow of b - a for huge finite
        # values and gives the right limit for +-inf. Only a = -inf, b = +inf
        # is undefined (NaN). Clipping to [a, b] removes rounding past the ends.
        with np.errstate(invalid="ignore", over="ignore"):
            mixed = np.clip((1 - t) * v_lo + t * v_hi, v_lo, v_hi)
        res = np.where((lo == hi) | (v_lo == v_hi), v_lo, mixed)

    res = np.where((n_valid == 0)[..., None], np.nan, res)
    res = np.moveaxis(res, -1, 0)                      # q axis first
    if q_arr.ndim == 0:
        res = res[0]
    return float(res) if np.ndim(res) == 0 else res
