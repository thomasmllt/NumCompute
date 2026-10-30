"""Descriptive statistics, in batch and streaming (chunk-by-chunk) form."""

import warnings
from collections import deque

import numpy as np

__all__ = [
    "StreamingStats",
    "StreamingHistogram",
    "chunk_moments",
    "merge_moments",
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "histogram",
    "quantiles",
]


def _as_numeric(data):
    """Convert data to a numeric NumPy array."""
    arr = np.asarray(data)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {arr.dtype}")
    return arr


def mean(data, axis=None):
    """Calculate the mean, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to calculate the mean. ``None`` uses all values.

    Returns
    -------
    float or ndarray
        Mean value, or means along the selected axis.
    """
    return np.nanmean(_as_numeric(data), axis=axis)


def median(data, axis=None):
    """Calculate the median, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to calculate the median.

    Returns
    -------
    float or ndarray
        Median value, or medians along the selected axis.
    """
    return np.nanmedian(_as_numeric(data), axis=axis)


def std(data, axis=None, ddof=0):
    """Calculate the standard deviation, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to calculate the standard deviation.
    ddof : int, default=0
        Delta degrees of freedom used in the calculation.

    Returns
    -------
    float or ndarray
        Standard deviation value or axis-wise standard deviations.
    """
    return np.nanstd(_as_numeric(data), axis=axis, ddof=ddof)


def minimum(data, axis=None):
    """Find the minimum, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to find the minimum.

    Returns
    -------
    scalar or ndarray
        Minimum value, or minima along the selected axis.
    """
    return np.nanmin(_as_numeric(data), axis=axis)


def maximum(data, axis=None):
    """Find the maximum, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to find the maximum.

    Returns
    -------
    scalar or ndarray
        Maximum value, or maxima along the selected axis.
    """
    return np.nanmax(_as_numeric(data), axis=axis)


def histogram(data, bins=10, range=None):
    """Create a histogram after removing NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    bins : int or array_like, default=10
        Number of bins or the bin edges.
    range : tuple or None, default=None
        Lower and upper range of the bins.

    Returns
    -------
    counts : ndarray
        Number of values in each bin.
    edges : ndarray
        Bin boundaries, with length ``len(counts) + 1``.
    """
    arr = _as_numeric(data)
    valid = arr[~np.isnan(arr)]
    return np.histogram(valid, bins=bins, range=range)


def quantiles(data, q, axis=None):
    """Calculate quantiles between 0 and 1, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    q : float or array_like
        Quantile values in the interval ``[0, 1]``.
    axis : int or None, default=None
        Axis along which to calculate quantiles.

    Returns
    -------
    float or ndarray
        Quantile value(s). For multiple ``q`` values, the q dimension is
        returned first.

    Raises
    ------
    ValueError
        If a quantile is outside ``[0, 1]`` or is NaN.
    """
    return np.nanquantile(_as_numeric(data), _check_q(q), axis=axis)


def _check_q(q):
    """Validate quantile values and return them as a float array."""
    q_arr = np.asarray(q, dtype=float)
    if np.isnan(q_arr).any() or np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be in the interval [0, 1]")
    return q_arr


def _as_2d_float(X_chunk):
    """Convert a chunk to a float array of shape (n_samples, n_features)."""
    arr = _as_numeric(X_chunk).astype(float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"X_chunk must be 1D or 2D, got {arr.ndim}D")
    return arr


def chunk_moments(X):
    """Count, mean and sum of squared deviations of each column, ignoring NaN.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Float input data.

    Returns
    -------
    count : ndarray, shape (n_features,)
        Number of non-NaN values per column.
    mean : ndarray, shape (n_features,)
        Column means (NaN for columns with no valid value).
    m2 : ndarray, shape (n_features,)
        Sum of squared deviations from the column mean.
    """
    valid = ~np.isnan(X)
    count = valid.sum(axis=0)
    total = np.where(valid, X, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.full(X.shape[1], np.nan),
                     where=count > 0)
    deviations = np.where(valid, X - mean, 0.0)
    m2 = (deviations ** 2).sum(axis=0)
    return count, mean, m2


def merge_moments(count_a, mean_a, m2_a, count_b, mean_b, m2_b):
    """Combine the moments of two disjoint groups of data.

    Uses the parallel update of Chan et al. (1979), which generalises
    Welford's algorithm from one sample to a whole chunk. It avoids the
    cancellation error of the naive ``E[x^2] - E[x]^2`` formula.

    Parameters
    ----------
    count_a, mean_a, m2_a : ndarray, shape (n_features,)
        Moments of the first group, as returned by :func:`chunk_moments`.
    count_b, mean_b, m2_b : ndarray, shape (n_features,)
        Moments of the second group.

    Returns
    -------
    count, mean, m2 : ndarray, shape (n_features,)
        Moments of the union of both groups.
    """
    count = count_a + count_b
    safe_count = np.where(count > 0, count, 1)
    # A group without data has a NaN mean: replace it so it adds nothing.
    mean_a = np.where(count_a > 0, mean_a, 0.0)
    mean_b = np.where(count_b > 0, mean_b, 0.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * count_b / safe_count
    m2 = m2_a + m2_b + delta ** 2 * count_a * count_b / safe_count
    return count, np.where(count > 0, mean, np.nan), m2


class StreamingStats:
    """Running per-feature statistics over a stream of data chunks.

    Mean and variance are updated exactly with :func:`merge_moments`, so
    each chunk is processed in one vectorised pass and past data does not
    need to be kept. Quantiles are estimated from a reservoir sample of at
    most ``max_samples`` rows: they are exact until that many rows have been
    seen, and an unbiased random sample of the stream afterwards.
    NaN values are treated as missing and ignored feature by feature.

    Parameters
    ----------
    ddof : int, default=0
        Delta degrees of freedom used for ``var_`` and ``std_``.
    max_samples : int, default=10000
        Size of the reservoir used for quantile estimates.
    seed : int or None, default=None
        Seed of the random generator used for reservoir sampling.

    Attributes
    ----------
    n_samples_seen_ : int
        Number of rows seen so far, including rows containing NaN.
    count_ : ndarray, shape (n_features,)
        Number of non-NaN values seen for each feature.
    mean_, var_, std_, min_, max_ : ndarray, shape (n_features,)
        Running statistics. They are floats when the chunks are 1D, and
        NaN for features that have no valid value yet.

    Complexity
    ----------
    ``update_stats``: time O(n_chunk * n_features),
    space O(max_samples * n_features).

    Examples
    --------
    >>> stats = StreamingStats()
    >>> for chunk in np.array_split(np.arange(10.0), 3):
    ...     _ = stats.update_stats(chunk)
    >>> stats.mean_
    4.5
    """

    def __init__(self, ddof=0, max_samples=10000, seed=None):
        if not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.ddof = ddof
        self.max_samples = max_samples
        self.seed = seed
        self.reset()

    def reset(self):
        """Forget all the data seen so far.

        Returns
        -------
        self : StreamingStats
        """
        self.n_samples_seen_ = 0
        self.n_features_ = None
        self._is_1d = False
        self._count = None
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None
        self._reservoir = None
        self._rng = np.random.default_rng(self.seed)
        return self

    def update_stats(self, X_chunk):
        """Update the statistics with a new chunk of data.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples,) or (n_samples, n_features)
            New data. A 1D chunk is treated as a single feature.

        Returns
        -------
        self : StreamingStats

        Raises
        ------
        TypeError
            If the chunk is not numeric.
        ValueError
            If the chunk is not 1D/2D or its number of features differs
            from the previous chunks.
        """
        X = _as_2d_float(X_chunk)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self._is_1d = np.ndim(X_chunk) == 1
            empty = np.zeros(self.n_features_)
            self._count, self._m2 = empty.copy(), empty.copy()
            self._mean = np.full(self.n_features_, np.nan)
            self._min = np.full(self.n_features_, np.nan)
            self._max = np.full(self.n_features_, np.nan)
            self._reservoir = np.empty((0, self.n_features_))
        elif X.shape[1] != self.n_features_:
            raise ValueError(f"X_chunk has {X.shape[1]} features, expected "
                             f"{self.n_features_}")
        if X.shape[0] == 0:
            return self

        count, mean, m2 = chunk_moments(X)
        self._count, self._mean, self._m2 = merge_moments(
            self._count, self._mean, self._m2, count, mean, m2)

        valid = ~np.isnan(X)
        chunk_min = np.where(valid, X, np.inf).min(axis=0)
        chunk_max = np.where(valid, X, -np.inf).max(axis=0)
        # fmin/fmax ignore NaN, so features without data stay NaN.
        self._min = np.fmin(self._min, np.where(count > 0, chunk_min, np.nan))
        self._max = np.fmax(self._max, np.where(count > 0, chunk_max, np.nan))

        self._update_reservoir(X)
        self.n_samples_seen_ += X.shape[0]
        return self

    def _update_reservoir(self, X):
        """Keep a uniform random sample of rows (Vitter's Algorithm R)."""
        free = self.max_samples - self._reservoir.shape[0]
        if free > 0:
            self._reservoir = np.vstack([self._reservoir, X[:free]])
        rest = X[max(free, 0):]
        if rest.shape[0] == 0:
            return
        # Row number i (0-based, over the whole stream) replaces a random
        # slot j in [0, i] when j falls inside the reservoir.
        first = self.n_samples_seen_ + max(free, 0)
        positions = first + np.arange(rest.shape[0])
        slots = self._rng.integers(0, positions + 1)
        keep = slots < self.max_samples
        # With repeated slots NumPy keeps the last assignment, which matches
        # the sequential algorithm where later rows overwrite earlier ones.
        self._reservoir[slots[keep]] = rest[keep]

    def _format(self, values):
        """Return a float for 1D streams, an array otherwise."""
        self._check_fitted()
        values = np.asarray(values, dtype=float)
        if not self._is_1d:
            return values
        values = values[..., 0]
        return float(values) if values.ndim == 0 else values

    def _check_fitted(self):
        if self.n_features_ is None:
            raise ValueError("No data seen yet: call update_stats first")

    @property
    def count_(self):
        return self._format(self._count)

    @property
    def mean_(self):
        return self._format(self._mean)

    @property
    def var_(self):
        self._check_fitted()
        dof = self._count - self.ddof
        var = np.divide(self._m2, dof, out=np.full(self.n_features_, np.nan),
                        where=dof > 0)
        return self._format(var)

    @property
    def std_(self):
        return np.sqrt(self.var_)

    @property
    def min_(self):
        return self._format(self._min)

    @property
    def max_(self):
        return self._format(self._max)

    def quantiles(self, q):
        """Estimate quantiles of each feature from the reservoir sample.

        Parameters
        ----------
        q : float or array_like
            Quantile values in the interval ``[0, 1]``.

        Returns
        -------
        float or ndarray
            Shape (n_features,) for a scalar ``q``, (n_q, n_features) for an
            array; the feature axis is dropped for 1D streams.

        Raises
        ------
        ValueError
            If no data has been seen or a quantile is outside ``[0, 1]``.
        """
        q_arr = _check_q(q)
        self._check_fitted()
        with warnings.catch_warnings():
            # Features that only contain NaN give NaN, without a warning.
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanquantile(self._reservoir, q_arr, axis=0)
        return self._format(result)


class StreamingHistogram:
    """Histogram updated chunk by chunk, optionally over a sliding window.

    The bin edges are fixed once: from ``range`` if given, otherwise from the
    finite values of the first non-empty chunk. Later values outside the
    edges are not binned but are counted in ``n_outside_``. NaN and infinite
    values are ignored. Like :func:`histogram`, the input is flattened.

    Parameters
    ----------
    bins : int, default=10
        Number of equal-width bins.
    range : tuple of float or None, default=None
        Lower and upper edge of the bins.
    window : int or None, default=None
        If given, only the last ``window`` chunks are counted.

    Attributes
    ----------
    counts_ : ndarray of int, shape (bins,)
        Number of values in each bin (within the window, if any).
    edges_ : ndarray, shape (bins + 1,)
        Bin edges.
    n_outside_ : int
        Number of finite values that fell outside the edges.

    Complexity
    ----------
    ``update_stats``: time O(n_chunk + bins), space O(window * bins).
    """

    def __init__(self, bins=10, range=None, window=None):
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError("bins must be a positive integer")
        if window is not None and (
                not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError("window must be None or a positive integer")
        if range is not None and not range[0] <= range[1]:
            raise ValueError("range must satisfy range[0] <= range[1]")
        self.bins = bins
        self.range = range
        self.window = window
        self.reset()

    def reset(self):
        """Forget all the data seen so far.

        Returns
        -------
        self : StreamingHistogram
        """
        self.edges_ = None
        self.counts_ = None
        self.n_outside_ = 0
        self._history = deque() if self.window is not None else None
        return self

    def update_stats(self, X_chunk):
        """Add the values of a new chunk to the histogram.

        Parameters
        ----------
        X_chunk : array_like
            Numeric values of any shape.

        Returns
        -------
        self : StreamingHistogram

        Raises
        ------
        TypeError
            If the chunk is not numeric.
        """
        arr = _as_numeric(X_chunk).astype(float).ravel()
        values = arr[np.isfinite(arr)]
        if self.edges_ is None:
            if self.range is not None:
                low, high = self.range
            elif values.size > 0:
                low, high = values.min(), values.max()
            else:
                return self
            # NumPy widens an empty range (low == high) to low - 0.5, high + 0.5.
            self.edges_ = np.histogram_bin_edges(values, bins=self.bins,
                                                 range=(low, high))
            self.counts_ = np.zeros(self.bins, dtype=int)

        inside = (values >= self.edges_[0]) & (values <= self.edges_[-1])
        chunk_counts, _ = np.histogram(values[inside], bins=self.edges_)
        chunk_outside = int(values.size - inside.sum())

        if self._history is not None:
            if len(self._history) == self.window:
                old_counts, old_outside = self._history.popleft()
                self.counts_ -= old_counts
                self.n_outside_ -= old_outside
            self._history.append((chunk_counts, chunk_outside))
        self.counts_ += chunk_counts
        self.n_outside_ += chunk_outside
        return self

    def result(self):
        """Return the current histogram.

        Returns
        -------
        counts : ndarray of int, shape (bins,)
        edges : ndarray, shape (bins + 1,)

        Raises
        ------
        ValueError
            If the bin edges are not known yet.
        """
        if self.edges_ is None:
            raise ValueError("No finite data seen yet: call update_stats first")
        return self.counts_.copy(), self.edges_.copy()
