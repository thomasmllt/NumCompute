"""Decision tree classifier with buffered retraining for streaming data.

A decision tree cannot be updated cheaply one sample at a time, so
partial_fit() stores incoming chunks in a buffer and retrains the tree once
the buffer is full. Split search and prediction are vectorised with NumPy.
"""
import numpy as np

from .pipeline import Estimator
from .preprocessing import to_numeric

__all__ = ["DecisionTreeClassifier"]


def _impurity(counts, criterion):
    """Compute the impurity of class counts.

    Parameters
    ----------
    counts : ndarray, shape (..., n_classes)
        Number of samples of each class.
    criterion : {"gini", "entropy"}
        Impurity measure.

    Returns
    -------
    ndarray, shape (...)
        Impurity of each group (0.0 for a pure or empty group).
    """
    totals = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    if criterion == "gini":
        return 1.0 - (p ** 2).sum(axis=-1)
    log_p = np.log2(p, out=np.zeros(p.shape), where=p > 0)
    return -(p * log_p).sum(axis=-1)


def _best_split(X, y, n_classes, criterion):
    """Find the split with the largest impurity decrease.

    All candidate thresholds of all features are evaluated at once. NaN
    values are left out of the gain of a feature, and the gain is scaled by
    the fraction of valid samples so that features with many missing values
    are penalised.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples reaching the node.
    y : ndarray of int, shape (n_samples,)
        Encoded labels.
    n_classes : int
        Total number of classes.
    criterion : {"gini", "entropy"}
        Impurity measure.

    Returns
    -------
    (feature, threshold) or None
        Position of the best column and its threshold, or None if no column
        can separate the samples. Ties go to the lowest feature, then to the
        lowest threshold.

    Notes
    -----
    Time O(n_samples * n_features * (log(n_samples) + n_classes)),
    space O(n_samples * n_features * n_classes).
    """
    n_samples, n_features = X.shape
    order = np.argsort(X, axis=0)  # NaN values are sorted last
    X_sorted = np.take_along_axis(X, order, axis=0)

    # A threshold can only sit between two different values. Comparisons
    # with NaN are False, so missing values never create a candidate.
    features, rows = np.nonzero((X_sorted[:-1] < X_sorted[1:]).T)
    if features.size == 0:
        return None

    n_valid = (~np.isnan(X)).sum(axis=0)
    left_cumulative = np.cumsum(
        np.eye(n_classes, dtype=np.int32)[y[order]], axis=0)
    valid_totals = left_cumulative[n_valid - 1, np.arange(n_features)]

    left = left_cumulative[rows, features]
    right = valid_totals[features] - left
    n_left = rows + 1
    n_total = n_valid[features]
    child_impurity = (n_left * _impurity(left, criterion)
                      + (n_total - n_left) * _impurity(right, criterion)) / n_total
    parent_impurity = _impurity(valid_totals, criterion)[features]
    gain = (parent_impurity - child_impurity) * n_total / n_samples

    # Rounding stops floating point noise from deciding between equal splits.
    best = np.argmax(np.round(gain, 12))
    feature, row = features[best], rows[best]
    low, high = X_sorted[row, feature], X_sorted[row + 1, feature]
    threshold = low / 2 + high / 2
    # The midpoint of two adjacent floats can round up to the larger one.
    return int(feature), float(low if threshold >= high else threshold)


class _Tree:
    """Binary tree stored as parallel arrays, indexed by node id.

    Parameters
    ----------
    feature : array_like of int, shape (n_nodes,)
        Feature tested at each node, -1 for leaves.
    threshold : array_like of float, shape (n_nodes,)
        Samples with ``x <= threshold`` go to the left child.
    left, right : array_like of int, shape (n_nodes,)
        Child node ids.
    missing_left : array_like of bool, shape (n_nodes,)
        Whether samples with a NaN value go to the left child.
    proba : array_like of float, shape (n_nodes, n_classes)
        Class probabilities of the training samples in each node.
    depth : int
        Depth of the deepest leaf.
    """

    def __init__(self, feature, threshold, left, right, missing_left, proba,
                 depth):
        self.feature = np.asarray(feature, dtype=int)
        self.threshold = np.asarray(threshold, dtype=float)
        self.left = np.asarray(left, dtype=int)
        self.right = np.asarray(right, dtype=int)
        self.missing_left = np.asarray(missing_left, dtype=bool)
        self.proba = np.asarray(proba, dtype=float)
        self.depth = depth

    def apply(self, X):
        """Return the leaf id of each row of X, shape (n_samples,).

        All samples move down one level per iteration, so the loop runs at
        most ``depth`` times.
        """
        nodes = np.zeros(X.shape[0], dtype=int)
        active = np.flatnonzero(self.feature[nodes] >= 0)
        while active.size > 0:
            current = nodes[active]
            values = X[active, self.feature[current]]
            go_left = np.where(np.isnan(values), self.missing_left[current],
                               values <= self.threshold[current])
            nodes[active] = np.where(go_left, self.left[current],
                                     self.right[current])
            active = active[self.feature[nodes[active]] >= 0]
        return nodes


class DecisionTreeClassifier(Estimator):
    """Depth-limited CART classifier with buffered streaming updates.

    ``fit()`` trains the tree on the given data. ``partial_fit()`` appends a
    chunk to an internal buffer and, once it holds at least ``buffer_size``
    samples, retrains the tree on the buffered data and clears the buffer.
    Between retrains, ``predict()`` uses the most recently trained tree.

    NaN values in X are treated as missing: they are ignored when searching
    for a split, then sent to the child that received the most samples.
    Probability ties are resolved in favour of the lowest class in
    ``classes_``.

    Parameters
    ----------
    max_depth : int or None, default=5
        Maximum depth of the tree. ``None`` means no limit.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure used to choose the splits.
    min_samples_split : int, default=2
        Minimum number of samples a node needs to be split.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features drawn at random at each node: a count, a fraction
        of the features, or all of them with ``None``. The other features are
        only tried if the drawn ones cannot split the node.
    buffer_size : int, default=500
        Number of buffered samples that triggers a retrain.
    seed : int or None, default=None
        Seed of the random generator used for ``max_features``.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Sorted labels known to the current tree. They are kept across
        retrains, so a class missing from a buffer gets a probability of 0.
    n_features_ : int
        Number of features of the input data.
    n_retrains_ : int
        Number of times the tree has been trained.
    n_buffered_ : int
        Number of samples currently waiting in the buffer.
    depth_ : int
        Depth of the current tree.
    n_nodes_ : int
        Number of nodes of the current tree.

    Complexity
    ----------
    ``partial_fit``: amortised time O(n_chunk) plus one retrain per
    ``buffer_size`` samples, space O(buffer_size * n_features).
    ``predict``: time O(n_samples * depth), space O(n_samples).

    Examples
    --------
    >>> X = np.arange(40.0).reshape(-1, 1)
    >>> y = (X[:, 0] >= 20).astype(int)
    >>> tree = DecisionTreeClassifier(max_depth=2, buffer_size=30, seed=0)
    >>> for start in range(0, 40, 10):
    ...     _ = tree.partial_fit(X[start:start + 10], y[start:start + 10])
    >>> tree.n_retrains_, tree.n_buffered_
    (1, 10)
    """

    def __init__(self, max_depth=5, criterion="gini", min_samples_split=2,
                 max_features=None, buffer_size=500, seed=None):
        if max_depth is not None and (
                not isinstance(max_depth, (int, np.integer)) or max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Unknown criterion: {criterion!r}")
        if not isinstance(min_samples_split, (int, np.integer)) \
                or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")
        if isinstance(max_features, str):
            valid = max_features in ("sqrt", "log2")
        elif isinstance(max_features, (float, np.floating)):
            valid = 0.0 < max_features <= 1.0
        else:
            valid = max_features is None or max_features >= 1
        if not valid:
            raise ValueError(f"Invalid max_features: {max_features!r}")
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._tree = None
        self.classes_ = None
        self.n_features_ = None
        self.n_retrains_ = 0
        self._clear_buffer()

    def _clear_buffer(self):
        self._buffer_X = []
        self._buffer_y = []
        self.n_buffered_ = 0

    def _check_X(self, X, new_stream=False):
        """Return X as a float array of shape (n_samples, n_features).

        ``new_stream=True`` skips the comparison with the known feature count.
        """
        arr = to_numeric(X, "X").astype(float)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2D, got {arr.ndim}D")
        if np.isinf(arr).any():
            raise ValueError("X contains infinite values")
        if not new_stream and self.n_features_ is not None \
                and arr.shape[1] != self.n_features_:
            raise ValueError(f"X has {arr.shape[1]} features, expected "
                             f"{self.n_features_}")
        return arr

    def _check_X_y(self, X, y, new_stream=False):
        """Validate training data and return it as arrays."""
        arr = self._check_X(X, new_stream)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError(f"y must be 1D, got {labels.ndim}D")
        if labels.shape[0] != arr.shape[0]:
            raise ValueError(f"X has {arr.shape[0]} samples but y has "
                             f"{labels.shape[0]}")
        if np.issubdtype(labels.dtype, np.floating) and np.isnan(labels).any():
            raise ValueError("y contains NaN values")
        return arr, labels

    def fit(self, X, y):
        """Train the tree from scratch, discarding any previous state.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data. NaN values are treated as missing.
        y : array_like, shape (n_samples,)
            Class labels.

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        TypeError
            If X is not numeric.
        ValueError
            If the shapes are invalid, X is empty or contains infinite
            values, or y contains NaN.
        """
        X, y = self._check_X_y(X, y, new_stream=True)
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset")
        self._rng = np.random.default_rng(self.seed)
        self.classes_ = None
        self.n_retrains_ = 0
        self._clear_buffer()
        self.n_features_ = X.shape[1]
        self._train(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Add a chunk to the buffer and retrain if the buffer is full.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
            New samples. NaN values are treated as missing.
        y_chunk : array_like, shape (n_samples,)
            Class labels of the new samples.

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        TypeError
            If X_chunk is not numeric.
        ValueError
            If the shapes are invalid, the number of features changes, or
            the data contains infinite values or NaN labels.
        """
        X, y = self._check_X_y(X_chunk, y_chunk)
        self.n_features_ = X.shape[1]
        if X.shape[0] > 0:
            self._buffer_X.append(X)
            self._buffer_y.append(y)
            self.n_buffered_ += X.shape[0]
        if self.n_buffered_ >= self.buffer_size:
            self.flush()
        return self

    def flush(self):
        """Retrain on the buffered samples now, even if the buffer is not full.

        This is useful at the end of a stream, where the last samples may
        never fill the buffer. It does nothing if the buffer is empty.

        Returns
        -------
        self : DecisionTreeClassifier
        """
        if self.n_buffered_ > 0:
            X = np.vstack(self._buffer_X)
            y = np.concatenate(self._buffer_y)
            self._clear_buffer()
            self._train(X, y)
        return self

    def _train(self, X, y):
        """Replace the current tree by one trained on (X, y)."""
        known = y if self.classes_ is None else np.concatenate([self.classes_, y])
        self.classes_ = np.unique(known)
        encoded = np.searchsorted(self.classes_, y)
        self._tree = self._build(X, encoded)
        self.n_retrains_ += 1

    def _n_drawn_features(self):
        """Return how many features to draw at each node."""
        if self.max_features is None:
            return self.n_features_
        if self.max_features == "sqrt":
            count = np.sqrt(self.n_features_)
        elif self.max_features == "log2":
            count = np.log2(self.n_features_)
        elif isinstance(self.max_features, (float, np.floating)):
            count = self.max_features * self.n_features_
        else:
            count = self.max_features
        return int(np.clip(count, 1, self.n_features_))

    def _find_split(self, X, y):
        """Return (feature, threshold) for a node, or None if it cannot split."""
        n_classes = self.classes_.size
        n_drawn = self._n_drawn_features()
        if n_drawn == self.n_features_:
            return _best_split(X, y, n_classes, self.criterion)
        permutation = self._rng.permutation(self.n_features_)
        for drawn in (permutation[:n_drawn], permutation[n_drawn:]):
            columns = np.sort(drawn)
            split = _best_split(X[:, columns], y, n_classes, self.criterion)
            if split is not None:
                return int(columns[split[0]]), split[1]
        return None

    def _build(self, X, y):
        """Grow a tree with an explicit stack, so depth is not limited by
        Python's recursion limit."""
        feature, threshold, left, right, missing_left, proba = \
            [], [], [], [], [], []
        max_depth_reached = 0
        stack = [(np.arange(y.size), 0, -1, False)]
        while stack:
            samples, depth, parent, is_left = stack.pop()
            node = len(feature)
            if parent >= 0:
                (left if is_left else right)[parent] = node
            max_depth_reached = max(max_depth_reached, depth)

            counts = np.bincount(y[samples], minlength=self.classes_.size)
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            missing_left.append(False)
            proba.append(counts / samples.size)

            can_split = (samples.size >= self.min_samples_split
                         and np.count_nonzero(counts) > 1
                         and (self.max_depth is None or depth < self.max_depth))
            split = self._find_split(X[samples], y[samples]) if can_split else None
            if split is None:
                continue

            feature[node], threshold[node] = split
            values = X[samples, feature[node]]
            goes_left = values <= threshold[node]
            missing = np.isnan(values)
            missing_left[node] = bool(
                goes_left.sum() >= samples.size - missing.sum() - goes_left.sum())
            goes_left = goes_left | (missing & missing_left[node])
            stack.append((samples[~goes_left], depth + 1, node, False))
            stack.append((samples[goes_left], depth + 1, node, True))
        return _Tree(feature, threshold, left, right, missing_left, proba,
                     max_depth_reached)

    def _check_fitted(self):
        if self._tree is None:
            raise ValueError("This DecisionTreeClassifier has no trained tree "
                             "yet: call fit, or partial_fit until the buffer "
                             "is full, or flush")

    def predict_proba(self, X):
        """Predict class probabilities with the most recently trained tree.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples to predict. NaN values are treated as missing.

        Returns
        -------
        ndarray, shape (n_samples, n_classes)
            Probabilities, with columns in the order of ``classes_``.

        Raises
        ------
        ValueError
            If no tree has been trained yet or X has an invalid shape.
        """
        self._check_fitted()
        arr = self._check_X(X)
        return self._tree.proba[self._tree.apply(arr)]

    def predict(self, X):
        """Predict class labels with the most recently trained tree.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples to predict. NaN values are treated as missing.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted labels, taken from ``classes_``.

        Raises
        ------
        ValueError
            If no tree has been trained yet or X has an invalid shape.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def is_fitted_(self):
        return self._tree is not None

    @property
    def depth_(self):
        self._check_fitted()
        return self._tree.depth

    @property
    def n_nodes_(self):
        self._check_fitted()
        return self._tree.feature.size
