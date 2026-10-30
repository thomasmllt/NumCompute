"""Classification and regression metrics, in batch and streaming form.

Each streaming metric follows the same API: ``update(y_true_chunk,
y_pred_chunk)`` adds a chunk, ``result()`` returns the current value and
``reset()`` forgets everything. With ``window=k``, only the last ``k`` chunks
are counted, which gives a rolling metric.
"""

from collections import deque

import numpy as np

from .rank import rank

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "roc_auc",
    "StreamingMetric",
    "StreamingConfusionMatrix",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingAUC",
    "StreamingMSE",
]


def _check_inputs(y_true, y_pred):
    """Convert inputs to arrays and check that they have the same size."""
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if true.size != pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    return true, pred


def _drop_missing(y_true, y_pred):
    """Check the inputs and remove the pairs containing a NaN."""
    true, pred = _check_inputs(y_true, y_pred)
    # NaN is the only value that differs from itself.
    keep = (true == true) & (pred == pred)
    return true[keep], pred[keep]


def accuracy(y_true, y_pred):
    """Return the proportion of correct predictions.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.

    Returns
    -------
    float
        Fraction of matching labels.

    Raises
    ------
    ValueError
        If the inputs have different lengths or are empty.
    """
    true, pred = _check_inputs(y_true, y_pred)
    if true.size == 0:
        raise ValueError("cannot calculate accuracy for empty data")
    return np.mean(true == pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """Return a confusion matrix with rows for true labels.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    labels : array_like or None, default=None
        Labels to include and their row/column order. If None, labels are
        inferred from both inputs.

    Returns
    -------
    ndarray, shape (n_classes, n_classes)
        Counts with true labels on rows and predicted labels on columns.

    Raises
    ------
    ValueError
        If the inputs have different lengths.
    """
    true, pred = _check_inputs(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    if len(labels) == 0:
        return matrix

    true_matches = true[:, None] == labels
    pred_matches = pred[:, None] == labels
    true_indices = np.argmax(true_matches, axis=1)
    pred_indices = np.argmax(pred_matches, axis=1)
    known = true_matches.any(axis=1) & pred_matches.any(axis=1)
    np.add.at(matrix, (true_indices[known], pred_indices[known]), 1)
    return matrix


def precision(y_true, y_pred, positive_label=1):
    """Return binary precision for the selected positive label.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    positive_label : scalar, default=1
        Label treated as positive.

    Returns
    -------
    float
        True positives divided by predicted positives, or 0.0 when the
        denominator is zero.
    """
    true, pred = _check_inputs(y_true, y_pred)
    true_positive = np.sum((true == positive_label) & (pred == positive_label))
    false_positive = np.sum((true != positive_label) & (pred == positive_label))
    total = true_positive + false_positive
    return 0.0 if total == 0 else true_positive / total


def recall(y_true, y_pred, positive_label=1):
    """Return binary recall for the selected positive label.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    positive_label : scalar, default=1
        Label treated as positive.

    Returns
    -------
    float
        True positives divided by actual positives, or 0.0 when the
        denominator is zero.
    """
    true, pred = _check_inputs(y_true, y_pred)
    true_positive = np.sum((true == positive_label) & (pred == positive_label))
    false_negative = np.sum((true == positive_label) & (pred != positive_label))
    total = true_positive + false_negative
    return 0.0 if total == 0 else true_positive / total


def f1(y_true, y_pred, positive_label=1):
    """Return the binary F1 score for the selected positive label.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    positive_label : scalar, default=1
        Label treated as positive.

    Returns
    -------
    float
        Harmonic mean of precision and recall, or 0.0 when both are zero.
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def mse(y_true, y_pred):
    """Return the mean squared error.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted numeric values.

    Returns
    -------
    float
        Mean of the squared prediction errors.

    Raises
    ------
    ValueError
        If the inputs have different lengths or are empty.
    """
    true, pred = _check_inputs(y_true, y_pred)
    if true.size == 0:
        raise ValueError("cannot calculate MSE for empty data")
    return np.mean((true - pred) ** 2)


def roc_auc(y_true, y_score, positive_label=1):
    """Return the area under the ROC curve.

    It is computed from the Mann-Whitney U statistic: the probability that a
    random positive sample gets a higher score than a random negative one,
    ties counting for one half. Pairs containing a NaN are ignored.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_score : array_like, shape (n_samples,)
        Predicted scores, higher meaning more likely to be positive.
    positive_label : scalar, default=1
        Label treated as positive.

    Returns
    -------
    float
        Area under the curve, or NaN if only one class is present.

    Raises
    ------
    ValueError
        If the inputs have different lengths.
    """
    true, score = _drop_missing(y_true, y_score)
    positive = true == positive_label
    n_positive = np.sum(positive)
    n_negative = true.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return np.nan
    rank_sum = np.sum(rank(score.astype(float), method="average")[positive])
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


class StreamingMetric:
    """Base class of the streaming metrics.

    A metric is defined by a state computed on each chunk (an array of
    counts or sums), which only has to be added up. Subclasses compute that
    state in ``update`` and give it to ``_add_chunk``.

    Parameters
    ----------
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.

    Complexity
    ----------
    ``update``: space O(state) without window, O(window * state) with one.
    """

    def __init__(self, window=None):
        if window is not None and (
                not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError("window must be None or a positive integer")
        self.window = window
        self.reset()

    def reset(self):
        """Forget all the data seen so far.

        Returns
        -------
        self : StreamingMetric
        """
        self._total = None
        self._history = deque(maxlen=self.window)
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add a new chunk of data.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array_like, shape (n_samples,)
            True and predicted values. Pairs containing a NaN are ignored.

        Returns
        -------
        self : StreamingMetric

        Raises
        ------
        ValueError
            If the inputs have different lengths.
        """
        raise NotImplementedError

    def result(self):
        """Return the current value of the metric."""
        raise NotImplementedError

    def _add_chunk(self, state):
        if self.window is None:
            self._total = state if self._total is None else self._total + state
        else:
            # Summing again is safer than subtracting the oldest chunk, which
            # would accumulate rounding errors on float states.
            self._history.append(state)
            self._total = sum(self._history)

    def _check_updated(self):
        if self._total is None:
            raise ValueError("No data seen yet: call update first")


class StreamingConfusionMatrix(StreamingMetric):
    """Confusion matrix accumulated over a stream of chunks.

    Parameters
    ----------
    labels : array_like or None, default=None
        Labels to include and their row/column order. Other labels are
        ignored. If None, labels are added as they appear, in sorted order.
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.

    Attributes
    ----------
    labels_ : ndarray, shape (n_classes,)
        Labels of the rows and columns of the matrix.

    Complexity
    ----------
    ``update``: time O(n_chunk * n_classes), space O(n_classes^2), multiplied
    by ``window`` if there is one.
    """

    def __init__(self, labels=None, window=None):
        self.labels = labels
        super().__init__(window)

    def reset(self):
        """Forget all the data seen so far.

        Returns
        -------
        self : StreamingConfusionMatrix
        """
        self.labels_ = None if self.labels is None else np.asarray(self.labels)
        return super().reset()

    def update(self, y_true_chunk, y_pred_chunk):
        """Add a new chunk of data.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array_like, shape (n_samples,)
            True and predicted labels. Pairs containing a NaN are ignored.

        Returns
        -------
        self : StreamingConfusionMatrix

        Raises
        ------
        ValueError
            If the inputs have different lengths.
        """
        true, pred = _drop_missing(y_true_chunk, y_pred_chunk)
        if true.size == 0:
            return self
        if self.labels is None:
            self._add_labels(np.unique(np.concatenate((true, pred))))
        self._add_chunk(confusion_matrix(true, pred, self.labels_))
        return self

    def _add_labels(self, labels):
        """Insert new labels, moving the stored counts to their new position."""
        if self.labels_ is None:
            self.labels_ = labels
            return
        merged = np.union1d(self.labels_, labels)
        if merged.size == self.labels_.size:
            return
        old_position = np.searchsorted(merged, self.labels_)

        def resize(matrix):
            resized = np.zeros((merged.size, merged.size), dtype=int)
            resized[np.ix_(old_position, old_position)] = matrix
            return resized

        self._history = deque(map(resize, self._history), maxlen=self.window)
        if self._total is not None:
            self._total = resize(self._total)
        self.labels_ = merged

    def result(self):
        """Return the current confusion matrix.

        Returns
        -------
        ndarray, shape (n_classes, n_classes)
            Counts with true labels on rows and predicted labels on columns.

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        self._check_updated()
        return self._total.copy()


class StreamingAccuracy(StreamingConfusionMatrix):
    """Accuracy accumulated over a stream of chunks.

    Parameters
    ----------
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.
    """

    def __init__(self, window=None):
        super().__init__(window=window)

    def result(self):
        """Return the proportion of correct predictions.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        self._check_updated()
        return float(np.trace(self._total) / self._total.sum())


class _BinaryMetric(StreamingConfusionMatrix):
    """Base of the metrics that focus on one positive label."""

    def __init__(self, positive_label=1, window=None):
        self.positive_label = positive_label
        super().__init__(window=window)

    def _binary_counts(self):
        """Return the true positives, false positives and false negatives."""
        self._check_updated()
        found = np.flatnonzero(self.labels_ == self.positive_label)
        if found.size == 0:
            return 0, 0, 0
        i = found[0]
        true_positive = self._total[i, i]
        false_positive = self._total[:, i].sum() - true_positive
        false_negative = self._total[i, :].sum() - true_positive
        return true_positive, false_positive, false_negative


class StreamingPrecision(_BinaryMetric):
    """Binary precision accumulated over a stream of chunks.

    Parameters
    ----------
    positive_label : scalar, default=1
        Label treated as positive.
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.
    """

    def result(self):
        """Return the precision, or 0.0 when the denominator is zero.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        true_positive, false_positive, _ = self._binary_counts()
        total = true_positive + false_positive
        return 0.0 if total == 0 else float(true_positive / total)


class StreamingRecall(_BinaryMetric):
    """Binary recall accumulated over a stream of chunks.

    Parameters
    ----------
    positive_label : scalar, default=1
        Label treated as positive.
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.
    """

    def result(self):
        """Return the recall, or 0.0 when the denominator is zero.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        true_positive, _, false_negative = self._binary_counts()
        total = true_positive + false_negative
        return 0.0 if total == 0 else float(true_positive / total)


class StreamingF1(_BinaryMetric):
    """Binary F1 score accumulated over a stream of chunks.

    Parameters
    ----------
    positive_label : scalar, default=1
        Label treated as positive.
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.
    """

    def result(self):
        """Return the F1 score, or 0.0 when there is no true positive.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        true_positive, false_positive, false_negative = self._binary_counts()
        total = 2 * true_positive + false_positive + false_negative
        return 0.0 if total == 0 else float(2 * true_positive / total)


class StreamingAUC(StreamingMetric):
    """Area under the ROC curve accumulated over a stream of chunks.

    Keeping every score would grow without limit, so scores are counted in
    ``n_bins`` equal-width bins, for each class. Two scores in the same bin
    are treated as a tie: the error against :func:`roc_auc` is bounded by the
    share of such pairs and shrinks as ``n_bins`` grows.

    Parameters
    ----------
    positive_label : scalar, default=1
        Label treated as positive.
    n_bins : int, default=1000
        Number of bins used to count the scores.
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.

    Complexity
    ----------
    ``update``: time O(n_chunk + n_bins), space O(n_bins), multiplied by
    ``window`` if there is one.
    """

    def __init__(self, positive_label=1, n_bins=1000, window=None):
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        self.positive_label = positive_label
        self.n_bins = n_bins
        super().__init__(window)

    def update(self, y_true_chunk, y_score_chunk):
        """Add a new chunk of data.

        Parameters
        ----------
        y_true_chunk : array_like, shape (n_samples,)
            True labels.
        y_score_chunk : array_like, shape (n_samples,)
            Predicted probabilities of the positive label, in [0, 1].
            Pairs containing a NaN are ignored.

        Returns
        -------
        self : StreamingAUC

        Raises
        ------
        ValueError
            If the inputs have different lengths or a score is out of range.
        """
        true, score = _drop_missing(y_true_chunk, y_score_chunk)
        if true.size == 0:
            return self
        score = score.astype(float)
        if score.min() < 0.0 or score.max() > 1.0:
            raise ValueError("y_score_chunk must contain values in [0, 1]")
        bins = np.minimum((score * self.n_bins).astype(int), self.n_bins - 1)
        positive = true == self.positive_label
        self._add_chunk(np.stack((
            np.bincount(bins[positive], minlength=self.n_bins),
            np.bincount(bins[~positive], minlength=self.n_bins),
        )))
        return self

    def result(self):
        """Return the area under the curve.

        Returns
        -------
        float
            Area under the curve, or NaN if only one class has been seen.

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        self._check_updated()
        positive, negative = self._total
        n_positive, n_negative = positive.sum(), negative.sum()
        if n_positive == 0 or n_negative == 0:
            return np.nan
        negative_below = np.cumsum(negative) - negative
        wins = np.sum(positive * (negative_below + 0.5 * negative))
        return float(wins / (n_positive * n_negative))


class StreamingMSE(StreamingMetric):
    """Mean squared error accumulated over a stream of chunks.

    Parameters
    ----------
    window : int or None, default=None
        If given, only the last ``window`` non-empty chunks are counted.
    """

    def update(self, y_true_chunk, y_pred_chunk):
        """Add a new chunk of data.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array_like, shape (n_samples,)
            True and predicted numeric values. Pairs containing a NaN are
            ignored.

        Returns
        -------
        self : StreamingMSE

        Raises
        ------
        ValueError
            If the inputs have different lengths.
        """
        true, pred = _drop_missing(y_true_chunk, y_pred_chunk)
        if true.size > 0:
            self._add_chunk(np.array([np.sum((true - pred) ** 2), true.size]))
        return self

    def result(self):
        """Return the mean of the squared prediction errors.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no data has been seen yet.
        """
        self._check_updated()
        return float(self._total[0] / self._total[1])
