"""Simple classification and regression metrics."""

import numpy as np

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
]


def _check_inputs(y_true, y_pred):
    """Convert inputs to arrays and check that they have the same size."""
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if true.size != pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    return true, pred


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
