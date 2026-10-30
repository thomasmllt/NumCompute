import numpy as np
import pytest

from numcompute_stream.metrics import (
    accuracy,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
)


def test_accuracy():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_confusion_matrix():
    result = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
    assert np.array_equal(result, [[1, 1], [1, 1]])


def test_classification_metrics():
    true = [1, 1, 1, 0]
    pred = [1, 1, 0, 1]
    assert precision(true, pred) == 2 / 3
    assert recall(true, pred) == 2 / 3
    assert f1(true, pred) == 2 / 3


def test_mse():
    assert mse([1, 2, 3], [1, 4, 2]) == 5 / 3


def test_metrics_reject_different_lengths():
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])


def test_metrics_empty_inputs():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        mse([], [])


def test_metrics_all_predictions_equal():
    assert accuracy([1, 1, 1], [1, 1, 1]) == 1.0
    assert confusion_matrix([1, 1, 1], [1, 1, 1]).tolist() == [[3]]
