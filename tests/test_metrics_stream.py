"""Unit tests for the streaming metrics.

Each streaming result is compared with the batch metric computed on the same
data, after feeding it chunk by chunk.
"""
import numpy as np
import pytest

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingMSE,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_auc,
)

ALL_METRICS = [StreamingConfusionMatrix, StreamingAccuracy, StreamingPrecision,
               StreamingRecall, StreamingF1, StreamingAUC, StreamingMSE]
PAIRS = [(StreamingAccuracy, accuracy), (StreamingPrecision, precision),
         (StreamingRecall, recall), (StreamingF1, f1), (StreamingMSE, mse)]


def _stream(metric, y_true, y_pred, n_chunks):
    for true, pred in zip(np.array_split(y_true, n_chunks),
                          np.array_split(y_pred, n_chunks)):
        metric.update(true, pred)
    return metric


def _labels(seed, n=500):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    return y_true, np.where(rng.random(n) < 0.8, y_true, 1 - y_true)


def _scores(seed, n=2000):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    return y_true, np.clip(0.35 * y_true + rng.normal(0.35, 0.2, size=n), 0, 1)


@pytest.mark.parametrize("make, batch", PAIRS)
def test_stream_matches_batch(make, batch):
    y_true, y_pred = _labels(0)
    assert np.isclose(_stream(make(), y_true, y_pred, 7).result(),
                      batch(y_true, y_pred))


@pytest.mark.parametrize("make, batch", PAIRS)
def test_window_matches_batch_on_last_chunks(make, batch):
    y_true, y_pred = _labels(1, n=600)
    metric = _stream(make(window=2), y_true, y_pred, 6)
    assert np.isclose(metric.result(), batch(y_true[400:], y_pred[400:]))


@pytest.mark.parametrize("make", ALL_METRICS)
def test_result_requires_data(make):
    with pytest.raises(ValueError):
        make().result()


@pytest.mark.parametrize("make", ALL_METRICS)
def test_reset_forgets_data(make):
    metric = make().update([0, 1], [0.0, 1.0]).reset()
    with pytest.raises(ValueError):
        metric.result()


@pytest.mark.parametrize("make", ALL_METRICS)
def test_update_rejects_different_lengths(make):
    with pytest.raises(ValueError):
        make().update([0, 1], [1.0])


@pytest.mark.parametrize("make", ALL_METRICS)
def test_empty_chunk_is_ignored(make):
    metric = make().update([0, 1, 1], [0.0, 1.0, 0.0])
    before = metric.result()
    metric.update([], [])
    assert np.array_equal(metric.result(), before)


@pytest.mark.parametrize("make", ALL_METRICS)
def test_only_nan_chunk_counts_as_no_data(make):
    metric = make().update([np.nan], [1.0])
    with pytest.raises(ValueError):
        metric.result()


@pytest.mark.parametrize("make", ALL_METRICS)
@pytest.mark.parametrize("window", [0, -1, 1.5])
def test_invalid_window(make, window):
    with pytest.raises(ValueError):
        make(window=window)


def test_accuracy_multiclass_and_strings():
    y_true = np.array(["a", "b", "c", "a"])
    y_pred = np.array(["a", "c", "c", "b"])
    metric = _stream(StreamingAccuracy(), y_true, y_pred, 4)
    assert metric.result() == 0.5


def test_metrics_ignore_nan_pairs():
    metric = StreamingAccuracy().update([1.0, np.nan, 0.0], [1.0, 0.0, np.nan])
    assert metric.result() == 1.0


def test_precision_without_positive_prediction_is_zero():
    assert StreamingPrecision().update([1, 1], [0, 0]).result() == 0.0
    assert StreamingRecall().update([0, 0], [0, 0]).result() == 0.0
    assert StreamingF1().update([0, 0], [0, 0]).result() == 0.0


def test_positive_label_is_used():
    y_true, y_pred = ["cat", "dog", "dog"], ["cat", "cat", "dog"]
    metric = StreamingPrecision(positive_label="cat").update(y_true, y_pred)
    assert metric.result() == 0.5


def test_confusion_matrix_stream_matches_batch():
    rng = np.random.default_rng(2)
    y_true, y_pred = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    metric = _stream(StreamingConfusionMatrix(), y_true, y_pred, 5)
    assert np.array_equal(metric.result(), confusion_matrix(y_true, y_pred))


def test_confusion_matrix_grows_with_new_labels():
    metric = StreamingConfusionMatrix().update([1, 3], [1, 3])
    metric.update([0, 1], [0, 3])
    assert metric.labels_.tolist() == [0, 1, 3]
    assert metric.result().tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_confusion_matrix_window_with_new_labels():
    metric = StreamingConfusionMatrix(window=2)
    metric.update([1, 1], [1, 1])
    metric.update([0], [1])
    metric.update([2], [2])
    assert metric.labels_.tolist() == [0, 1, 2]
    assert metric.result().tolist() == [[0, 1, 0], [0, 0, 0], [0, 0, 1]]


def test_confusion_matrix_fixed_labels():
    metric = StreamingConfusionMatrix(labels=[1, 0]).update([0, 1, 2], [0, 0, 2])
    assert metric.result().tolist() == [[0, 1], [0, 1]]


def test_confusion_matrix_result_is_a_copy():
    metric = StreamingConfusionMatrix().update([0], [0])
    metric.result()[0, 0] = 99
    assert metric.result().tolist() == [[1]]


def test_empty_chunk_does_not_consume_window():
    metric = StreamingAccuracy(window=1).update([1], [1])
    metric.update([], [])
    assert metric.result() == 1.0


def test_roc_auc_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_roc_auc_single_class_is_nan():
    assert np.isnan(roc_auc([1, 1], [0.2, 0.8]))
    assert np.isnan(StreamingAUC().update([1, 1], [0.2, 0.8]).result())


def test_auc_stream_close_to_batch():
    y_true, y_score = _scores(3)
    metric = _stream(StreamingAUC(), y_true, y_score, 11)
    assert abs(metric.result() - roc_auc(y_true, y_score)) < 1e-3


def test_auc_exact_on_tied_scores():
    y_true = [0, 0, 1, 1, 0, 1]
    y_score = [0.25, 0.75, 0.25, 0.75, 0.25, 0.75]
    metric = _stream(StreamingAUC(n_bins=10), y_true, y_score, 3)
    assert metric.result() == roc_auc(y_true, y_score)


def test_auc_window_close_to_batch_on_last_chunks():
    y_true, y_score = _scores(4)
    metric = _stream(StreamingAUC(window=5), y_true, y_score, 10)
    assert abs(metric.result() - roc_auc(y_true[1000:], y_score[1000:])) < 1e-3


def test_auc_accepts_score_of_exactly_one():
    assert StreamingAUC(n_bins=4).update([0, 1], [0.0, 1.0]).result() == 1.0


def test_auc_positive_label():
    metric = StreamingAUC(positive_label="yes").update(["no", "yes"], [0.1, 0.9])
    assert metric.result() == 1.0


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_auc_rejects_scores_out_of_range(score):
    with pytest.raises(ValueError):
        StreamingAUC().update([0], [score])


def test_auc_invalid_n_bins():
    with pytest.raises(ValueError):
        StreamingAUC(n_bins=0)


def test_mse_window_forgets_huge_error_exactly():
    # Subtracting the old chunk from a running sum would lose the small error.
    metric = StreamingMSE(window=1).update([1e12], [0.0]).update([1.0], [2.0])
    assert metric.result() == 1.0
