"""Unit tests for StreamTrainer: prequential scoring, logs and edge cases."""
import numpy as np
import pytest

from numcompute_stream.metrics import StreamingAccuracy, StreamingAUC, StreamingF1
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer, iter_chunks
from numcompute_stream.tree import DecisionTreeClassifier


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return rng.normal(size=(n, 3)) * 0.5 + y[:, None], y


def test_iter_chunks_keeps_order_and_rejects_mismatch():
    X, y = np.arange(10).reshape(5, 2), np.arange(5)
    chunks = list(iter_chunks(X, y, chunk_size=2))
    assert [len(y_chunk) for _, y_chunk in chunks] == [2, 2, 1]
    assert np.array_equal(np.concatenate([y_chunk for _, y_chunk in chunks]), y)
    with pytest.raises(ValueError):
        list(iter_chunks(X, y[:2], chunk_size=2))


def test_run_is_prequential():
    X, y = _data()
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=3, buffer_size=100))
    trainer.run(iter_chunks(X, y, chunk_size=50))

    # Replay the stream by hand: score with the tree of the previous chunks.
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    correct = total = 0
    for i, (X_chunk, y_chunk) in enumerate(iter_chunks(X, y, chunk_size=50)):
        if tree.is_fitted_:
            hits = tree.predict(X_chunk) == y_chunk
            correct, total = correct + hits.sum(), total + hits.size
            assert np.isclose(trainer.history("accuracy")[i], hits.mean())
            assert np.isclose(trainer.history("cumulative_accuracy")[i],
                              correct / total)
        tree.partial_fit(X_chunk, y_chunk)


def test_chunks_before_first_tree_are_not_scored():
    X, y = _data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    trainer.run(iter_chunks(X, y, chunk_size=50))
    assert trainer.history("scored").tolist()[:3] == [0.0, 0.0, 1.0]
    assert np.isnan(trainer.history("accuracy")[:2]).all()


def test_fit_log_tracks_retrains_buffer_and_memory():
    X, y = _data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    trainer.run(iter_chunks(X, y, chunk_size=50))
    assert trainer.history("retrained").tolist() == [0, 1, 0, 1, 0, 1]
    assert trainer.history("n_buffered").tolist() == [50, 0, 50, 0, 50, 0]
    memory = trainer.history("memory_bytes")
    # The first chunk is buffered: its 50 * 3 floats must be counted.
    assert memory[0] > 50 * 3 * 8
    assert memory[2] > memory[1]


def test_summary():
    X, y = _data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    assert np.isnan(trainer.summary()["cumulative_accuracy"])
    trainer.run(iter_chunks(X, y, chunk_size=50))
    summary = trainer.summary()
    assert summary["n_chunks"] == 6 and summary["n_samples"] == 300
    assert summary["n_retrains"] == 3
    assert summary["cumulative_accuracy"] > 0.8


def test_works_with_pipeline_and_extra_metrics():
    X, y = _data()
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', DecisionTreeClassifier(max_depth=3, buffer_size=100)),
    ])
    trainer = StreamTrainer(pipe, metrics={
        "rolling": StreamingAccuracy(window=2), "f1": StreamingF1()})
    trainer.run(iter_chunks(X, y, chunk_size=50))
    assert trainer.history("retrained").sum() == 3
    assert trainer.history("cumulative_f1")[-1] > 0.8
    # The window of the rolling metric must apply to its cumulative value only.
    assert np.isclose(trainer.history("rolling")[-1],
                      trainer.history("accuracy")[-1])


def test_empty_chunk_is_logged_without_score():
    X, y = _data(n=100)
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    trainer.fit_chunk(X, y)
    record = trainer.score_chunk(X[:0], y[:0])
    assert not record["scored"] and np.isnan(record["accuracy"])


def test_invalid_arguments():
    with pytest.raises(TypeError):
        StreamTrainer(StandardScaler())
    with pytest.raises(TypeError):
        StreamTrainer(DecisionTreeClassifier(), metrics={"m": len})
    with pytest.raises(TypeError):
        StreamTrainer(DecisionTreeClassifier(), metrics={"auc": StreamingAUC()})
    with pytest.raises(KeyError):
        StreamTrainer(DecisionTreeClassifier()).history("accuracy")


def test_verbose_prints_records(capsys):
    X, y = _data(n=100)
    StreamTrainer(DecisionTreeClassifier(buffer_size=50), verbose=True) \
        .process_chunk(X, y)
    assert "retrained=True" in capsys.readouterr().out
