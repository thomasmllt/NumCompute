"""Benchmark: effect of ``buffer_size`` on a streaming decision tree.

A synthetic stream of 20,000 samples arrives in chunks of 100. Halfway
through, the rule that decides the labels changes (an abrupt concept drift),
so the model must forget what it learnt. Each run follows the prequential
protocol of StreamTrainer: every chunk is scored before the model trains on
it.

For each buffer size the script records the accuracy, the number of
retrains, the time they take and the peak memory, then plots the per-chunk
accuracy of the smallest and largest buffers. Run from the project root
with:

    python benchmark/bench_buffer_size.py
"""
import csv
from pathlib import Path

import numpy as np

from numcompute_stream.stream import StreamTrainer, iter_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import compare_models

BUFFER_SIZES = (100, 250, 500, 1_000, 2_000, 5_000)
N_SAMPLES = 20_000
CHUNK_SIZE = 100
MAX_DEPTH = 6
SEED = 42
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "buffer_size_results.csv"
FIGURE = HERE / "buffer_size_accuracy.png"


def make_stream(n=N_SAMPLES, seed=SEED, noise=0.1):
    """Return (X, y) with an abrupt drift at the middle of the stream.

    Before the drift the label depends on features 0 and 1, after it on
    features 2 and 3. A fraction ``noise`` of the labels is flipped, so no
    model can exceed about 90% accuracy.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    half = n // 2
    y = np.empty(n, dtype=int)
    y[:half] = X[:half, 0] + X[:half, 1] > 0
    y[half:] = X[half:, 2] - X[half:, 3] > 0.5
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y


def run_one(buffer_size, X, y):
    """Stream the data through one tree and return the trainer."""
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, buffer_size=buffer_size,
                                  seed=SEED)
    return StreamTrainer(tree).run(iter_chunks(X, y, CHUNK_SIZE))


def summarise(buffer_size, trainer):
    """Return the figures of one run as a table row."""
    summary = trainer.summary()
    accuracy = trainer.history("accuracy")
    retrain_times = trainer.history("fit_time")[trainer.history("retrained") == 1]
    after_drift = accuracy[len(accuracy) // 2:]
    return {
        "buffer_size": buffer_size,
        "cumulative_accuracy": summary["cumulative_accuracy"],
        "accuracy_after_drift": np.nanmean(after_drift),
        "unscored_chunks": int((trainer.history("scored") == 0).sum()),
        "n_retrains": summary["n_retrains"],
        "mean_retrain_ms": 1000 * retrain_times.mean(),
        "total_fit_ms": 1000 * summary["fit_time"],
        "peak_memory_kb": summary["peak_memory_bytes"] / 1024,
    }


def run_benchmarks(buffer_sizes=BUFFER_SIZES):
    """Return the table rows and the trainer of each buffer size."""
    X, y = make_stream()
    trainers = {b: run_one(b, X, y) for b in buffer_sizes}
    rows = [summarise(b, trainer) for b, trainer in trainers.items()]
    return rows, trainers


def write_results(rows, path=OUTPUT):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.4f}" if isinstance(value, float)
                             else value for key, value in row.items()})
    return path


def main():
    rows, trainers = run_benchmarks()
    path = write_results(rows)
    small, large = min(trainers), max(trainers)
    compare_models(trainers[small].history("accuracy"),
                   trainers[large].history("accuracy"),
                   (f"buffer_size={small}", f"buffer_size={large}"),
                   title="Accuracy per chunk", ylabel="Accuracy",
                   xlabel=f"Chunk (drift at chunk {N_SAMPLES // CHUNK_SIZE // 2})",
                   save_path=FIGURE, show=False)

    print(f"Stream: {N_SAMPLES} samples, chunks of {CHUNK_SIZE}, drift at the "
          f"middle, max_depth={MAX_DEPTH}, seed={SEED}\n")
    print("| buffer_size | Cumulative acc. | Acc. after drift | Unscored chunks "
          "| Retrains | Mean retrain (ms) | Total fit (ms) | Peak memory (KB) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['buffer_size']} | {row['cumulative_accuracy']:.3f} | "
              f"{row['accuracy_after_drift']:.3f} | {row['unscored_chunks']} | "
              f"{row['n_retrains']} | {row['mean_retrain_ms']:.2f} | "
              f"{row['total_fit_ms']:.1f} | {row['peak_memory_kb']:.0f} |")
    print(f"\nSaved the table to {path}\nSaved the figure to {FIGURE}")


if __name__ == "__main__":
    main()
