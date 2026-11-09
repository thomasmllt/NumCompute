"""Benchmark: Python loops against the vectorised streaming code.

Three operations of the streaming framework are timed against a plain
Python reference that computes the same result one value at a time:

- split search of the decision tree (``tree._best_split``),
- tree prediction (``_Tree.apply``, one level of the tree per iteration),
- running mean and variance (``StreamingStats.update_stats``).

Each loop result is checked against the vectorised one before timing, so the
two versions are known to agree. Run from the project root with:

    python benchmark/bench_loop_vs_vectorised.py
"""
import csv
import math
import platform
import sys
from pathlib import Path

import numpy as np

from numcompute_stream.benchmarking import _memory_gb, _time
from numcompute_stream.stats import StreamingStats
from numcompute_stream.tree import DecisionTreeClassifier, _best_split

SIZES = (1_000, 10_000)
N_FEATURES = 10
N_CLASSES = 3
REPEATS = 3
SEED = 42
OUTPUT = Path(__file__).resolve().parent / "stream_loop_vs_vectorised.csv"


def make_data(n, seed=SEED):
    """Return a classification dataset whose classes overlap."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, size=n)
    X = rng.normal(size=(n, N_FEATURES)) + 0.5 * y[:, None]
    return X, y


# ---------------------------------------------------------------- references


def _loop_gini(counts):
    total = sum(counts)
    return 1.0 - sum((c / total) ** 2 for c in counts) if total else 0.0


def loop_best_split(X, y, n_classes):
    """Gini split search with one Python iteration per candidate threshold.

    Same rules as ``_best_split``: NaN values are skipped, the gain is scaled
    by the fraction of valid samples, ties go to the lowest feature then the
    lowest threshold.
    """
    n_samples, n_features = X.shape
    best_gain, best = -math.inf, None
    for feature in range(n_features):
        pairs = sorted((X[i, feature], y[i]) for i in range(n_samples)
                       if not math.isnan(X[i, feature]))
        n_valid = len(pairs)
        totals = [0] * n_classes
        for _, label in pairs:
            totals[label] += 1
        parent = _loop_gini(totals)
        left = [0] * n_classes
        for k in range(n_valid - 1):
            left[pairs[k][1]] += 1
            low, high = pairs[k][0], pairs[k + 1][0]
            if low == high:
                continue
            right = [t - l for t, l in zip(totals, left)]
            n_left = k + 1
            child = (n_left * _loop_gini(left)
                     + (n_valid - n_left) * _loop_gini(right)) / n_valid
            gain = round((parent - child) * n_valid / n_samples, 12)
            if gain > best_gain:
                threshold = low / 2 + high / 2
                best_gain = gain
                best = (feature, low if threshold >= high else threshold)
    return best


def loop_apply(tree, X):
    """Send each sample down the tree on its own."""
    leaves = []
    for row in X:
        node = 0
        while tree.feature[node] >= 0:
            value = row[tree.feature[node]]
            if math.isnan(value):
                go_left = tree.missing_left[node]
            else:
                go_left = value <= tree.threshold[node]
            node = tree.left[node] if go_left else tree.right[node]
        leaves.append(node)
    return np.array(leaves)


def loop_running_stats(chunks):
    """Welford's update, one value at a time."""
    n_features = chunks[0].shape[1]
    count, mean, m2 = 0, [0.0] * n_features, [0.0] * n_features
    for chunk in chunks:
        for row in chunk:
            count += 1
            for j, value in enumerate(row):
                delta = value - mean[j]
                mean[j] += delta / count
                m2[j] += delta * (value - mean[j])
    return np.array(mean), np.array(m2) / count


def vectorised_running_stats(chunks):
    stats = StreamingStats(max_samples=1)  # quantiles are not compared
    for chunk in chunks:
        stats.update_stats(chunk)
    return stats.mean_, stats.var_


# ---------------------------------------------------------------- benchmark


def run_benchmarks(sizes=SIZES, repeats=REPEATS):
    """Return one row per (operation, n) with both timings in milliseconds."""
    rows = []
    for n in sizes:
        X, y = make_data(n)
        tree = DecisionTreeClassifier(max_depth=8).fit(X, y)._tree
        chunks = np.array_split(X, max(1, n // 100))  # chunks of ~100 rows

        operations = [
            ("split search", lambda: loop_best_split(X, y, N_CLASSES),
             lambda: _best_split(X, y, N_CLASSES, "gini")),
            ("tree predict", lambda: loop_apply(tree, X),
             lambda: tree.apply(X)),
            ("running mean/var", lambda: loop_running_stats(chunks),
             lambda: vectorised_running_stats(chunks)),
        ]
        for name, loop_fn, vector_fn in operations:
            _check_agreement(name, loop_fn(), vector_fn())
            loop_ms = _time(loop_fn, repeats)
            vector_ms = _time(vector_fn, repeats)
            rows.append({"operation": name, "n": n, "loop_ms": loop_ms,
                         "vectorised_ms": vector_ms,
                         "speedup": loop_ms / vector_ms})
    return rows


def _check_agreement(name, loop_result, vector_result):
    if name == "split search":
        same = (loop_result[0] == vector_result[0]
                and loop_result[1] == vector_result[1])
    elif name == "tree predict":
        same = np.array_equal(loop_result, vector_result)
    else:
        same = all(np.allclose(a, b) for a, b in zip(loop_result, vector_result))
    if not same:
        raise AssertionError(f"{name}: loop and vectorised results differ")


def environment():
    memory = _memory_gb()
    return {
        "repeats": REPEATS, "seed": SEED,
        "method": "time.perf_counter, best of repeats",
        "cpu": platform.processor() or "unknown",
        "ram_gb": f"{memory:.1f}" if memory else "unknown",
        "os": f"{platform.system()} {platform.release()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
    }


def write_results(rows, path=OUTPUT):
    meta = environment()
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*rows[0], *meta])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **meta, "loop_ms": f"{row['loop_ms']:.3f}",
                             "vectorised_ms": f"{row['vectorised_ms']:.3f}",
                             "speedup": f"{row['speedup']:.1f}"})
    return path


def main():
    rows = run_benchmarks()
    path = write_results(rows)
    meta = environment()
    print(f"Environment: {meta['cpu']}, {meta['ram_gb']} GB, {meta['os']}, "
          f"Python {meta['python']}, NumPy {meta['numpy']}")
    print(f"Method: {meta['method']} ({REPEATS}), seed={SEED}, "
          f"{N_FEATURES} features, {N_CLASSES} classes\n")
    print("| Operation | n | Python loop (ms) | Vectorised (ms) | Speed-up (x) |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['operation']} | {row['n']} | {row['loop_ms']:.2f} | "
              f"{row['vectorised_ms']:.2f} | {row['speedup']:.0f} |")
    print(f"\nAll loop results matched the vectorised ones. Saved to {path}")


if __name__ == "__main__":
    main()
