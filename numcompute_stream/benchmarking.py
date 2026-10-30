"""Reproducible comparisons between Python loops and NumPy operations."""

import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .preprocessing import StandardScaler
from .rank import rank
from .sort_search import topk

__all__ = ["run_benchmarks", "write_results", "main"]


def _memory_gb():
    """Return the installed memory when available."""
    try:
        import os
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _time(function, repeats):
    """Return the best time in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def _loop_scaler(values):
    means = [sum(column) / len(column) for column in values.T]
    variances = [
        sum((value - mean) ** 2 for value in column) / len(column)
        for column, mean in zip(values.T, means)
    ]
    return np.array([
        [(value - means[column]) / (variances[column] ** 0.5)
         for column, value in enumerate(row)]
        for row in values
    ])


def _loop_topk(values, k):
    return np.array(sorted(values, reverse=True)[:k])


def _loop_rank(values):
    sorted_values = sorted(values)
    ranks = {}
    for position, value in enumerate(sorted_values, start=1):
        if value not in ranks:
            ranks[value] = position
    return np.array([ranks[value] for value in values])


def _loop_histogram(values, bins):
    low, high = float(values.min()), float(values.max())
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    return np.array(counts)


def run_benchmarks(n=10000, repeats=5, seed=42):
    """Run benchmarks and return rows for the performance table."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n)
    matrix = values.reshape(n // 10, 10)
    k = n // 10
    scaler = StandardScaler()
    scaler.fit(matrix)

    operations = [
        (
            "StandardScaler",
            n,
            lambda: _loop_scaler(matrix),
            lambda: scaler.transform(matrix),
        ),
        (
            "top-k",
            n,
            lambda: _loop_topk(values, k),
            lambda: topk(values, k, return_indices=False),
        ),
        (
            "rank",
            n,
            lambda: _loop_rank(values),
            lambda: rank(values, method="dense"),
        ),
        (
            "histogram",
            n,
            lambda: _loop_histogram(values, 20),
            lambda: np.histogram(values, bins=20),
        ),
    ]

    rows = []
    for name, size, loop_function, vector_function in operations:
        loop_ms = _time(loop_function, repeats)
        vector_ms = _time(vector_function, repeats)
        rows.append((name, size, loop_ms, vector_ms, loop_ms / vector_ms))
    return rows


def write_results(rows, output_dir, n, repeats, seed):
    """Write loop and vectorised timings to separate CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory = _memory_gb()
    metadata = {
        "n": n,
        "repeats": repeats,
        "seed": seed,
        "method": "time.perf_counter, best of repeats",
        "cpu": platform.processor() or "unknown",
        "ram_gb": f"{memory:.1f}" if memory else "unknown",
        "os": f"{platform.system()} {platform.release()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
    }
    fields = [
        "operation", "n", "time_ms", "repeats", "seed", "method",
        "cpu", "ram_gb", "os", "python", "numpy",
    ]
    loop_path = output_dir / "loop_results.csv"
    vectorised_path = output_dir / "vectorised_results.csv"

    with loop_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, size, loop_ms, _, _ in rows:
            writer.writerow({
                **metadata, "operation": name, "n": size, "time_ms": f"{loop_ms:.6f}",
            })

    with vectorised_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, size, _, vector_ms, _ in rows:
            writer.writerow({
                **metadata, "operation": name, "n": size, "time_ms": f"{vector_ms:.6f}",
            })
    return loop_path, vectorised_path


def main(output_dir=None):
    n, repeats, seed = 10000, 5, 42
    rows = run_benchmarks(n=n, repeats=repeats, seed=seed)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "benchmark"
    loop_path, vectorised_path = write_results(
        rows, output_dir, n, repeats, seed
    )
    speedups = [row[4] for row in rows]
    memory = _memory_gb()
    memory_text = f"{memory:.1f} GB" if memory else "unknown"

    print(
        f"Vectorised versions were {min(speedups):.1f}–{max(speedups):.1f}x "
        f"faster than Python loops at n = {n}."
    )
    print(
        f"Environment: {platform.processor() or 'unknown CPU'}, {memory_text}, "
        f"{platform.system()} {platform.release()}, Python {sys.version_info.major}."
        f"{sys.version_info.minor}, NumPy {np.__version__}."
    )
    print(
        f"Method: time.perf_counter, best of {repeats} repeats, "
        f"input size n={n}, random seed={seed}."
    )
    print("Run from the project root with: python -m numcompute.benchmarking")
    print(f"Saved loop results to: {loop_path}")
    print(f"Saved vectorised results to: {vectorised_path}")
    print()
    print("| Operation | n | Python loop (ms) | Vectorised (ms) | Speed-up (x) |")
    print("|---|---:|---:|---:|---:|")
    for name, size, loop_ms, vector_ms, speedup in rows:
        print(f"| {name} | {size} | {loop_ms:.3f} | {vector_ms:.3f} | {speedup:.1f} |")
    print()
    fastest = rows[int(np.argmax(speedups))][0]
    print("Discussion")
    print(
        f"Vectorisation helped most for {fastest}, because NumPy performs the "
        "main array operations in compiled code instead of Python loops."
    )
    print(
        "For small inputs the difference can be smaller because function-call "
        "and array-creation overhead becomes more important."
    )


if __name__ == "__main__":
    main()
