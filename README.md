# NumCompute-Stream

A small machine learning framework for **streaming data**, built with NumPy and
matplotlib only. Data arrives in chunks, and every component can learn from one
chunk at a time with `partial_fit`.

It extends NumCompute with:

- a decision tree classifier that buffers chunks and retrains when the buffer is full,
- streaming preprocessing, statistics and metrics,
- a training loop that logs metrics, time and memory for each chunk,
- plots of the metrics over time.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/thomasmllt/NumCompute.git
cd NumCompute
git checkout assignment2.2-a3227265
pip install -e ".[dev]"     # numpy, matplotlib, plus pytest and ipykernel
```

## Quick start

```python
from numcompute_stream import (load_csv, Pipeline, StandardScaler,
                               DecisionTreeClassifier, StreamTrainer, iter_chunks,
                               plot_metric_over_time)

data = load_csv("demo/banknote.csv", skiprows=1)
X, y = data[:, :-1], data[:, -1]

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=200)),
])

# Each chunk is used to test the model, then to train it (partial_fit).
trainer = StreamTrainer(pipe).run(iter_chunks(X, y, chunk_size=50))

print(trainer.summary())
plot_metric_over_time(trainer.history("cumulative_accuracy"), ylabel="Accuracy")
```

## Main modules

| Module | What it adds for streaming |
| --- | --- |
| `tree.py` | `DecisionTreeClassifier` with `fit`, `partial_fit` (buffered), `flush`, `predict`, `predict_proba` |
| `stream.py` | `StreamTrainer` (`fit_chunk`, `score_chunk`, `run`, `history`, `summary`) and `iter_chunks` |
| `preprocessing.py` | `partial_fit` for `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer` |
| `stats.py` | `StreamingStats.update_stats` (mean, variance, min, max, quantiles), `StreamingHistogram` |
| `metrics.py` | Streaming accuracy, precision, recall, F1, AUC, MSE with `update`, `result`, `reset` and an optional rolling `window` |
| `pipeline.py` | `Pipeline.partial_fit` |
| `visualise.py` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth` |

The NumCompute modules (`io`, `sort_search`, `rank`, `utils`, `optim`) are still included.

## Demo

[`demo/stream_demo.ipynb`](demo/stream_demo.ipynb) loads a CSV file, streams it in
chunks through a pipeline and plots the results. Open it in Jupyter or VS Code and
run all cells.

The dataset is *Banknote Authentication* by V. Lohweg, from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
(CC BY 4.0). A header row was added to the file.

## Tests

```bash
python -m pytest
```

## Benchmarks

Run from the project root:

```bash
python benchmark/bench_loop_vs_vectorised.py   # Python loops vs NumPy
python benchmark/bench_buffer_size.py          # effect of buffer_size
```

Each script prints a table and saves its results in `benchmark/`.
The loop vs NumPy results for NumCompute itself come from
`python -m numcompute_stream.benchmarking`.

## Project structure

```
NumCompute/
├── numcompute_stream/   # the package
├── tests/               # unit tests
├── demo/                # stream_demo.ipynb, quickstart.ipynb, data files
├── benchmark/           # benchmark scripts and results
├── README.md
└── pyproject.toml
```
