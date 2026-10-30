# NumCompute

A lightweight scientific computing toolkit built from scratch with **plain Python and NumPy only** (no pandas, scikit-learn or PyTorch). It provides vectorised, reusable components for data loading, preprocessing, sorting and ranking, statistics, evaluation metrics, numerical gradients, pipelines and benchmarking.

## Installation

Requires Python 3.10+ and NumPy 2.0+.

```bash
git clone https://github.com/HwistleK/NumCompute.git
cd NumCompute
pip install -e .            # installs numcompute and NumPy
pip install -e ".[dev]"     # also installs pytest, to run the tests
```

## Quick start

```python
import numpy as np
from numcompute_stream import (load_csv, StandardScaler, SimpleImputer, Pipeline,
                        topk, rank, percentile, mean, accuracy, grad)

X = load_csv("data.csv", delimiter=",", skiprows=1)    # missing values -> NaN

pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
X_tr = pipe.fit_transform(X)

scores = np.array([72, 95, 88, 95, 60, np.nan, 81])
topk(scores, 3)                       # (array([95., 95., 88.]), array([1, 3, 2]))
rank(scores, method="average")        # ties share 5.5, NaN stays unranked
percentile(scores, [25, 50, 75])      # array([74.25, 84.5 , 93.25])
mean(scores)                          # 81.83..., NaN ignored

accuracy([1, 0, 1, 1], [1, 1, 1, 0])  # 0.5
grad(lambda x: np.sum(x ** 2), np.array([2.0, 3.0]))   # array([4., 6.])
```

A full end-to-end walkthrough is in [`demo/quickstart.ipynb`](demo/quickstart.ipynb).

## API overview

| Module | Main API |
| --- | --- |
| `io` | `load_csv(filepath, delimiter, skiprows, missing_values, filling_values, dtype)` |
| `preprocessing` | `StandardScaler`, `MinMaxScaler(feature_range)`, `OneHotEncoder`, `SimpleImputer(strategy, fill_value)` |
| `pipeline` | `Transformer`, `Estimator`, `Pipeline(steps)` with `fit`, `transform`, `fit_transform`, `predict` |
| `stats` | `mean`, `median`, `std`, `minimum`, `maximum`, `histogram`, `quantiles` (NaN ignored, `axis` supported) |
| `sort_search` | `stable_sort`, `stable_argsort`, `multikey_sort`, `topk`, `quickselect`, `binary_search` |
| `rank` | `rank(method='average' \| 'dense' \| 'ordinal')`, `percentile(interpolation='linear' \| 'lower' \| 'higher' \| 'midpoint')` |
| `utils` | `euclidean_distance`, `sigmoid`, `softmax`, `logsumexp`, `batch_generator` |
| `metrics` | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse` |
| `optim` | `grad(f, x, h, method='central' \| 'forward')`, `jacobian(F, x, h, method)` |
| `benchmarking` | `run_benchmarks`, `write_results`, `main` |

Conventions used across the package:

- **Estimator API:** `fit(X) -> self`, `transform(X) -> X_out`, `fit_transform(X) -> X_out`
- **Errors:** `TypeError` for wrong input types, `ValueError` for wrong values or shapes
- **NaN:** treated as missing (ignored in statistics and percentiles, left unranked by `rank`)
- **Docstrings:** NumPy style, including shapes, exceptions and time/space complexity

## Running the tests

```bash
python -m pytest -v
```

## Performance

Python loops vs. vectorised NumPy implementations (n = 10,000, best of 5 runs, seed 42).
Environment: Apple Silicon (arm), 16 GB RAM, macOS (Darwin 25.1.0), Python 3.13, NumPy 2.5.0.

| Operation | n | Python loop (ms) | Vectorised (ms) | Speed-up (×) |
| --- | ---: | ---: | ---: | ---: |
| StandardScaler | 10,000 | 3.682 | 0.015 | 244.8 |
| top-k | 10,000 | 1.867 | 0.211 | 8.9 |
| rank | 10,000 | 3.645 | 0.637 | 5.7 |
| histogram | 10,000 | 1.291 | 0.095 | 13.6 |

Reproduce (writes CSVs to `benchmark/` and prints the table):

```bash
python -m numcompute.benchmarking
```

## Project structure

```
NumCompute/
├── numcompute/        # the package (10 modules)
├── tests/             # pytest unit tests, incl. randomised tests
├── demo/              # quickstart.ipynb
├── benchmark/         # loop vs. vectorised results (CSV)
├── README.md
└── pyproject.toml
```
