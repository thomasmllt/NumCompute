"""Training loop for streaming data, with per-chunk logging.

StreamTrainer feeds chunks to a model or pipeline, keeps cumulative metrics
and records what happened on each chunk. Its logs are plain lists of
dictionaries, and history() returns one column of them as an array, ready to
be plotted.
"""
import copy
import sys
import time
from collections import deque

import numpy as np

from .metrics import StreamingAccuracy, StreamingAUC, StreamingMetric
from .utils import batch_generator

__all__ = ["StreamTrainer", "iter_chunks"]


def iter_chunks(X, y, chunk_size):
    """Split a dataset into consecutive chunks, keeping the order of the rows.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input data.
    y : array_like, shape (n_samples,)
        Target values.
    chunk_size : int
        Number of samples per chunk. The last chunk may be smaller.

    Yields
    ------
    X_chunk : ndarray, shape (chunk_size, n_features)
    y_chunk : ndarray, shape (chunk_size,)

    Raises
    ------
    ValueError
        If X and y have different lengths or chunk_size is not positive.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")
    for rows in batch_generator(np.arange(y.shape[0]), chunk_size, shuffle=False):
        yield X[rows], y[rows]


def _deep_sizeof(obj, seen=None):
    """Return the size in bytes of an object and of everything it refers to."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        children = [*obj.keys(), *obj.values()]
    elif isinstance(obj, (list, tuple, set, deque)):
        children = obj
    elif hasattr(obj, "__dict__"):
        children = [vars(obj)]
    else:
        children = []
    return sys.getsizeof(obj) + sum(_deep_sizeof(c, seen) for c in children)


class StreamTrainer:
    """Train and evaluate a model chunk by chunk, and log the progress.

    ``process_chunk`` follows the prequential (test-then-train) protocol: a
    chunk is first used to score the model, then to train it, so that every
    score is measured on data the model has not seen yet. ``fit_chunk`` and
    ``score_chunk`` give access to each half separately.

    Parameters
    ----------
    model : object
        Pipeline or estimator implementing ``partial_fit`` and ``predict``.
    metrics : dict of {str: StreamingMetric} or None, default=None
        Additional metrics comparing true and predicted labels, with scalar
        results. Accuracy is always included under the name ``"accuracy"``.
    verbose : bool, default=False
        Whether to print each log record.

    Attributes
    ----------
    fit_log_ : list of dict
        One record per ``fit_chunk``: ``chunk``, ``n_samples``, ``fit_time``,
        ``retrained``, ``n_retrains``, ``n_buffered``, ``memory_bytes``.
    score_log_ : list of dict
        One record per ``score_chunk``: ``chunk``, ``n_samples``, ``scored``,
        ``score_time``, and for each metric its value on the chunk (``name``)
        and since the start of the stream (``cumulative_name``). Metrics are
        NaN and ``scored`` is False if the model cannot predict yet or the
        chunk is empty.
    n_chunks_ : int
        Number of chunks trained on. The ``chunk`` field of a record is the
        value of this counter when the record started.

    Complexity
    ----------
    Time and space per chunk: those of the model and metrics, plus
    O(number of objects held by them) to measure the memory footprint.

    Examples
    --------
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> X = np.arange(100.0).reshape(-1, 1)
    >>> y = (X[:, 0] >= 50).astype(int)
    >>> rows = np.random.default_rng(0).permutation(100)
    >>> trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=20))
    >>> _ = trainer.run(iter_chunks(X[rows], y[rows], chunk_size=20))
    >>> trainer.history("scored")[:2].tolist()
    [0.0, 1.0]
    >>> trainer.summary()["n_retrains"]
    5
    """

    def __init__(self, model, metrics=None, verbose=False):
        if not (hasattr(model, "partial_fit") and hasattr(model, "predict")):
            raise TypeError("model must implement partial_fit and predict")
        all_metrics = {"accuracy": StreamingAccuracy(), **(metrics or {})}
        for name, metric in all_metrics.items():
            if not isinstance(metric, StreamingMetric):
                raise TypeError(f"Metric '{name}' must be a StreamingMetric")
            if isinstance(metric, StreamingAUC):
                raise TypeError(f"Metric '{name}' needs scores, but "
                                "StreamTrainer compares predicted labels")
        self.model = model
        self.metrics = all_metrics
        self.verbose = verbose
        # A reset copy measures a metric on one chunk without touching the
        # cumulative one.
        self._chunk_metrics = {name: copy.deepcopy(metric).reset()
                               for name, metric in all_metrics.items()}
        self.n_chunks_ = 0
        self.fit_log_ = []
        self.score_log_ = []

    @property
    def _final_step(self):
        """Return the object that makes the predictions."""
        steps = getattr(self.model, "steps", None)
        return steps[-1][1] if steps else self.model

    def fit_chunk(self, X, y):
        """Train the model on a chunk and log the update.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data of the chunk.
        y : array_like, shape (n_samples,)
            Target values of the chunk.

        Returns
        -------
        record : dict
            The new entry of ``fit_log_``.
        """
        final = self._final_step
        retrains_before = getattr(final, "n_retrains_", 0)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        fit_time = time.perf_counter() - start
        n_retrains = getattr(final, "n_retrains_", 0)

        record = {
            "chunk": self.n_chunks_,
            "n_samples": np.shape(y)[0],
            "fit_time": fit_time,
            "retrained": n_retrains > retrains_before,
            "n_retrains": n_retrains,
            "n_buffered": getattr(final, "n_buffered_", 0),
            "memory_bytes": _deep_sizeof((self.model, self.metrics)),
        }
        self.n_chunks_ += 1
        self._log(self.fit_log_, record)
        return record

    def score_chunk(self, X, y):
        """Evaluate the model on a chunk and update the metrics.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data of the chunk.
        y : array_like, shape (n_samples,)
            True labels of the chunk.

        Returns
        -------
        record : dict
            The new entry of ``score_log_``.
        """
        n_samples = np.shape(y)[0]
        can_predict = getattr(self._final_step, "is_fitted_", True)
        scored = bool(can_predict and n_samples > 0)
        record = {"chunk": self.n_chunks_, "n_samples": n_samples,
                  "scored": scored, "score_time": np.nan}
        if scored:
            start = time.perf_counter()
            y_pred = self.model.predict(X)
            record["score_time"] = time.perf_counter() - start

        for name, metric in self.metrics.items():
            record[name] = record[f"cumulative_{name}"] = np.nan
            if scored:
                chunk_metric = self._chunk_metrics[name].reset()
                record[name] = chunk_metric.update(y, y_pred).result()
                record[f"cumulative_{name}"] = metric.update(y, y_pred).result()
        self._log(self.score_log_, record)
        return record

    def process_chunk(self, X, y):
        """Score the model on a chunk, then train it on the same chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data of the chunk.
        y : array_like, shape (n_samples,)
            Target values of the chunk.

        Returns
        -------
        self : StreamTrainer
        """
        self.score_chunk(X, y)
        self.fit_chunk(X, y)
        return self

    def run(self, chunks):
        """Apply :meth:`process_chunk` to a whole stream.

        Parameters
        ----------
        chunks : iterable of (X_chunk, y_chunk)
            The stream, for example from :func:`iter_chunks`.

        Returns
        -------
        self : StreamTrainer
        """
        for X_chunk, y_chunk in chunks:
            self.process_chunk(X_chunk, y_chunk)
        return self

    def history(self, key):
        """Return one field of the logs, with one value per record.

        Parameters
        ----------
        key : str
            Field of ``score_log_`` or ``fit_log_``, for example
            ``"cumulative_accuracy"``. ``score_log_`` is searched first.

        Returns
        -------
        ndarray of float, shape (n_records,)

        Raises
        ------
        KeyError
            If no log contains this field.
        """
        for log in (self.score_log_, self.fit_log_):
            if log and key in log[0]:
                return np.array([record[key] for record in log], dtype=float)
        raise KeyError(f"No log contains the field {key!r}")

    def summary(self):
        """Return the main figures of the stream so far.

        Returns
        -------
        dict
            ``n_chunks``, ``n_samples``, ``n_retrains``, ``fit_time`` (total),
            ``peak_memory_bytes`` and ``cumulative_accuracy``.
        """
        return {
            "n_chunks": self.n_chunks_,
            "n_samples": sum(record["n_samples"] for record in self.fit_log_),
            "n_retrains": self.fit_log_[-1]["n_retrains"] if self.fit_log_ else 0,
            "fit_time": sum(record["fit_time"] for record in self.fit_log_),
            "peak_memory_bytes": max(
                (record["memory_bytes"] for record in self.fit_log_), default=0),
            "cumulative_accuracy": (self.score_log_[-1]["cumulative_accuracy"]
                                    if self.score_log_ else np.nan),
        }

    def _log(self, log, record):
        log.append(record)
        if self.verbose:
            print(" ".join(f"{key}={value:.4g}" if isinstance(value, float)
                           else f"{key}={value}"
                           for key, value in record.items()))
