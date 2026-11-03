"""Reusable matplotlib plots for streaming metrics and predictions.

Every function takes the same display options:

- ``ax``: draw on an existing Axes, for example to build a grid of plots. The
  caller then owns the figure, so it is neither shown nor closed here.
- ``save_path``: write the figure to this file, creating missing folders.
- ``show``: display the figure, inline in a notebook or in a window in a
  script. With ``show=False`` the figure is closed after saving.

Each function returns the Axes it drew on. NaN values, such as the chunks
scored before the first tree exists, appear as gaps in the lines.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .metrics import _drop_missing
from .preprocessing import to_numeric

__all__ = ["plot_metric_over_time", "compare_models",
           "plot_predictions_vs_ground_truth"]

_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_SERIES_1 = "#2a78d6"
_SERIES_2 = "#eb6834"
_FIGURE_SIZE = (7, 4)


def _as_series(values, name):
    """Return a metric history as a 1D float array with at least one value."""
    arr = to_numeric(values, name).astype(float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D")
    if not np.isfinite(arr).any():
        raise ValueError(f"{name} has no finite value to plot")
    return arr


def _get_axes(ax):
    """Return (axes, whether a new figure was created for them)."""
    if ax is not None:
        return ax, False
    fig, ax = plt.subplots(figsize=_FIGURE_SIZE, layout="constrained")
    fig.set_facecolor(_SURFACE)
    return ax, True


def _style_axes(ax, title, xlabel, ylabel):
    """Apply the shared style: light grid, left title, only two spines."""
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, loc="left", color=_TEXT, fontsize=12)
    ax.set_xlabel(xlabel, color=_TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=_TEXT_SECONDARY)
    ax.tick_params(colors=_TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _add_legend(ax):
    """Put the legend above the plot, where it cannot hide any data."""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2,
              frameon=False, labelcolor=_TEXT_SECONDARY, borderaxespad=0.0)


def _finish(ax, owns_figure, save_path, show):
    """Save, then show or close the figure according to the options."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=150, facecolor=ax.figure.get_facecolor())
    if owns_figure:
        if show:
            plt.show()
        else:
            plt.close(ax.figure)
    return ax


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric",
                          xlabel="Chunk", ax=None, save_path=None, show=True):
    """Plot a metric across the chunks of a stream.

    Parameters
    ----------
    metric_values : array_like, shape (n_chunks,)
        Metric value of each chunk, for example from
        ``StreamTrainer.history("cumulative_accuracy")``.
    title, ylabel, xlabel : str
        Texts of the plot.
    ax : matplotlib Axes or None, default=None
        Axes to draw on. See the module documentation for this and the
        following options.
    save_path : str, Path or None, default=None
        File to save the figure to.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib Axes

    Raises
    ------
    TypeError
        If the values are not numeric.
    ValueError
        If the values are not 1D or none of them is finite.
    """
    values = _as_series(metric_values, "metric_values")
    ax, owns_figure = _get_axes(ax)
    _style_axes(ax, title, xlabel, ylabel)
    ax.plot(values, color=_SERIES_1, linewidth=1.5, marker="o", markersize=3)
    return _finish(ax, owns_figure, save_path, show)


def compare_models(metric1, metric2, labels, title="Model comparison",
                   ylabel="Metric", xlabel="Chunk", ax=None, save_path=None,
                   show=True):
    """Compare two models on the same streaming metric.

    The second model is dashed, so the plot stays readable in black and white.

    Parameters
    ----------
    metric1, metric2 : array_like, shape (n_chunks,)
        Metric history of each model. Their lengths may differ.
    labels : sequence of two str
        Names of the two models, for the legend.
    title, ylabel, xlabel : str
        Texts of the plot.
    ax, save_path, show
        Display options, see :func:`plot_metric_over_time`.

    Returns
    -------
    matplotlib Axes

    Raises
    ------
    TypeError
        If the values are not numeric.
    ValueError
        If a history is invalid or ``labels`` does not hold two names.
    """
    if len(labels) != 2:
        raise ValueError(f"labels must hold 2 names, got {len(labels)}")
    first = _as_series(metric1, "metric1")
    second = _as_series(metric2, "metric2")
    ax, owns_figure = _get_axes(ax)
    _style_axes(ax, title, xlabel, ylabel)
    ax.plot(first, color=_SERIES_1, linewidth=1.5, marker="o", markersize=3,
            label=labels[0])
    ax.plot(second, color=_SERIES_2, linewidth=1.5, marker="o", markersize=3,
            linestyle="--", label=labels[1])
    _add_legend(ax)
    return _finish(ax, owns_figure, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs ground truth",
                                     xlabel="Sample", ylabel="Class", ax=None,
                                     save_path=None, show=True):
    """Compare the predicted and true class of each sample of a chunk.

    Each sample is a dot on the row of its true class. A wrong prediction adds
    a cross on the row of the predicted class, so errors differ by their shape
    as well as their color.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels. Pairs containing a NaN are ignored.
    title, xlabel, ylabel : str
        Texts of the plot.
    ax, save_path, show
        Display options, see :func:`plot_metric_over_time`.

    Returns
    -------
    matplotlib Axes

    Raises
    ------
    ValueError
        If the inputs have different lengths or no sample is left.
    """
    true, pred = _drop_missing(y_true, y_pred)
    if true.size == 0:
        raise ValueError("y_true and y_pred have no sample to plot")
    classes = np.unique(np.concatenate([true, pred]))
    true_rows = np.searchsorted(classes, true)
    pred_rows = np.searchsorted(classes, pred)
    samples = np.arange(true.size)
    wrong = true_rows != pred_rows

    ax, owns_figure = _get_axes(ax)
    _style_axes(ax, title, xlabel, ylabel)
    ax.scatter(samples, true_rows, s=24, color=_SERIES_1, linewidths=0,
               label="Ground truth")
    ax.scatter(samples[wrong], pred_rows[wrong], s=40, color=_SERIES_2,
               marker="x", linewidths=1.5,
               label=f"Wrong prediction ({wrong.sum()} of {true.size})")
    ax.set_yticks(np.arange(classes.size), labels=[str(c) for c in classes])
    ax.set_ylim(-0.5, classes.size - 0.5)
    _add_legend(ax)
    return _finish(ax, owns_figure, save_path, show)
