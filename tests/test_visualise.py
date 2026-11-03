"""Unit tests for the plotting functions, without opening any window."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_metric_keeps_nan_gaps_and_applies_texts():
    values = [np.nan, np.nan, 0.8, 0.9]
    ax = plot_metric_over_time(values, title="Accuracy", ylabel="Score", show=False)
    assert ax.get_title(loc="left") == "Accuracy" and ax.get_ylabel() == "Score"
    assert np.array_equal(ax.lines[0].get_ydata(), values, equal_nan=True)


def test_save_path_creates_folders(tmp_path):
    path = tmp_path / "plots" / "metric.png"
    plot_metric_over_time([0.5, 0.6], save_path=path, show=False)
    assert path.stat().st_size > 0


def test_show_option_controls_the_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    plt.close("all")
    plot_metric_over_time([0.5, 0.6], show=True)
    assert shown == [True]
    plt.close("all")
    plot_metric_over_time([0.5, 0.6], show=False)
    assert plt.get_fignums() == []


def test_given_axes_stay_open_for_subplots():
    fig, axes = plt.subplots(1, 2)
    assert plot_metric_over_time([0.5, 0.6], ax=axes[0]) is axes[0]
    compare_models([0.5], [0.6], ("a", "b"), ax=axes[1])
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_compare_models_labels_each_series():
    ax = compare_models([0.5, 0.6], [0.7], ("small", "large"), show=False)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["small", "large"]
    assert len(ax.lines[1].get_ydata()) == 1


def test_predictions_mark_errors_and_support_strings():
    ax = plot_predictions_vs_ground_truth(["a", "b", "b", "c"], ["a", "c", "b", "c"],
                                          show=False)
    truth, errors = ax.collections
    assert truth.get_offsets().shape[0] == 4
    assert errors.get_offsets().tolist() == [[1.0, 2.0]]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        plot_metric_over_time([np.nan, np.nan], show=False)
    with pytest.raises(ValueError):
        plot_metric_over_time(np.ones((2, 2)), show=False)
    with pytest.raises(TypeError):
        plot_metric_over_time(["a"], show=False)
    with pytest.raises(ValueError):
        compare_models([1.0], [1.0], ("only one",), show=False)
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [0], show=False)
