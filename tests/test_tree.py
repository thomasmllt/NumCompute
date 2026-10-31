"""Unit tests for DecisionTreeClassifier: learning, buffered retraining and
the numerical edge cases of split search.
"""
import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


def _blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = rng.normal(size=(n, 4)) * 0.4 + y[:, None]
    return X, y


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_learns_separable_classes(criterion):
    X, y = _blobs()
    tree = DecisionTreeClassifier(max_depth=4, criterion=criterion).fit(X, y)
    X_test, y_test = _blobs(seed=1)
    assert np.mean(tree.predict(X_test) == y_test) > 0.9


def test_buffer_retrains_only_when_full():
    X, y = _blobs(n=100)
    tree = DecisionTreeClassifier(buffer_size=50)
    tree.partial_fit(X[:30], y[:30])
    assert not tree.is_fitted_ and tree.n_buffered_ == 30
    with pytest.raises(ValueError):
        tree.predict(X)

    tree.partial_fit(X[30:60], y[30:60])  # 60 >= 50: retrain on all 60
    assert tree.n_retrains_ == 1 and tree.n_buffered_ == 0
    expected = DecisionTreeClassifier(buffer_size=50).fit(X[:60], y[:60])
    assert np.array_equal(tree.predict(X), expected.predict(X))


def test_predict_keeps_last_tree_between_retrains():
    X, y = _blobs(n=100)
    tree = DecisionTreeClassifier(buffer_size=50).partial_fit(X[:50], y[:50])
    before = tree.predict(X)
    tree.partial_fit(X[50:80], 1 - np.minimum(y[50:80], 1))
    assert np.array_equal(tree.predict(X), before)


def test_flush_trains_on_incomplete_buffer():
    X, y = _blobs(n=20)
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(X, y).flush()
    assert tree.is_fitted_ and tree.n_buffered_ == 0


def test_classes_are_kept_across_retrains():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit(X[:2], ["a", "b"])
    tree.partial_fit(X[2:], ["b", "c"])
    assert tree.classes_.tolist() == ["a", "b", "c"]
    proba = tree.predict_proba(X)
    assert proba.shape == (4, 3) and np.all(proba[:, 0] == 0.0)


def test_max_depth_and_min_samples_split():
    X, y = _blobs()
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).depth_ <= 2
    stump = DecisionTreeClassifier(min_samples_split=len(y) + 1).fit(X, y)
    assert stump.n_nodes_ == 1


def test_pure_or_constant_data_gives_a_single_leaf():
    X = np.ones((5, 2))
    tree = DecisionTreeClassifier().fit(X, [0, 1, 1, 1, 0])
    assert tree.n_nodes_ == 1
    assert tree.predict(X).tolist() == [1] * 5
    assert DecisionTreeClassifier().fit(np.eye(3), [2, 2, 2]).n_nodes_ == 1


def test_nan_features_are_supported():
    X, y = _blobs()
    X[::3, 0] = np.nan
    X[:, 3] = np.nan
    tree = DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert np.mean(tree.predict(X) == y) > 0.9
    assert not np.isnan(tree.predict_proba(np.full((1, 4), np.nan))).any()


def test_split_between_adjacent_floats():
    low = 1.0
    X = np.array([[low], [np.nextafter(low, 2.0)]])
    assert DecisionTreeClassifier().fit(X, [0, 1]).predict(X).tolist() == [0, 1]


def test_max_features_is_reproducible():
    X, y = _blobs()
    trees = [DecisionTreeClassifier(max_features="sqrt", seed=3).fit(X, y)
             for _ in range(2)]
    assert np.array_equal(trees[0].predict_proba(X), trees[1].predict_proba(X))


@pytest.mark.parametrize("kwargs", [
    {"max_depth": 0}, {"criterion": "mse"}, {"min_samples_split": 1},
    {"buffer_size": 0}, {"max_features": 0.0}, {"max_features": "all"},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)


def test_invalid_inputs():
    tree = DecisionTreeClassifier(buffer_size=2).partial_fit(np.ones((2, 2)), [0, 1])
    with pytest.raises(ValueError):
        tree.partial_fit(np.ones((2, 3)), [0, 1])
    with pytest.raises(ValueError):
        tree.predict(np.ones(2))
    with pytest.raises(ValueError):
        tree.partial_fit([[np.inf, 0.0]], [0])
    with pytest.raises(ValueError):
        tree.partial_fit(np.ones((2, 2)), [0])
    with pytest.raises(TypeError):
        tree.predict([["a", "b"]])
