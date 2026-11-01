"""Unit tests for the pipeline module (NumCompute framework)."""
import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline, Transformer, Estimator
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier

class DummyMultiplier(Transformer):
    def __init__(self, factor=2.0):
        self.factor = factor
    def transform(self, X):
        return np.asarray(X) * self.factor

class DummyEstimator(Estimator):
    def fit(self, X, y=None):
        self.coef_ = 1.0
        return self
    def predict(self, X):
        return np.asarray(X) * 2

def test_pipeline_transform():
    X = [[1.0, 2.0], [3.0, 4.0]]
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('mult', DummyMultiplier(10.0))
    ])
    Xt = pipe.fit_transform(X)
    assert Xt.shape == (2, 2)

def test_pipeline_predict():
    X = [1, 2, 3]
    pipe = Pipeline([
        ('model', DummyEstimator())
    ])
    preds = pipe.fit(X).predict(X)
    assert preds.tolist() == [2, 4, 6]
    
def test_full_pipeline_workflow():
    """Test an end-to-end pipeline combining SimpleImputer and MinMaxScaler."""
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    from numcompute_stream.preprocessing import SimpleImputer, MinMaxScaler
    
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy="mean")),
        ('scale', MinMaxScaler(feature_range=(0.0, 1.0)))
    ])
    
    Xt = pipe.fit_transform(X)
    assert Xt.shape == (3, 2)
    assert not np.isnan(Xt).any()
    assert np.allclose(Xt.min(), 0.0)
    assert np.allclose(Xt.max(), 1.0)


def test_partial_fit_with_decision_tree():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    X = rng.normal(size=(600, 3)) * 0.3 + y[:, None]
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', DecisionTreeClassifier(max_depth=3, buffer_size=200))
    ])
    for X_chunk, y_chunk in zip(np.array_split(X, 6), np.array_split(y, 6)):
        assert pipe.partial_fit(X_chunk, y_chunk) is pipe

    scaler, tree = pipe.steps[0][1], pipe.steps[1][1]
    assert tree.n_retrains_ == 3
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.mean(pipe.predict(X) == y) > 0.9


def test_partial_fit_transformers_match_fit():
    X = np.arange(24.0).reshape(8, 3) ** 2
    make_pipe = lambda: Pipeline([
        ('mult', DummyMultiplier(2.0)),
        ('scale', StandardScaler())
    ])
    stream = make_pipe()
    for chunk in np.array_split(X, 3):
        stream.partial_fit(chunk)
    assert np.allclose(stream.transform(X), make_pipe().fit_transform(X))


def test_partial_fit_checks_every_step_first():
    scaler = StandardScaler()
    pipe = Pipeline([('scale', scaler), ('model', DummyEstimator())])
    with pytest.raises(ValueError, match="model"):
        pipe.partial_fit([[1.0], [2.0]])
    assert scaler.mean_ is None
