"""Unit tests for the pipeline module (NumCompute framework)."""
import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline, Transformer, Estimator
from numcompute_stream.preprocessing import StandardScaler

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