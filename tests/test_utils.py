"""Unit tests for the utils module (NumCompute framework)."""
import numpy as np
import pytest
from numcompute_stream.utils import euclidean_distance, sigmoid, softmax, logsumexp, batch_generator

def test_euclidean_distance():
    u = [0.0, 0.0]
    v = [3.0, 4.0]
    assert np.allclose(euclidean_distance(u, v), 5.0)

def test_sigmoid():
    x = np.array([0.0, 100.0, -100.0])
    s = sigmoid(x)
    
    assert np.isclose(s[0], 0.5)
    assert np.isclose(s[1], 1.0)
    assert np.isclose(s[2], 0.0)

def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    sm = softmax(x)
    assert np.allclose(np.sum(sm), 1.0)
    assert sm.argmax() == 2

def test_logsumexp():
    x = np.array([1.0, 2.0, 3.0])
    lse = logsumexp(x)
    expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
    assert np.allclose(lse, expected)

def test_batch_generator():
    X = np.arange(10)
    batches = list(batch_generator(X, batch_size=3, shuffle=False))
    assert len(batches) == 4
    assert np.array_equal(batches[0], [0, 1, 2])
    assert np.array_equal(batches[3], [9])