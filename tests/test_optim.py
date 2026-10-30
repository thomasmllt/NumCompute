import numpy as np
import pytest

from numcompute_stream.optim import grad, jacobian


def test_grad_central():
    result = grad(lambda x: np.sum(x ** 2), np.array([2.0, 3.0]))
    assert np.allclose(result, [4.0, 6.0])


def test_grad_forward():
    result = grad(lambda x: np.sum(x ** 2), np.array([2.0, 3.0]), method="forward")
    assert np.allclose(result, [4.0, 6.0], atol=1e-4)


def test_jacobian():
    def function(x):
        return np.array([x[0] ** 2, x[0] * x[1]])

    result = jacobian(function, np.array([2.0, 3.0]))
    expected = np.array([[4.0, 0.0], [3.0, 2.0]])
    assert np.allclose(result, expected)


def test_invalid_method_and_step():
    with pytest.raises(ValueError):
        grad(lambda x: x[0], [1.0], method="backward")
    with pytest.raises(ValueError):
        grad(lambda x: x[0], [1.0], h=0)
    with pytest.raises(ValueError):
        jacobian(lambda x: x ** 2, [1.0], h=0)


def test_grad_empty_input():
    result = grad(lambda x: np.sum(x ** 2), np.array([]))
    assert result.shape == (0,)


def test_grad_non_contiguous_input():
    values = np.arange(6.0)[::2]
    result = grad(lambda x: np.sum(x ** 2), values)
    assert np.allclose(result, 2 * values)
