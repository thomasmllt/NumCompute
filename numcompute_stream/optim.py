"""Simple numerical differentiation functions."""

import numpy as np

__all__ = ["grad", "jacobian"]


def _check_step(h):
    if not isinstance(h, (int, float, np.integer, np.floating)) or h <= 0:
        raise ValueError("h must be a positive number")


def grad(f, x, h=1e-5, method="central"):
    """Approximate the gradient of a scalar function.

    Parameters
    ----------
    f : callable
        Function returning a scalar.
    x : array_like
        Point at which the gradient is estimated.
    h : float, default=1e-5
        Positive finite-difference step.
    method : {'central', 'forward'}, default='central'
        Finite-difference method.

    Returns
    -------
    ndarray, shape like x
        Estimated partial derivatives.

    Raises
    ------
    ValueError
        If ``h`` is not positive or ``method`` is unsupported.

    Complexity
    ----------
    Time: O(n) function evaluations for n input values.
    Space: O(n).
    """
    _check_step(h)
    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")

    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)

    for index in np.ndindex(x.shape):
        step = np.zeros_like(x)
        step[index] = h
        if method == "central":
            result[index] = (f(x + step) - f(x - step)) / (2 * h)
        else:
            result[index] = (f(x + step) - f(x)) / h

    return result


def jacobian(F, x, h=1e-5, method="central"):
    """Approximate the Jacobian of a vector-valued function.

    Parameters
    ----------
    F : callable
        Function returning an array.
    x : array_like
        Point at which the Jacobian is estimated.
    h : float, default=1e-5
        Positive finite-difference step.
    method : {'central', 'forward'}, default='central'
        Finite-difference method.

    Returns
    -------
    ndarray, shape F(x).shape + x.shape
        Estimated partial derivatives.

    Raises
    ------
    ValueError
        If ``h`` is not positive or ``method`` is unsupported.

    Complexity
    ----------
    Time: O(n) function evaluations for n input values.
    Space: O(n + output_size).
    """
    _check_step(h)
    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")

    x = np.asarray(x, dtype=float)
    base = np.asarray(F(x), dtype=float)
    result = np.empty(base.shape + x.shape, dtype=float)

    for index in np.ndindex(x.shape):
        step = np.zeros_like(x)
        step[index] = h
        if method == "central":
            difference = (np.asarray(F(x + step)) - np.asarray(F(x - step))) / (2 * h)
        else:
            difference = (np.asarray(F(x + step)) - base) / h
        result[(...,) + index] = difference

    return result
