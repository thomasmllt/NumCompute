"""Utility functions including activations, distances, logsumexp and batching.
Fully vectorised using Numpy (no python loops).
"""
import numpy as np

__all__ = ["euclidean_distance", "sigmoid", "softmax", "logsumexp", "batch_generator"]

def euclidean_distance(x, y):
    """Compute eucledian distances between vectors/matrices."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return np.linalg.norm(x_arr - y_arr, axis=-1)

def sigmoid(x):
    """Computes numerically stable sigmoid activation."""
    arr = np.asarray(x, dtype=float)
    return np.where(
        arr >= 0,
        1.0 / (1.0 + np.exp(-arr)),
        np.exp(arr) / (1.0 + np.exp(arr))
    )
    
def softmax(x, axis=-1):
    """Compute numerically stable softmax activation."""
    arr = np.asarray(x, dtype = float)
    max_val = np.max(arr, axis=axis, keepdims=True)
    exp_arr = np.exp(arr - max_val)
    return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)

def logsumexp(x, axis=-1, keepdims=False):
    """Compute log-sum-exp safely to prevent numerical verflow."""
    arr = np.asarray(x, dtype=float)
    max_val = np.max(arr, axis=axis, keepdims=True)
    max_val_s = max_val if keepdims else np.squeeze(max_val, axis=axis)
    return max_val_s + np.log(np.sum(np.exp(arr - max_val), axis=axis, keepdims=keepdims))

def batch_generator(X, batch_size=32, shuffle=True, seed=None):
    """Yield mini-batches from a dataset."""
    arr = np.asarray(X)
    n_samples = arr.shape[0]
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
        
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield arr[batch_idx]