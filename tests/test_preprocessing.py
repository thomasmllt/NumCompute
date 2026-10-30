"""Unit tests for the preprocessing module (NumCompute framework).

This file checks the correctness of data preprocessing tranformers: StandardScaler, 
MinMaxScaler, OneHotEncoder and SimpleImputer. It ensures proper behavior with fitting, tranforming,
handling and imputing NaNs and raising appropriate exceptions (ValueError & TypeError).

Covered Test Cases:
- StandardScaler: mean centering and unit variance scaling.
- MinMaxScaler: scaling features to specified ranges, like (0.0, 1.0).
- OneHotEncoder: categorical encoding to binary metrics.
- SimpleImputer: missing value imputation using mean, median or constant strategies.
- Exception handling: for unfitted estimators and invalid types.
"""
import numpy as np
import pytest
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

def test_standard_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    standard_scaler = StandardScaler()
    scaled_X = standard_scaler.fit_transform(X)
    
    assert np.allclose(scaled_X.mean(axis=0), 0.0)
    assert np.allclose(scaled_X.std(axis=0), 1.0)
    
def test_minmax_scaler():
    X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    minmax_scaler = MinMaxScaler()
    scaled_X = minmax_scaler.fit_transform(X)
    
    assert np.allclose(scaled_X.min(axis=0), 0.0)
    assert np.allclose(scaled_X.max(axis=0), 1.0)
    
def test_onehot_encoder():
    X = np.array([[2, 2], [1, 3], [2, 3]])
    encoder = OneHotEncoder()
    encoded_X = encoder.fit_transform(X)
    assert encoded_X.shape == (3, 4)
    assert np.all(encoded_X.sum(axis=1) == 2.0)
    
def test_simple_imputer():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [2.0, 4.0]])
    imputer = SimpleImputer(strategy="mean")
    imputed_X = imputer.fit_transform(X)
    
    assert not np.isnan(imputed_X).any()
    assert imputed_X[0, 1] == 3.0

def test_preprocessing_not_fitted_error():
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.transform([1, 2])
    
def test_preprocessing_constant_column():
    """Test StandardScaler behavior with a constant feature column (zero variance)."""
    X_const = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_const)
    assert not np.isnan(scaled).any()
    assert np.allclose(scaled[:, 0], 0.0)

def test_simple_imputer_median():
    """Test SimpleImputer using the median strategy with NaN values."""
    X_nan = np.array([[1.0, np.nan], [2.0, 2.0], [10.0, 3.0]])
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(X_nan)
    assert not np.isnan(imputed).any()
    assert imputed[0, 1] == 2.5