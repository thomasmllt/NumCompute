"""Data preprocessing transformers for NumCompute framework:
Includes StandardScaler, MinMaxScaler, OneHotEncoder, and SimpleImputer.
All transformations are fully vectorized using Numpy.
"""
import numpy as np
__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]

def to_numeric(a, data="data"):
    """Convert input into a numeric ndarray and raise TypeError otherwise"""
    arr = np.asarray(a)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"{data} must be numeric, got {arr.dtype} instead.")
    return arr

class StandardScaler:
    """
    Standardize features by substracting the mean and scaling to unit variance.
    
    Complexity:
    -----------
    ** Fit:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_features)
    ** Transform:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_samples * n_features)
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X, y=None):
        """Compute mean and standard deviation to be used later for scaling.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input data to compute mean and standrad deviation.
        ------------
        Returns :
            self: object -> Fitted scaler.
        """
        
        arr = to_numeric(X, "X")
        self.mean_ = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        
        #Avoid zero division when std is 0
        self.scale_ = np.where(std==0.0, 1.0, std)
        
        return self
    
    def transform(self, X):
        """Perform data standardization by centering and scaling it.
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input data to compute mean and standrad deviation.
            ------------
        Returns :
            self: object -> Scaled output."""
        
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet!")
        arr = to_numeric(X, "X")
        
        return (arr - self.mean_) / self.scale_
        
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
        
        
class MinMaxScaler:
    """Transform data by scaling each feature to a given range.
    __________
    Complexity:
        ** Fit & Transform:
            - Time : 0(n_samples * n_features)
            - Space : 0(n_features)
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.min_ = None
        self.scale_ = None
        self.feature_range = feature_range
        
    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used later for scaling."""
        arr = to_numeric(X, "X")
        minimum = np.min(arr, axis=0)
        maximum = np.max(arr, axis=0)
        data_range = maximum - minimum
        data_range = np.where(data_range==0.0, 1.0, data_range)
        
        feature_min, feature_max = self.feature_range
        feature_range = feature_max - feature_min
        
        self.scale_ = feature_range / data_range
        self.min_ = feature_min - minimum * self.scale_
        return self
    
    def transform(self, X):
        if self.scale_ is None or self.min_ is None:
            raise ValueError("This MinMaxScaler instance sould be fitted before!")
        arr = to_numeric(X, "X")
        return arr * self.scale_ + self.min_
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
class OneHotEncoder:
    """Encode categorical features as a binary matrix by creating a new binary column for each unique category.
    __________
    Complexity:
        ** Fit & Transform:
            - Time : 0(n_samples * n_features)
            - Space : 0(n_samples * n_unique_categories)
    """
    
    def __init__(self):
        self.categories_ = []
        
    def fit(self, X, y=None):
        """Fit a OneHotEncoder to X."""
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
        self.categories_ = [np.unique(arr[:, i]) for i in range(arr.shape[1])]
        return self
    
    def transform(self, X):
        """Transform X using one-hot encoding
        """
        if not self.categories_:
            raise ValueError("OneHotEncoder class instance is not fitted yet!")
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
        #Initiate encoded matrix
        encoded_cols = []
        for i, catgs in enumerate(self.categories_):
            col = arr[:,i]
            
            #Create a boolean matrix for broadcasting
            encoded = (col[:,None] == catgs).astype(float)
            encoded_cols.append(encoded)
        return np.hstack(encoded_cols)
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
class SimpleImputer:
    """Imputaion transformer for completing missing values (NaN).
    Complexity:
        ** Fit & Transform:
            - Time : 0(n_samples * n_features)
            - Space : 0(n_features)
    """
    
    def __init__(self, strategy="mean", fill_value=None):
        if strategy not in ["mean", "median", "constant"]:
            raise ValueError("Unrecognized input strategy: {strategy}.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
    
    def fit(self, X, y=None):
        """Compute imputaion statistics."""
        arr = to_numeric(X, "X")
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(arr, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(arr, axis=0)
        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be specified as strategy is 'constant'")
            self.statistics_ = np.full(arr.shape[1], self.fill_value)
        return self
    
    def transform(self, X):
        """Impure missing values in X."""
        if self.statistics_ is None:
            raise ValueError("SimpleImputer class instance is not fitted yet!")
        arr = to_numeric(X, "X").copy()
        mask = np.isnan(arr)
        if mask.any():
            rows, cols = np.nonzero(mask)
            arr[rows, cols] = self.statistics_[cols]
        return arr
    
    def fit_transform(self, X, y=None):
        """Fit to data , then transform it."""
        return self.fit(X).transform(X)