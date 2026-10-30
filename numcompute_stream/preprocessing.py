"""Data preprocessing transformers for NumCompute framework:
Includes StandardScaler, MinMaxScaler, OneHotEncoder, and SimpleImputer.
All transformations are fully vectorized using Numpy.
Every transformer can be fitted in one go with fit() or chunk by chunk with
partial_fit(). In both cases NaN values are treated as missing and ignored.
"""
import numpy as np

from .stats import StreamingStats, chunk_moments, merge_moments

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]

def to_numeric(a, data="data"):
    """Convert input into a numeric ndarray and raise TypeError otherwise"""
    arr = np.asarray(a)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"{data} must be numeric, got {arr.dtype} instead.")
    return arr

def _as_2d(arr, n_features=None):
    """Return arr with shape (n_samples, n_features), a 1D input being one feature."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got {arr.ndim}D")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(f"X has {arr.shape[1]} features, expected {n_features}")
    return arr

class StandardScaler:
    """
    Standardize features by substracting the mean and scaling to unit variance.
    
    Complexity:
    -----------
    ** Fit & Partial fit:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_features)
    ** Transform:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_samples * n_features)
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Forget all the data seen so far."""
        self.mean_ = None
        self.scale_ = None
        self.n_features_ = None
        self._count = None
        self._m2 = None
        self._min = None
        self._max = None

    def fit(self, X, y=None):
        """Compute mean and standard deviation to be used later for scaling.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input data to compute mean and standrad deviation.
        ------------
        Returns :
            self: object -> Fitted scaler.
        """
        
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Update the mean and standard deviation with a new chunk of data.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) or (n_samples,) -> new data, 1D being one feature.
        ------------
        Returns :
            self: object -> Updated scaler.
        """
        arr = _as_2d(to_numeric(X, "X").astype(float), self.n_features_)
        if arr.shape[0] == 0:
            return self

        if self.mean_ is None:
            self.n_features_ = arr.shape[1]
            self._count = np.zeros(self.n_features_)
            self._m2 = np.zeros(self.n_features_)
            self.mean_ = np.full(self.n_features_, np.nan)
            self._min = np.full(self.n_features_, np.nan)
            self._max = np.full(self.n_features_, np.nan)

        self._count, self.mean_, self._m2 = merge_moments(
            self._count, self.mean_, self._m2, *chunk_moments(arr))
        self._min = np.fmin(self._min, np.fmin.reduce(arr, axis=0))
        self._max = np.fmax(self._max, np.fmax.reduce(arr, axis=0))

        #Avoid zero division on constant columns: rounding can leave a tiny std
        #there, so they are detected with the range and not with std==0
        std = np.sqrt(self._m2 / np.where(self._count > 0, self._count, 1))
        self.scale_ = np.where(self._max > self._min, std, 1.0)

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
    When streaming, a value outside the range seen so far is scaled outside
    feature_range until partial_fit() has been called on it.
    __________
    Complexity:
        ** Fit, Partial fit & Transform:
            - Time : 0(n_samples * n_features)
            - Space : 0(n_features)
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self._reset()

    def _reset(self):
        """Forget all the data seen so far."""
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used later for scaling."""
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Update the minimum and maximum with a new chunk of data.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) or (n_samples,) -> new data, 1D being one feature.
        ------------
        Returns :
            self: object -> Updated scaler.
        """
        n_features = None if self.data_min_ is None else self.data_min_.size
        arr = _as_2d(to_numeric(X, "X").astype(float), n_features)
        if arr.shape[0] == 0:
            return self

        chunk_min = np.fmin.reduce(arr, axis=0)
        chunk_max = np.fmax.reduce(arr, axis=0)
        if self.data_min_ is None:
            self.data_min_, self.data_max_ = chunk_min, chunk_max
        else:
            self.data_min_ = np.fmin(self.data_min_, chunk_min)
            self.data_max_ = np.fmax(self.data_max_, chunk_max)

        has_range = self.data_max_ > self.data_min_
        data_range = np.where(has_range, self.data_max_ - self.data_min_, 1.0)

        feature_min, feature_max = self.feature_range
        feature_range = feature_max - feature_min

        self.scale_ = feature_range / data_range
        self.min_ = feature_min - self.data_min_ * self.scale_
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
    Categories found by partial_fit() are added after the known ones, so that
    a column keeps the same meaning while the stream grows. Each call of fit()
    sorts them. NaN is not a category and is encoded as a row of zeros.
    __________
    Complexity:
        ** Fit, Partial fit & Transform:
            - Time : 0(n_samples * n_features)
            - Space : 0(n_samples * n_unique_categories)
    """

    def __init__(self):
        self.categories_ = []

    def fit(self, X, y=None):
        """Fit a OneHotEncoder to X."""
        self.categories_ = []
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Add the categories of a new chunk of data.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) or (n_samples,) -> new data, 1D being one feature.
        ------------
        Returns :
            self: object -> Updated encoder.
        """
        arr = _as_2d(X, len(self.categories_) or None)
        if arr.shape[0] == 0:
            return self

        if not self.categories_:
            self.categories_ = [np.empty(0, dtype=arr.dtype)] * arr.shape[1]
        for i, catgs in enumerate(self.categories_):
            col = arr[:, i]
            #NaN is the only value that differs from itself
            new_catgs = np.setdiff1d(col[col == col], catgs)
            self.categories_[i] = np.concatenate([catgs, new_catgs])
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
    The running median is estimated from a random sample of at most
    max_samples rows (see StreamingStats): it is exact until that many rows
    have been seen, because an exact median would require keeping all the data.
    __________
    Complexity:
        ** Fit, Partial fit & Transform:
            - Time : 0(n_samples * n_features)
            - Space : 0(n_features), 0(max_samples * n_features) for the median
    """

    def __init__(self, strategy="mean", fill_value=None, max_samples=10000, seed=None):
        if strategy not in ["mean", "median", "constant"]:
            raise ValueError(f"Unrecognized input strategy: {strategy}.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.max_samples = max_samples
        self.seed = seed
        self._reset()

    def _reset(self):
        """Forget all the data seen so far."""
        self.statistics_ = None
        #Only the median needs more than one stored row
        size = self.max_samples if self.strategy == "median" else 1
        self._stats = StreamingStats(max_samples=size, seed=self.seed)

    def fit(self, X, y=None):
        """Compute imputaion statistics."""
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Update the imputation statistics with a new chunk of data.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) or (n_samples,) -> new data, 1D being one feature.
        ------------
        Returns :
            self: object -> Updated imputer.
        """
        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value must be specified as strategy is 'constant'")
        self._stats.update_stats(_as_2d(to_numeric(X, "X")))
        if self._stats.n_samples_seen_ == 0:
            return self

        if self.strategy == "mean":
            self.statistics_ = self._stats.mean_
        elif self.strategy == "median":
            self.statistics_ = self._stats.quantiles(0.5)
        elif self.strategy == "constant":
            self.statistics_ = np.full(self._stats.n_features_, self.fill_value)
        return self

    def transform(self, X):
        """Impure missing values in X."""
        if self.statistics_ is None:
            raise ValueError("SimpleImputer class instance is not fitted yet!")
        arr = to_numeric(X, "X")
        return np.where(np.isnan(arr), self.statistics_, arr)

    def fit_transform(self, X, y=None):
        """Fit to data , then transform it."""
        return self.fit(X).transform(X)