"""Pipeline abstraction for chaining transformers and estimators.
Provides Transformer and Estimator base classes, and a Pipeline class.
"""
import numpy as np

__all__ = ["Transformer", "Estimator", "Pipeline"]

    
class Transformer:
    """Base class for all the transformers in the NumCompute framework."""
    
    def fit(self, X, y=None):
        """Fit transformer to data"""
        return self

    def partial_fit(self, X, y=None):
        """Update transformer with a chunk of data. Stateless by default."""
        return self

    def transform(self, X):
        """Transform data."""
        raise NotImplementedError

    def fit_transform(self, X, y=None):
            """Fit to data then transform it."""
            return self.fit(X,y).transform(X)
    
class Estimator:
    """Base class for all the estimators in the NumCompute framework."""
    
    def fit(self, X, y=None):
        """Fit transformer to data"""
        return self

    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
    
class Pipeline:
    """Pipeline sequence of data tranformers with a final estimator.
    ______
    Complexity:
        *Fit/Transform : 
            -Time:O(sum of steps time)
            -Space:O(memory of largest step)
    """
    
    def __init__(self, steps):
        if not isinstance(steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in steps):
            raise TypeError("Steps must be a list of (name, transform) tuples.")
        self.steps = steps
        
    def fit(self, X, y=None):
        """Fit all transformers."""
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                Xt = step.fit(Xt, y).transform(Xt)
        if self.steps:
            name, step = self.steps[-1]
            if hasattr(step, "fit"):
                step.fit(Xt, y)
        return self

    def partial_fit(self, X, y=None):
        """Update all steps with a new chunk of data.

        Each transformer is updated with the chunk, then transforms it for
        the next step. The final step is only updated. All steps are checked
        first, so that an invalid pipeline does not update part of its steps.

        A transformer whose output width changes breaks the following steps:
        a OneHotEncoder that discovers a category adds a column, which a
        decision tree rejects because its number of features is fixed.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New chunk of data.
        y : array_like, shape (n_samples,), optional
            Target values of the chunk.

        Returns
        -------
        self : Pipeline

        Raises
        ------
        ValueError
            If a step does not implement partial_fit, or an intermediate
            step does not implement transform.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(f"Step '{name}' does not implement partial_fit method.")
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' does not implement transform method.")

        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.partial_fit(Xt, y).transform(Xt)
        if self.steps:
            self.steps[-1][1].partial_fit(Xt, y)
        return self

    def transform(self, X, y=None):
        """Fit all steps and transform data."""
        Xt = X
        for name, step in self.steps:
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                raise ValueError(f"Step '{name}' does not implement transform method.")
        return Xt
    
    def fit_transform(self, X, y=None):
        """Fit all steps and transform the data."""
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                Xt = step.fit(Xt, y).transform(Xt)
        
        if self.steps:
            name, step = self.steps[-1]
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            elif hasattr(step, "fit") and hasattr(step, "transform"):
                Xt = step.fit(Xt, y).transform(Xt)
            elif hasattr(step, "fit"):
                step.fit(Xt, y)
        return Xt
    
    def predict(self, X):
        """Transform intermediate steps and predict on the final estimator."""
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                raise ValueError(f"Step '{name}' does not implement transform method.")
        
        if self.steps:
            name, step = self.steps[-1]
            if hasattr(step, "predict"):
                return step.predict(Xt)
            else:
                raise ValueError(f"Final step '{name}' does not implement predict method.")
        return Xt