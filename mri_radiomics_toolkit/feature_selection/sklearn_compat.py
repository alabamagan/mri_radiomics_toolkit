"""sklearn-compatible feature selectors.

This module provides feature selectors that are compatible with sklearn's Pipeline API.
These selectors can be used as transformers in a sklearn pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Any

from sklearn.base import BaseEstimator, TransformerMixin

from .supervised_selection import SupervisedSelectionStep


class SklearnFeatureSelector(BaseEstimator, TransformerMixin):
    """Base class for sklearn-compatible feature selectors.
    
    This is a wrapper around our feature selection steps that follows sklearn's
    transformer protocol so they can be used in sklearn.pipeline.Pipeline.
    """
    
    def __init__(self, selector_step):
        """Initialize the selector.
        
        Args:
            selector_step: The feature selection step to wrap
        """
        self.selector_step = selector_step
        self.feature_names_in_ = None
        self.selected_feature_names_ = None
        self.selected_feature_indices_ = None
        self._is_df_input = False
        
    def fit(self, X, y=None, **fit_params):
        """Fit the selector to the data.
        
        Args:
            X: Input features (samples, features)
            y: Target values
            **fit_params: Additional parameters to pass to the selector
            
        Returns:
            self: The fitted selector
        """
        # Remember if input is a DataFrame (to preserve column names)
        self._is_df_input = isinstance(X, pd.DataFrame)
        
        # Remember feature names/indices
        if self._is_df_input:
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = np.arange(X.shape[1])
            
        # Convert to DataFrame if needed for our selector
        if not self._is_df_input:
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        # Convert y to DataFrame if needed
        if isinstance(y, pd.Series):
            y_df = pd.DataFrame(y)
        elif not isinstance(y, pd.DataFrame) and y is not None:
            y_df = pd.DataFrame(y)
        else:
            y_df = y
            
        # For our selectors, features should be rows, samples should be columns
        X_df = X_df.T
        
        # Fit the selector step
        self.selector_step.fit(X_df, y_df)
        
        # Get selected features
        selected_features = self.selector_step.transform(X_df)
        
        # Store selected feature info
        if self._is_df_input:
            self.selected_feature_names_ = selected_features.columns
            # Find indices of selected features
            self.selected_feature_indices_ = np.array([
                np.where(self.feature_names_in_ == name)[0][0]
                for name in self.selected_feature_names_
            ])
        else:
            self.selected_feature_indices_ = np.array([
                i for i in range(X.shape[1]) 
                if i in selected_features.columns
            ])
            
        return self
        
    def transform(self, X):
        """Transform the data by selecting features.
        
        Args:
            X: Input features (samples, features)
            
        Returns:
            X_transformed: The transformed features (samples, selected_features)
        """
        # Check if fitted
        if self.selected_feature_indices_ is None:
            raise ValueError("Selector has not been fitted")
            
        # Select features
        if self._is_df_input and isinstance(X, pd.DataFrame):
            return X[self.selected_feature_names_]
        else:
            return X[:, self.selected_feature_indices_]
            
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        
        Args:
            input_features: Input feature names (ignored)
            
        Returns:
            output_feature_names: Output feature names
        """
        # Check if fitted
        if self.selected_feature_names_ is None and self.selected_feature_indices_ is None:
            raise ValueError("Selector has not been fitted")
            
        # Return selected feature names
        if self.selected_feature_names_ is not None:
            return self.selected_feature_names_
        else:
            return np.array([f"feature_{i}" for i in self.selected_feature_indices_])


class ElasticNetSelector(SklearnFeatureSelector):
    """Feature selector using elastic net.
    
    This selector uses a single elastic net model to select features.
    It's compatible with the original ENetSelector in the codebase.
    """
    
    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, n_features: int = None):
        """Initialize the elastic net selector.
        
        Args:
            alpha: Regularization strength (higher values = more regularization)
            l1_ratio: Ratio between L1 and L2 regularization (1.0 = LASSO, 0.0 = Ridge)
            n_features: Maximum number of features to select (None = all non-zero coef features)
        """
        # Create a SupervisedSelectionStep with n_trials=1 to use a single elastic net
        selector_step = SupervisedSelectionStep(
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_trials=1,  # Use a single elastic net run
            n_features=n_features or 9999,  # If None, use a large number to keep all non-zero coef features
            boosting=False  # No boosting for single elastic net
        )
        
        super().__init__(selector_step=selector_step)
        
        # Save parameters for sklearn's get_params
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_features = n_features 