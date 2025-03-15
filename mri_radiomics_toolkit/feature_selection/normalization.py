"""Feature normalization for feature selection.

This module provides a feature selection step that normalizes features to have zero mean
and unit variance, which is important for many machine learning algorithms.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union

from sklearn import preprocessing

from .base import FeatureSelectionStep


class NormalizationStep(FeatureSelectionStep):
    """Normalize features to have zero mean and unit variance.
    
    This step applies standard scaling to features, which is important for many
    machine learning algorithms, particularly those that are distance-based or
    use regularization.
    """
    
    def __init__(self, name: Optional[str] = "Normalization"):
        """Initialize the normalization step.
        
        Args:
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.scaler = None
        self.scaler_b = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> 'NormalizationStep':
        """Fit the normalization step on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Ignored, included for API consistency
            X_b: Optional second set of features
                
        Returns:
            self: The fitted instance
        """
        self._logger.info("Fitting feature normalization")
        
        # Create and fit the scaler
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(X.values)
            
        # If X_b is provided, fit a separate scaler for it
        if X_b is not None:
            self.scaler_b = preprocessing.StandardScaler()
            self.scaler_b.fit(X_b.values)
                
        self._fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply normalization to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            X_b: Optional second set of features
            
        Returns:
            X_transformed: The normalized features, or a tuple of (X_transformed, X_b_transformed)
        """
        self._check_fitted()
        
        # Transform X
        X_scaled = self.scaler.transform(X.values)
        X_transformed = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            
        # Transform X_b if provided
        if X_b is not None:
            if self.scaler_b is None:
                self._logger.warning("X_b provided but no scaler was fitted for it. Using the scaler fitted on X.")
                scaler_to_use = self.scaler
            else:
                scaler_to_use = self.scaler_b
                
            X_b_scaled = scaler_to_use.transform(X_b.values)
            X_b_transformed = pd.DataFrame(X_b_scaled, index=X_b.index, columns=X_b.columns)
            return X_transformed, X_b_transformed
                
        return X_transformed 