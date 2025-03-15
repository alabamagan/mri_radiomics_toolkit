"""Variance filter for feature selection.

This module provides a feature selection step that filters features based on their variance.
Features with low variance are likely to be constant or near-constant across samples and
thus provide little discriminative power for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union

from sklearn import feature_selection as skfs

from .base import FeatureSelectionStep


class VarianceFilterStep(FeatureSelectionStep):
    """Filter features based on their variance.
    
    This step removes features with variance below a specified threshold. Features with low
    variance are likely to be constant or near-constant across samples and thus provide little
    discriminative power for machine learning models.
    """
    
    def __init__(self, threshold: float = 0.05, name: Optional[str] = "VarianceFilter"):
        """Initialize the variance filter step.
        
        Args:
            threshold: Variance threshold for filtering features. Features with variance
                less than or equal to this threshold will be removed. Default is 0.05.
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.var_filter = None
        self.selected_features_idx = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> 'VarianceFilterStep':
        """Fit the variance filter on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Ignored, included for API consistency
            X_b: Optional second set of features, if provided both feature sets must pass
                the variance threshold to be selected
                
        Returns:
            self: The fitted instance
        """
        self._logger.info(f"Filtering features with variance threshold {self.threshold}")
        
        # Create and fit the variance filter
        self.var_filter = skfs.VarianceThreshold(threshold=self.threshold)
        
        # Fit the variance filter
        self.var_filter.fit(X)
        var_feat_idx = self.var_filter.get_support()
        self.selected_features_idx = var_feat_idx
        
        # If X_b is provided, filter based on both sets
        if X_b is not None:
            var_filter_b = skfs.VarianceThreshold(threshold=self.threshold)
            var_filter_b.fit(X_b)
            var_feat_idx_b = var_filter_b.get_support()
            
            # Only include features that pass the threshold in both sets
            self.selected_features_idx = np.logical_and(var_feat_idx, var_feat_idx_b)
            
            self._logger.info(f"Filtered features based on both feature sets")
            
        # Count features
        n_initial = X.shape[1]
        n_selected = np.sum(self.selected_features_idx)
        self._logger.info(f"Selected {n_selected}/{n_initial} features based on variance threshold")
        
        # Set selected feature names from indices
        self._selected_features = X.columns[self.selected_features_idx].tolist()
        
        self._fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the variance filter to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            X_b: Optional second set of features
            
        Returns:
            X_transformed: The transformed features, or a tuple of (X_transformed, X_b_transformed)
        """
        self._check_fitted()
        
        # Apply the filter to the columns
        X_transformed = X.iloc[:, self.selected_features_idx]
        
        # If X_b is provided, apply the same transformation
        if X_b is not None:
            X_b_transformed = X_b.iloc[:, self.selected_features_idx]
            return X_transformed, X_b_transformed
        
        return X_transformed 