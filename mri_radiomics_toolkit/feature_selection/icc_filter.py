"""ICC filter for feature selection.

This module provides a feature selection step that filters features based on their
Intraclass Correlation Coefficient (ICC). Features with low ICC values indicate poor
reliability across different segmentations or measurements and should be removed.
"""

import pandas as pd
import numpy as np
import multiprocessing as mpi
from typing import Optional, Tuple, Union, List

import pingouin as pg
from functools import partial
from tqdm.auto import tqdm

from .base import FeatureSelectionStep


class ICCFilterStep(FeatureSelectionStep):
    """Filter features based on their Intraclass Correlation Coefficient (ICC).
    
    This step removes features with ICC values below a specified threshold. ICC measures
    the reliability and consistency of features across different segmentations or measurements.
    Features with low ICC values indicate poor reliability and should be removed.
    """
    
    def __init__(self, threshold: float = 0.9, ICC_form: str = 'ICC2k', name: Optional[str] = "ICCFilter"):
        """Initialize the ICC filter step.
        
        Args:
            threshold: ICC threshold for filtering features. Features with ICC value
                less than this threshold will be removed. Default is 0.9.
            ICC_form: The form of ICC to compute. One of ['ICC1', 'ICC2', 'ICC3', 'ICC1k', 'ICC2k', 'ICC3k'].
                Default is 'ICC2k'.
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.ICC_form = ICC_form
        self.icc_values = None
        
    def compute_ICC(self, X_a: pd.DataFrame, X_b: pd.DataFrame) -> pd.DataFrame:
        """Compute the ICC of features calculated from two sets of segmentation.
        
        Args:
            X_a: Features from first segmentation (n_samples, n_features)
            X_b: Features from second segmentation (n_samples, n_features)
                
        Returns:
            icc_df: DataFrame containing ICC values for each feature.
            outted_features: List of features that were filtered out during ICC computation.
        """
        self._logger.info("Computing ICC values")
        
        # Prepare data for ICC computation
        feature_names = X_a.columns
        icc_df = []
        outted_features = []
        
        # Use multiprocessing to speed up computation
        pool = mpi.Pool(mpi.cpu_count())
        
        # Prepare data for each feature
        feature_data = []
        for feature in feature_names:
            # Combine data from both segmentations
            df = pd.DataFrame({
                'Patient': np.repeat(range(len(X_a)), 2),
                'Segmentation': ['A'] * len(X_a) + ['B'] * len(X_b),
                'value': np.concatenate([X_a[feature], X_b[feature]])
            })
            feature_data.append((df, feature))
        
        # Compute ICC for each feature
        res = pool.starmap_async(
            partial(pg.intraclass_corr, targets='Patient', raters='Segmentation', ratings='value'),
            [data for data, _ in feature_data]
        )
        pool.close()
        pool.join()
        results = res.get()
        
        # Process results
        for (_, feature), icc in zip(feature_data, results):
            self._logger.info(f"Computing ICC: {feature}")
            icc['Feature'] = feature
            icc.set_index(['Feature', 'Type'], inplace=True)
            icc_df.append(icc)
        
        icc_df = pd.concat(icc_df)
        
        # Filter by specified ICC form
        drop_this = ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]
        if self.ICC_form not in drop_this:
            raise AttributeError(f"ICC form can only be one of the following: {drop_this}")
        
        drop_this.remove(self.ICC_form)
        icc_df = icc_df.drop(drop_this, level=1)
        
        return icc_df, outted_features
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: pd.DataFrame = None) -> 'ICCFilterStep':
        """Fit the ICC filter on the input data.
        
        Args:
            X: Input features from the first segmentation (n_samples, n_features)
            y: Ignored, included for API consistency
            X_b: Input features from the second segmentation (n_samples, n_features). Must be provided.
                
        Returns:
            self: The fitted instance
            
        Raises:
            ValueError: If X_b is not provided.
        """
        if X_b is None:
            raise ValueError("X_b must be provided for ICC filtering. It should contain features from a second segmentation.")
        
        self._logger.info(f"Filtering features with ICC threshold {self.threshold}")
        
        # Compute ICC values
        icc_values, _ = self.compute_ICC(X, X_b)
        self.icc_values = icc_values
        
        # Select features that pass the ICC threshold
        self._selected_features = icc_values.loc[icc_values['ICC'] >= self.threshold].index.get_level_values(0).unique()
        
        n_total = X.shape[1]
        n_selected = len(self._selected_features)
        self._logger.info(f"Selected {n_selected}/{n_total} features with ICC >= {self.threshold}")
        
        self._fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the ICC filter to the input data.
        
        Args:
            X: Input features from the first segmentation (n_samples, n_features)
            X_b: Optional input features from the second segmentation
            
        Returns:
            X_transformed: Features from the first segmentation that passed the ICC threshold
            X_b_transformed: Features from the second segmentation that passed the ICC threshold (if X_b is provided)
        """
        self._check_fitted()
        
        # Filter features
        X_transformed = X[self._selected_features]
        
        # Apply the same transformation to X_b if provided
        if X_b is not None:
            X_b_transformed = X_b[self._selected_features]
            return X_transformed, X_b_transformed
        
        return X_transformed 