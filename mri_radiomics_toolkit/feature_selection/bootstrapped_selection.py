"""Bootstrapped feature selection for stability.

This module provides a feature selection step that uses bootstrapping to improve
the stability of feature selection. Features that consistently appear across multiple
bootstrap samples are more likely to be truly important.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path

from sklearn.utils import resample

from .base import FeatureSelectionStep, FeatureSelectionPipeline
from ..utils import StateManager


class BootstrappedSelectionStep(FeatureSelectionStep):
    """Bootstrapping a step or a pipeline.
    
    The selection_pipeline parameter can be a pipeline with multiple steps or a pipeline
    containing just a single selection step, making this method highly flexible.

    Examples:
        Using with a single selection step:
        
        >>> from mri_radiomics_toolkit.feature_selection import (
        ...     FeatureSelectionPipeline, VarianceFilterStep, BootstrappedSelectionStep
        ... )
        >>> # Create a pipeline with just one step
        >>> simple_pipeline = FeatureSelectionPipeline([VarianceFilterStep(threshold=0.05)])
        >>> # Bootstrap the simple pipeline
        >>> bootstrap_selector = BootstrappedSelectionStep(
        ...     selection_pipeline=simple_pipeline,
        ...     n_bootstrap=100,
        ...     threshold_percentage=0.7
        ... )
        >>> X_selected = bootstrap_selector.fit_transform(X, y)
        
        Using with a multi-step selection pipeline:
        
        >>> from mri_radiomics_toolkit.feature_selection import (
        ...     FeatureSelectionPipeline, VarianceFilterStep, 
        ...     SupervisedSelectionStep, BootstrappedSelectionStep
        ... )
        >>> # Create a pipeline with multiple steps
        >>> complex_pipeline = FeatureSelectionPipeline([
        ...     VarianceFilterStep(threshold=0.05),
        ...     SupervisedSelectionStep(n_features=20)
        ... ])
        >>> # Bootstrap the complex pipeline
        >>> bootstrap_selector = BootstrappedSelectionStep(
        ...     selection_pipeline=complex_pipeline,
        ...     n_bootstrap=100,
        ...     threshold_percentage=0.5
        ... )
        >>> X_selected = bootstrap_selector.fit_transform(X, y)
    """
    def __init__(self, 
                 selection_pipeline: FeatureSelectionPipeline,
                 n_bootstrap: int = 100,
                 bootstrap_ratio: Tuple[float, float] = (0.8, 1.0),
                 threshold_percentage: float = 0.5,
                 return_frequencies: bool = False,
                 random_state: Optional[int] = None,
                 name: Optional[str] = "BootstrappedSelection"):
        """Initialize the bootstrapped selection step.
        
        Args:
            selection_pipeline: The feature selection pipeline to apply to each bootstrap sample.
                This can be a pipeline with multiple steps or a pipeline with just a single
                selection step.
            n_bootstrap: Number of bootstrap samples to create.
            bootstrap_ratio: Range for random bootstrap sample sizes (ratio of original data size).
            threshold_percentage: Threshold for feature selection frequency. Features appearing
                in at least this percentage of bootstrap samples will be selected.
            return_frequencies: Whether to store and return feature frequencies.
            random_state: Random state for reproducibility.
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.selection_pipeline = selection_pipeline
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ratio = bootstrap_ratio
        self.threshold_percentage = threshold_percentage
        self.return_frequencies = return_frequencies
        self.random_state = random_state
        
        self.feature_frequencies = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> 'BootstrappedSelectionStep':
        """Fit the bootstrapped selection on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, 1)
            X_b: Optional second set of features
                
        Returns:
            self: The fitted instance
        """
        self._logger.info(f"Running bootstrapped feature selection with {self.n_bootstrap} samples")
        
        # Set random state for reproducibility
        np.random.seed(self.random_state)
        
        # Generate random bootstrap ratios
        bootstrap_l, bootstrap_u = self.bootstrap_ratio
        bootstrap_ratios = np.random.rand(self.n_bootstrap) * (bootstrap_u - bootstrap_l) + bootstrap_l
        
        # Store selected features from each bootstrap
        features_list = []
        
        # Run bootstrapping
        for i in range(self.n_bootstrap):
            self._logger.info(f"Bootstrap iteration {i+1}/{self.n_bootstrap}")
            
            # Create bootstrap sample of target values
            bootstrap_size = int(len(y) * bootstrap_ratios[i])
            bootstrap_y = resample(y, n_samples=bootstrap_size, random_state=self.random_state+i, stratify=y)
            
            # Get corresponding features
            bootstrap_X = X.loc[bootstrap_y.index]
            bootstrap_X_b = X_b.loc[bootstrap_y.index] if X_b is not None else None
            
            # Apply the selection pipeline to this bootstrap sample
            try:
                if bootstrap_X_b is not None:
                    bootstrap_features, _ = self.selection_pipeline.fit_transform(bootstrap_X, bootstrap_y, bootstrap_X_b)
                else:
                    bootstrap_features = self.selection_pipeline.fit_transform(bootstrap_X, bootstrap_y)
                    
                # Add selected features to the list
                features_list.append(bootstrap_features.columns.tolist())
            except Exception as e:
                self._logger.error(f"Error in bootstrap iteration {i+1}: {e}")
                # Continue with the next iteration
        
        # Count feature frequencies
        if not features_list:
            raise ValueError("No successful bootstrap iterations. Check for errors in the selection pipeline.")
            
        all_features = set()
        for feature_set in features_list:
            all_features.update(feature_set)
            
        # Count frequency of each feature
        feature_counts = {feature: 0 for feature in all_features}
        for feature_set in features_list:
            for feature in feature_set:
                feature_counts[feature] += 1
                
        # Calculate frequencies
        self.feature_frequencies = pd.DataFrame({
            'count': [feature_counts[feature] for feature in all_features],
            'frequency': [feature_counts[feature] / self.n_bootstrap for feature in all_features]
        }, index=list(all_features))
        
        # Select features that appear in at least threshold_percentage of bootstrap iterations
        self._selected_features = self.feature_frequencies[
            self.feature_frequencies['frequency'] >= self.threshold_percentage
        ].index.tolist()
        
        n_total = X.shape[1]
        n_selected = len(self._selected_features)
        self._logger.info(f"Selected {n_selected}/{n_total} features with frequency >= {self.threshold_percentage}")
        
        self._fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the feature selection to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            X_b: Optional second set of features
            
        Returns:
            X_transformed: The transformed features, or a tuple of (X_transformed, X_b_transformed)
        """
        self._check_fitted()
        
        # Filter features
        X_transformed = X[self._selected_features]
        
        # Apply the same transformation to X_b if provided
        if X_b is not None:
            X_b_transformed = X_b[self._selected_features]
            return X_transformed, X_b_transformed
        
        return X_transformed
        
    def get_feature_frequencies(self) -> pd.DataFrame:
        """Get the frequencies of features across bootstrap samples.
        
        Returns:
            feature_frequencies: DataFrame containing counts and frequencies for each feature
        """
        self._check_fitted()
        return self.feature_frequencies
        
    def save(self, filepath: Path):
        """Save the bootstrapped selection step to a file.
        
        Args:
            filepath: Path to save the step to
        """
        self._check_fitted()
        
        # Create a state dictionary with all necessary attributes
        state_dict = {
            'selected_features': self._selected_features,
            'feature_frequencies': self.feature_frequencies,
            'params': {
                'n_bootstrap': self.n_bootstrap,
                'bootstrap_ratio': self.bootstrap_ratio,
                'threshold_percentage': self.threshold_percentage,
                'return_frequencies': self.return_frequencies,
                'random_state': self.random_state,
                'name': self.name
            },
            'pipeline_state': self.selection_pipeline.__getstate__()  # Save the state of the pipeline
        }
        
        # Use StateManager to save the state
        StateManager.save_state(state_dict, filepath)
        
    @classmethod
    def load(cls, filepath: Path) -> 'BootstrappedSelectionStep':
        """Load a bootstrapped selection step from a file.
        
        Args:
            filepath: Path to load the step from
            
        Returns:
            step: The loaded bootstrapped selection step
        """
        try:
            saved_state = StateManager.load_state(filepath)

            # Because wrapped pipeline is unknown before class creation, we must implement this specifically
            from .base import FeatureSelectionPipeline
            selection_pipeline = FeatureSelectionPipeline()
            
            # Reload saved states
            selection_pipeline.__setstate__(saved_state['pipeline_state'])

            # Now create the bootstrapper instance
            params = saved_state['params']
            instance = cls(
                selection_pipeline=selection_pipeline,
                **params
            )
            
            # Load the states of the outer pipeline instance
            instance._selected_features = saved_state['selected_features']
            instance.feature_frequencies = saved_state['feature_frequencies']
            instance._fitted = True
            
            return instance
            
        except Exception as e:
            logger = MNTSLogger[cls.__name__]
            logger.error(f"Error loading bootstrapped selection step from {filepath}: {str(e)}")
            raise 