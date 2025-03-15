"""Bootstrapped Boosted RENT (BBRENT) feature selection.

This module provides a specialized feature selection step that combines bootstrapping with
the RENT/BRENT algorithm for supervised feature selection. This approach is designed to
improve stability in feature selection by selecting features that consistently appear
across multiple bootstrap samples.

This implementation maintains backward compatibility with the original feature_selection.py
implementation while leveraging the modular components of the new design.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, Iterable, List, Dict, Any
from pathlib import Path

from .base import FeatureSelectionStep, FeatureSelectionPipeline
from .supervised_selection import SupervisedSelectionStep
from .bootstrapped_selection import BootstrappedSelectionStep
from ..utils import StateManager


class BBRENTStep(BootstrappedSelectionStep):
    """Bootstrapped Boosted RENT (BBRENT) feature selection.
    
    This step implements the BBRENT algorithm which combines bootstrapping with
    the RENT/BRENT algorithm for supervised feature selection. It creates a pipeline
    with a single SupervisedSelectionStep and wraps it with BootstrappedSelectionStep
    to improve stability.
    
    This class maintains backward compatibility with the original implementation
    while leveraging the modular components of the new design.
    
    Examples:
        >>> from mri_radiomics_toolkit.feature_selection import BBRENTStep
        >>> # Create a BBRENT selector
        >>> bbrent = BBRENTStep(
        ...     criteria_threshold=(0.9, 0.5, 0.99),
        ...     n_trials=500,
        ...     n_bootstrap=250,
        ...     threshold_percentage=0.4,
        ...     boosting=True
        ... )
        >>> # Fit and transform
        >>> X_selected = bbrent.fit_transform(X, y)
    """
    
    def __init__(self,
                 criteria_threshold: Tuple[float, float, float] = (0.9, 0.5, 0.99),
                 n_trials: int = 500,
                 n_bootstrap: int = 250,
                 bootstrap_ratio: Tuple[float, float] = (0.8, 1.0),
                 threshold_percentage: float = 0.4,
                 return_frequency: bool = False,
                 boosting: bool = True,
                 random_state: Optional[int] = None,
                 name: Optional[str] = "BBRENT"):
        """Initialize the BBRENT feature selection step.
        
        Args:
            criteria_threshold: Thresholds for the three RENT criteria 
                (frequency, stability, effect size).
            n_trials: Number of trials to run for RENT/BRENT.
            n_bootstrap: Number of bootstrap samples to create.
            bootstrap_ratio: Range for random bootstrap sample sizes (ratio of original data size).
            threshold_percentage: Threshold for feature selection frequency. Features appearing
                in at least this percentage of bootstrap samples will be selected.
            return_frequency: Whether to store and return feature frequencies.
            boosting: Whether to use boosting (BRENT instead of RENT).
            random_state: Random state for reproducibility.
            name: Optional name for this step.
        """
        # Create a supervised selection step for RENT/BRENT
        supervised_step = SupervisedSelectionStep(
            criteria_threshold=criteria_threshold,
            n_trials=n_trials,
            boosting=boosting,
            name="RENT_Step"
        )
        
        # Create a pipeline with just the supervised step
        inner_pipeline = FeatureSelectionPipeline([supervised_step], name="RENT_Pipeline")
        
        # Initialize the BootstrappedSelectionStep with the pipeline
        super().__init__(
            selection_pipeline=inner_pipeline,
            n_bootstrap=n_bootstrap,
            bootstrap_ratio=bootstrap_ratio,
            threshold_percentage=threshold_percentage,
            return_frequencies=return_frequency,
            random_state=random_state,
            name=name
        )

    @property
    def n_trials(self):
        return self.selection_pipeline[0].n_trials

    @property
    def criteria_threshold(self):
        return self.selection_pipeline[0].criteria_threshold

    @property
    def boosting(self):
        return self.selection_pipeline[0].boosting

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> 'BBRENTStep':
        """Fit the BBRENT selection on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, 1)
            X_b: Optional second set of features
                
        Returns:
            self: The fitted instance
        """
        self._logger.info(f"Running BBRENT feature selection with {self.n_bootstrap} bootstraps and {self.n_trials} trials")
        
        # Fit the bootstrapped step
        super().fit(X, y, X_b)
        
        self._fitted = True
        return self
    
    def get_feature_frequencies(self) -> pd.DataFrame:
        """Get the frequencies of features across bootstrap samples.
        
        Returns:
            feature_frequencies: DataFrame containing counts and frequencies for each feature
        """
        self._check_fitted()
        return self.feature_frequencies
    
    def save(self, filepath: Path):
        """Save the BBRENT selection step to a file.
        
        Args:
            filepath: Path to save the step to
        """
        self._check_fitted()
        
        # Create a state dictionary with all necessary attributes
        state_dict = {
            'selected_features': self._selected_features,
            'feature_frequencies': self.feature_frequencies,
            'params': {
                'criteria_threshold': self.criteria_threshold,
                'n_trials': self.n_trials,
                'n_bootstrap': self.n_bootstrap,
                'bootstrap_ratio': self.bootstrap_ratio,
                'threshold_percentage': self.threshold_percentage,
                'return_frequencies': self.return_frequencies,
                'random_state': self.random_state,
                'name': self.name
            }
        }
        
        # Use StateManager to save the state
        StateManager.save_state(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: Path) -> 'BBRENTStep':
        """Load a BBRENT selection step from a file.
        
        Args:
            filepath: Path to load the step from
            
        Returns:
            step: The loaded BBRENT selection step
        """
        try:
            # Load the state using StateManager
            saved_state = StateManager.load_state(filepath)
            
            # Create instance with saved parameters
            params = saved_state['params']
            instance = cls(
                criteria_threshold=params['criteria_threshold'],
                n_trials=params['n_trials'],
                n_bootstrap=params['n_bootstrap'],
                bootstrap_ratio=params['bootstrap_ratio'],
                threshold_percentage=params['threshold_percentage'],
                return_frequency=params['return_frequencies'],
                boosting=True,
                random_state=params['random_state'],
                name=params['name']
            )
            
            # Restore state
            instance._selected_features = saved_state['selected_features']
            instance.feature_frequencies = saved_state['feature_frequencies']
            instance._fitted = True
            
            return instance
            
        except Exception as e:
            # Log the error
            instance._logger.error(f"Error loading BBRENT step from {filepath}: {str(e)}")
            raise 