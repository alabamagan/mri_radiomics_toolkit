"""Base classes for feature selection components.

This module provides the base classes that define the common interface for all feature
selection components in the feature_selection subpackage.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
import joblib

from mnts.mnts_logger import MNTSLogger
from ..utils import StateManager

class FeatureSelectionStep(ABC):
    """Base class for all feature selection steps.
    
    All feature selection steps should inherit from this class and implement
    the fit and transform methods.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the feature selection step.
        
        Args:
            name: Optional name for this step, defaults to class name
        """
        self.name = name or self.__class__.__name__
        self._fitted = False
        self._logger = MNTSLogger[self.name]
        self._selected_features = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> 'FeatureSelectionStep':
        """Fit the feature selection step on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, n_targets)
            X_b: Optional second set of features, for methods that require two sets (e.g., ICC filtering)
            
        Returns:
            self: The fitted instance
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the feature selection to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            X_b: Optional second set of features, for methods that require two sets
            
        Returns:
            X_transformed: The transformed features, or a tuple of (X_transformed, X_b_transformed)
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fit the feature selection step and apply it to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, n_targets)
            X_b: Optional second set of features, for methods that require two sets
            
        Returns:
            X_transformed: The transformed features, or a tuple of (X_transformed, X_b_transformed)
        """
        return self.fit(X, y, X_b).transform(X, X_b)
    
    def _check_fitted(self):
        """Check if the step has been fitted.
        
        Raises:
            ValueError: If the step has not been fitted
        """
        if not self._fitted:
            raise ValueError(f"{self.name} has not been fitted yet. Call fit() before using this method.")
    
    def save(self, filepath: Path):
        """Save the feature selection step to a file.
        
        Args:
            filepath: Path to save the step to
            
        Raises:
            ValueError: If the step has not been fitted
        """
        self._check_fitted()
        
        # Create a state dictionary with all necessary attributes
        state_dict = {
            'name': self.name,
            '_fitted': self._fitted,
            '_selected_features': self._selected_features
        }
        
        # Add any additional attributes specific to child classes
        for key, value in self.__dict__.items():
            if key not in ['_logger'] and not key.startswith('__'):
                state_dict[key] = value
            
        # Use StateManager to save the state
        StateManager.save_state(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: Path) -> 'FeatureSelectionStep':
        """Load a feature selection step from a file.
        
        Args:
            filepath: Path to load the step from
            
        Returns:
            step: The loaded feature selection step
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the loaded state is invalid
        """
        try:
            # Create a new instance
            inst = cls()
            
            # Load the state using StateManager
            state = StateManager.load_state(filepath)
            
            # Validate essential attributes
            required_attrs = ['name', '_fitted', '_selected_features']
            missing_attrs = [attr for attr in required_attrs if attr not in state]
            if missing_attrs:
                raise ValueError(f"Loaded state is missing required attributes: {missing_attrs}")
            
            # Update instance attributes
            inst.__dict__.update(state)
            
            # Reinitialize logger
            inst._logger = MNTSLogger[inst.name]
            
            return inst
            
        except Exception as e:
            inst._logger.error(f"Error loading state from {filepath}: {str(e)}")
            raise

    def __getstate__(self):
        """Prepare object state for serialization.
        
        Returns:
            dict: The object's serializable state
        """
        state = self.__dict__.copy()
        # Remove logger as it contains thread lock
        state.pop('_logger', None)
        return state

    def __setstate__(self, state):
        """Restore object state after deserialization.
        
        Args:
            state: The previously saved state
        """
        self.__dict__.update(state)
        # Recreate logger
        self._logger = MNTSLogger[self.name]
        
    @property
    def selected_features(self):
        """Get the selected features.
        
        Returns:
            The selected features or None if not fitted.
        """
        if not self._fitted:
            raise ValueError(f"{self.name} has not been fitted yet. Call fit() before using this method.")
        return self._selected_features
    
    @selected_features.setter
    def selected_features(self, value):
        """Set the selected features.
        
        Args:
            value: The selected feature names to set.
        """
        self._selected_features = value


class FeatureSelectionPipeline:
    """A pipeline of feature selection steps.
    
    This class combines multiple feature selection steps into a single pipeline that
    can be fit and applied as a single entity.
    """
    
    def __init__(self, steps: List[FeatureSelectionStep] = None, name: str = "FeatureSelectionPipeline"):
        """Initialize the feature selection pipeline.
        
        Args:
            steps: List of feature selection steps
            name: Name for this pipeline
        """
        self.steps = steps or []
        self.name = name
        self._logger = MNTSLogger[self.name]
        self._fitted = False
        
    def add_step(self, step: FeatureSelectionStep) -> 'FeatureSelectionPipeline':
        """Add a step to the pipeline.
        
        Args:
            step: The feature selection step to add
            
        Returns:
            self: The updated pipeline
        """
        self.steps.append(step)
        return self
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> 'FeatureSelectionPipeline':
        """Fit the pipeline on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, n_targets)
            X_b: Optional second set of features, for methods that require two sets
            
        Returns:
            self: The fitted pipeline
        """
        if not self.steps:
            self._logger.warning("Empty pipeline, nothing to fit")
            self._fitted = True
            return self
            
        X_transformed = X
        X_b_transformed = X_b
        
        for step in self.steps:
            self._logger.info(f"Fitting step: {step.name}")
            
            if X_b_transformed is not None:
                step.fit(X_transformed, y, X_b_transformed)
                X_transformed, X_b_transformed = step.transform(X_transformed, X_b_transformed)
            else:
                step.fit(X_transformed, y)
                X_transformed = step.transform(X_transformed)
                
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the pipeline to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            X_b: Optional second set of features, for methods that require two sets
            
        Returns:
            X_transformed: The transformed features, or a tuple of (X_transformed, X_b_transformed)
        """
        self._check_fitted()
        
        if not self.steps:
            self._logger.warning("Empty pipeline, returning input unchanged")
            if X_b is not None:
                return X, X_b
            return X
            
        X_transformed = X
        X_b_transformed = X_b
        
        for step in self.steps:
            self._logger.info(f"Applying step: {step.name}")
            
            if X_b_transformed is not None:
                X_transformed, X_b_transformed = step.transform(X_transformed, X_b_transformed)
            else:
                X_transformed = step.transform(X_transformed)
                
        if X_b is not None:
            return X_transformed, X_b_transformed
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fit the pipeline and apply it to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, n_targets)
            X_b: Optional second set of features, for methods that require two sets
            
        Returns:
            X_transformed: The transformed features, or a tuple of (X_transformed, X_b_transformed)
        """
        return self.fit(X, y, X_b).transform(X, X_b)
    
    def _check_fitted(self):
        """Check if the pipeline has been fitted.
        
        Raises:
            ValueError: If the pipeline has not been fitted
        """
        if not self._fitted:
            raise ValueError(f"{self.name} has not been fitted yet. Call fit() before using this method.")
    
    def save(self, filepath: Path):
        """Save the pipeline to a file.
        
        Args:
            filepath: Path to save the pipeline to
        """
        self._check_fitted()
        
        # Create a state dictionary
        state_dict = {
            'name': self.name,
            '_fitted': self._fitted,
            'steps': self.steps
        }
        
        # Use StateManager to save the state
        StateManager.save_state(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: Path) -> 'FeatureSelectionPipeline':
        """Load a pipeline from a file.
        
        Args:
            filepath: Path to load the pipeline from
            
        Returns:
            pipeline: The loaded pipeline
        """
        try:
            # Create a new instance
            inst = cls()
            
            # Load the state
            state = StateManager.load_state(filepath)
            
            # Validate essential attributes
            required_attrs = ['name', '_fitted', 'steps']
            missing_attrs = [attr for attr in required_attrs if attr not in state]
            if missing_attrs:
                raise ValueError(f"Loaded state is missing required attributes: {missing_attrs}")
            
            # Update instance attributes
            inst.__dict__.update(state)
            
            # Reinitialize logger
            inst._logger = MNTSLogger[inst.name]
            
            return inst
            
        except Exception as e:
            inst._logger.error(f"Error loading pipeline from {filepath}: {str(e)}")
            raise

    def __getitem__(self, item: int) -> FeatureSelectionStep:
        """Get a feature selection step by index.

        Args:
            item: The index of the step to retrieve.

        Returns:
            FeatureSelectionStep: The feature selection step at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if not isinstance(item, int):
            raise TypeError(f"Index must be an integer, got {type(item).__name__}")

        if item < 0 or item >= len(self.steps):
            raise IndexError(f"Index {item} is out of range for pipeline with {len(self.steps)} steps")

        return self.steps[item]

    def __getstate__(self):
        """Remove logger because it has thread lock"""
        state = self.__dict__.copy()
        state.pop('_logger', None)
        return state

    def __setstate__(self, state):
        """Recreate logger"""
        self.__dict__.update(state)
        self._logger = MNTSLogger[self.__class__.__name__]
        
    @property
    def selected_features(self):
        """Get the selected features from the last step in the pipeline.
        
        Returns:
            The selected features from the last step in the pipeline, or None if the pipeline
            is empty or not fitted.
        """
        if not self._fitted or not self.steps:
            raise ValueError("Pipeline is not fitted or empty")
            
        # Try to get selected_features from the last step
        last_step = self.steps[-1]
        if hasattr(last_step, 'selected_features'):
            return last_step.selected_features
            
        # If the last step doesn't have selected_features, try each step in reverse order
        for step in reversed(self.steps):
            if hasattr(step, 'selected_features') and step.selected_features is not None:
                return step.selected_features
                
        return  None