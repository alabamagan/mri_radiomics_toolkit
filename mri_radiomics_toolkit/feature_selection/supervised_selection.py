"""Supervised feature selection using RENT/BRENT.

This module provides a feature selection step that uses RENT (Repeated Elastic Net Technique)
or BRENT (Boosted RENT) for supervised feature selection. Using this alone does not bring 
benefits over using RENT, it must be used with bootstraping to get stable results.

.. seealso::
    :class:`mri_radiomics_toolkit.feature_selection.bootstrapped_selection.BootstrappedSelectionStep`
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, Iterable, List
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from RENT import RENT

from .base import FeatureSelectionStep


class SupervisedSelectionStep(FeatureSelectionStep):
    """Select features using RENT or BRENT.
    
    This step uses RENT (Repeated Elastic Net Technique) or BRENT (Boosted RENT)
    for supervised feature selection. RENT runs multiple elastic net models and
    evaluates features based on their selection frequency and coefficient values.
    BRENT adds boosting to improve the selection stability.
    """
    
    def __init__(self, 
                 alpha: Union[float, Iterable[float]] = 0.02,
                 l1_ratio: Union[float, Iterable[float]] = 0.5,
                 criteria_threshold: Tuple[float, float, float] = (0.8, 0.8, 0.99),
                 n_features: int = 5,
                 n_splits: int = 5,
                 n_trials: int = 100,
                 boosting: bool = True,
                 name: Optional[str] = "SupervisedSelection"):
        """Initialize the supervised feature selection step.
        
        Args:
            alpha: Regularization strength (higher values = more regularization).
                Can be a single value or an iterable of values to try.
            l1_ratio: Ratio between L1 and L2 regularization (1.0 = LASSO, 0.0 = Ridge).
                Can be a single value or an iterable of values to try.
            criteria_threshold: Thresholds for the three RENT criteria 
                (frequency, stability, effect size).
            n_features: Maximum number of features to select.
            n_splits: Number of train-test splits to use in RENT.
            n_trials: Number of trials to run for RENT/BRENT.
            boosting: Whether to use boosting (BRENT instead of RENT).
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.criteria_threshold = criteria_threshold
        self.n_features = n_features
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.boosting = boosting
        
        self.model = None
        self._selected_features = None
        self.feature_weights = None
        
    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> 'SupervisedSelectionStep':
        """Fit the supervised feature selection on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, 1)
            X_b: Ignored, included for API consistency
                
        Returns:
            self: The fitted instance
        """
        self._logger.info(f"Running {'BRENT' if self.boosting else 'RENT'} feature selection with {self.n_trials} trials")
        
        # Convert alpha to C parameter for RENT
        C = 1/self.alpha if isinstance(self.alpha, float) else 1/np.asarray(self.alpha)
        
        # Convert l1_ratio to list for RENT's convention
        l1_ratio = np.asarray([self.l1_ratio]) if isinstance(self.l1_ratio, float) else self.l1_ratio
        
        # Extract target values
        try:
            target_col = 'Status' if 'Status' in y.columns else y.columns[0]
            y_values = y[target_col].to_numpy().ravel()
        except:
            y_values = y.to_numpy().ravel()
            
        # Check if y has right format
        num_classes = len(np.unique(y_values))
        if num_classes > 2 and self.boosting:
            self._logger.warning("Multi-class classification detected but boosting is enabled. "
                               "BRENT only supports binary classification. Disabling boosting.")
            self.boosting = False
            
        # Prepare feature names and mapping
        feature_names = [str(col) for col in X.columns]
        feature_map = {feature_name: col for feature_name, col in zip(feature_names, X.columns)}
        
        # Use a single elastic net run if n_trials <= 1
        if self.n_trials <= 1:
            from sklearn.linear_model import ElasticNet
            
            if self.boosting:
                self._logger.warning("n_trials = 1 but boosting was enabled. Disabling boosting.")
                
            # Use ElasticNet directly
            alpha_value = self.alpha[0] if isinstance(self.alpha, (list, tuple, np.ndarray)) else self.alpha
            l1_ratio_value = self.l1_ratio[0] if isinstance(self.l1_ratio, (list, tuple, np.ndarray)) else self.l1_ratio
            
            model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio_value, tol=1E-5)
            
            # Format X for ElasticNet
            X_data = X
            
            # One-hot encode y if multi-class
            if num_classes > 2:
                y_onehot = pd.get_dummies(y_values)
                model.fit(X_data, y_onehot)
                
                # For multi-class, features with non-zero coef in any class are selected
                if model.coef_.ndim == 2:
                    non_zero_idx = np.unique(np.argwhere(model.coef_ != 0)[:, 1])
                else:
                    non_zero_idx = np.argwhere(model.coef_ != 0).ravel()
            else:
                # Binary classification
                model.fit(X_data, y_values)
                non_zero_idx = np.argwhere(model.coef_ != 0).ravel()
            
            # If we have more non-zero features than n_features, rank by coefficient magnitude
            if len(non_zero_idx) > self.n_features:
                self._logger.info(f"Number of non-zero coefficient features is {len(non_zero_idx)}, "
                                "more than the specified max number of features {self.n_features}")
                
                if model.coef_.ndim == 2:
                    coef_magnitude = np.abs(model.coef_[:, non_zero_idx]).mean(axis=0)
                else:
                    coef_magnitude = np.abs(model.coef_[non_zero_idx])
                    
                ordered_idx = np.argsort(coef_magnitude)[::-1]  # Sort in descending order
                selected_idx = non_zero_idx[ordered_idx[:self.n_features]]
            else:
                selected_idx = non_zero_idx
                
            # Get selected feature names
            selected_features = [X.columns[i] for i in selected_idx]
            self._selected_features = selected_features
            
            # Store weights for selected features
            if model.coef_.ndim == 2:
                weights = np.mean(np.abs(model.coef_[:, selected_idx]), axis=0)
            else:
                weights = np.abs(model.coef_[selected_idx])
                
            self.feature_weights = pd.Series(weights, index=selected_features)
            
        else:
            # Use RENT/BRENT
            self.model = RENT.RENT_Regression(
                data=pd.DataFrame(X),
                target=y_values,
                feat_names=feature_names,
                C=[C] if isinstance(C, float) else C,
                l1_ratios=l1_ratio,
                autoEnetParSel=False,
                poly='OFF',
                testsize_range=(1/float(self.n_splits), 1/float(self.n_splits)),
                K=self.n_trials,
                random_state=0,
                verbose=1,
                scale=False,
                boosting=self.boosting
            )
            
            # Train the model
            self.model.train()
            
            # Select features based on criteria
            selected_feature_idx = self.model.select_features(*self.criteria_threshold)
            selected_feature_names = [feature_names[idx] for idx in selected_feature_idx]

            # If we need to limit the number of features, rank by coefficients
            if len(selected_feature_names) > self.n_features:
                # Get coefficients from all models
                coefs_df = pd.DataFrame(
                    np.concatenate(self.model._weight_list, axis=0).T, 
                    index=feature_names
                )
                
                # Filter to selected features only
                coefs_df = coefs_df.loc[selected_feature_names]
                
                # Normalize coefficients
                coefs_df = coefs_df / coefs_df.pow(2).sum().pow(.5)
                
                # Rank by mean and variance
                mean_ranks = (coefs_df.shape[0] - coefs_df.T.mean().argsort())
                var_ranks = coefs_df.T.std().argsort()
                avg_ranks = (mean_ranks + 0.5 * var_ranks) / 1.5
                
                # Get top n_features
                top_features = avg_ranks.sort_values()[:self.n_features].index.tolist()
                selected_feature_names = top_features
                
                # Store feature weights
                self.feature_weights = pd.Series(coefs_df.loc[top_features].mean(axis=1).abs().values, 
                                               index=[feature_map[name] for name in top_features])
            else:
                # Store all selected features
                self.feature_weights = pd.Series(1.0, 
                                              index=[feature_map[name] for name in selected_feature_names])
                
            # Convert feature names back to original indices
            self._selected_features = [feature_map[name] for name in selected_feature_names]
            
        n_total = X.shape[1]
        n_selected = len(self._selected_features)
        self._logger.info(f"Selected {n_selected}/{n_total} features using {'BRENT' if self.boosting else 'RENT'}")
        
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
        
    def get_feature_weights(self) -> pd.Series:
        """Get the weights of selected features.
        
        Returns:
            feature_weights: Series containing weights for each selected feature
        """
        self._check_fitted()
        return self.feature_weights 