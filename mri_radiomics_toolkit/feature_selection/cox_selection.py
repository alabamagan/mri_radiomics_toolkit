"""Cox regression based feature selection.

This module provides a feature selection method based on univariate Cox regression analysis.
Features are selected based on their statistical significance (p-value) in predicting survival outcomes.
"""

import pandas as pd
import numpy as np
import lifelines
from typing import Optional, Union, List, Tuple, Dict, Any
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from .base import FeatureSelectionStep
from tqdm import tqdm
import concurrent.futures
import os
import matplotlib.pyplot as plt

class CoxRegressionSelector(FeatureSelectionStep):
    """Feature selection based on univariate Cox regression analysis.

    This selector performs univariate Cox regression analysis for each feature and selects
    features based on their statistical significance (p-value) in predicting survival outcomes.

    Args:
        p_value_threshold (float): Maximum p-value threshold for feature selection.
        max_features (int, optional): Maximum number of features to select, ranked by p-value.
            If None, all features below p_value_threshold are selected. Defaults to None.
        duration_col (str): Name of the column containing survival time.
        event_col (str): Name of the column containing event status (1=event occurred, 0=censored).
        penalizer (float, optional): L2 penalizer strength for the Cox model. Defaults to 0.0.
        reverse_selection (bool, optional): If True, select features with p-value > threshold.
            Defaults to False.
        alpha (float, optional): Alpha level for confidence intervals. Defaults to 0.05.
        handle_missing (str, optional): How to handle missing values: 'drop', 'impute_mean',
            'impute_median', or 'impute_zero'. Defaults to 'drop'.

    Attributes:
        selected_features (list): Names of the selected features after fitting.
        cox_results (pandas.DataFrame): DataFrame containing Cox regression results for each feature,
            including coefficient, p-value, hazard ratio, and confidence intervals.
        p_values (pandas.Series): Series containing p-values for each feature.

    Notes:
        Input dimensions:
        - X: pandas DataFrame, shape (n_samples, n_features)
          Contains the radiomics features extracted from MRI images.
        - y: pandas DataFrame, shape (n_samples, 2) or pandas Series
          Contains survival information with time-to-event and event indicator.
          If DataFrame, should contain columns for duration and event status.

    Returns:
        pandas.DataFrame: Transformed DataFrame containing only the selected features.

    Raises:
        ValueError: If handle_missing is not one of the supported methods.
        TypeError: If X is not a pandas DataFrame or y doesn't contain required survival information.

    Examples:
        >>> import pandas as pd
        >>> from mri_radiomics_toolkit.feature_selection import CoxRegressionSelector
        >>>
        >>> # Load your features and survival data
        >>> X = pd.DataFrame(...) # Your radiomics features (n_samples, n_features)
        >>> y = pd.DataFrame({'time': [...], 'event': [...]}) # Survival data (n_samples, 2)
        >>>
        >>> # Create a selector
        >>> selector = CoxRegressionSelector(
        ...     p_value_threshold=0.05,
        ...     max_features=10,
        ...     duration_col='time',
        ...     event_col='event'
        ... )
        >>>
        >>> # Fit and transform
        >>> selector.fit(X, y)
        >>> X_selected = selector.transform(X)
        >>>
        >>> # Examine results
        >>> print(f"Selected {len(selector.selected_features)} features")
        >>> print(selector.cox_results.head())
        >>>
        >>> # Visualize results
        >>> fig, ax = selector.plot_pvalues(top_n=20)
        >>> fig.savefig('cox_pvalues.png')
    """

    def __init__(self, 
                 p_value_threshold: float = 0.05,
                 max_features: Optional[int] = None,
                 duration_col: str = 'duration',
                 event_col: str = 'event',
                 penalizer: float = 0.0,
                 reverse_selection: bool = False,
                 alpha: float = 0.05,
                 num_worker: int = 1,
                 name: Optional[str] = None,
                 handle_missing: str = 'drop'):
        """Initialize the Cox regression feature selector.

        Args:
            p_value_threshold (float, optional):
                Maximum p-value threshold for feature selection. Defaults to 0.05.
            max_features (int, optional):
                Maximum number of features to select, ranked by p-value.
                If None, selects all features with p-values below the threshold. Defaults to None.
            duration_col (str, optional):
                Name of the column containing survival time. Defaults to 'duration'.
            event_col (str, optional):
                Name of the column containing event status (1=event, 0=censored). Defaults to 'event'.
            penalizer (float, optional):
                L2 penalizer strength for the Cox model. Larger values increase regularization. Defaults to 0.0.
            reverse_selection (bool, optional):
                If True, select features with p-value > threshold.
                If False, select features with p-value <= threshold. Defaults to False.
            alpha (float, optional):
                Alpha level for confidence intervals. Defaults to 0.05 (95% confidence interval).
            num_worker (int, optional):
                Number of worker threads to use for parallel processing. Defaults to 1 (no parallelism).
            name (str, optional):
                Name for this feature selection step. If None, a default name is used. Defaults to None.
            handle_missing (str, optional):
                How to handle missing values: 'drop', 'impute_mean', 'impute_median', or 'impute_zero'.
                Defaults to 'drop'.

        Returns:
            None

        Raises:
            ValueError: If handle_missing is not one of the supported methods.

        .. notes::
            This method initializes an instance of CoxRegressionSelector, setting parameters for subsequent fit() and transform() operations.
            Cox regression is a common method in survival analysis that can be used to identify features associated with survival outcomes.
        """

        super().__init__(name)
        self.p_value_threshold = p_value_threshold
        self.max_features = max_features
        self.duration_col = duration_col
        self.event_col = event_col
        self.penalizer = penalizer
        self.reverse_selection = reverse_selection
        self.alpha = alpha
        self.num_worker = num_worker
        self.handle_missing = handle_missing
        self._cox_results = None
        
    def _run_cox_for_feature(self, feature, X, y):
        # Prepare data for Cox regression
        data = pd.DataFrame({
            'duration': y[self.duration_col],
            'event': y[self.event_col],
            'feature': X[feature]
        })
        
        # Fit Cox model
        cph = CoxPHFitter(penalizer=self.penalizer, alpha=self.alpha)
        try:
            cph.fit(data, duration_col='duration', event_col='event')
            return {
                'feature': feature,
                'p_value': cph.summary.loc['feature', 'p'],
                'hazard_ratio': cph.summary.loc['feature', 'exp(coef)'],
                'c_index': cph.concordance_index_
            }
        except (ConvergenceError, ConvergenceWarning, ValueError) as e:
            self._logger.warning(f"Cox model fitting failed for feature {feature}: {str(e)}")
            return {
                'feature': feature,
                'p_value': 1.0,
                'hazard_ratio': 1.0,
                'c_index': 0.5
            }

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_b: Optional[pd.DataFrame] = None) -> 'CoxRegressionSelector':
        """Fit the Cox regression selector on the input data.

        TODO:
            - [ ] Need to update the input y so that it matches with other pipeline items
            - [ ] Need to fix a but where sometimes the data is said to be (X, 1) in shape
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values containing survival time and event status
            X_b: Not used in this selector
            
        Returns:
            self: The fitted selector
            
        Raises:
            ValueError: If required survival columns are missing from y
        """
        if y is None:
            raise ValueError("Survival data (y) is required for Cox regression selection")
            
        # Check for required columns
        required_cols = [self.duration_col, self.event_col]
        missing_cols = [col for col in required_cols if col not in y.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in survival data: {missing_cols}")
            
        # Perform univariate Cox regression for each feature
        self._logger.info(f"Performing Cox regression analysis on {len(X.columns)} features")
        cox_results = []
        
        if self.num_worker == 1:
            for feature in X.columns:
                # Prepare data for Cox regression
                self._logger.info(f"Fitting feature: {feature}")
                data = pd.DataFrame({
                    'duration': y[self.duration_col],
                    'event': y[self.event_col],
                    'feature': X[feature]
                })
                
                # Fit Cox model
                cph = CoxPHFitter()
                try:
                    cph.fit(data, duration_col='duration', event_col='event')
                    cox_results.append({
                        'feature': feature,
                        'p_value': cph.summary.loc['feature', 'p'],
                        'hazard_ratio': cph.summary.loc['feature', 'exp(coef)']
                    })
                except (lifelines.exceptions.ConvergenceError, ValueError) as e:
                    self._logger.warning(f"Cox model fitting failed for feature {feature}: {str(e)}")
                    # If model fails to converge, use high p-value
                    cox_results.append({
                        'feature': feature,
                        'p_value': 1.0,
                        'hazard_ratio': 1.0
                    })
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.num_worker, os.cpu_count())) as executor:
                futures = {executor.submit(self._run_cox_for_feature, feature, X, y): feature for feature in X.columns}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Cox regression analysis"):
                    try:
                        result = future.result()
                        cox_results.append(result)
                    except Exception as e:
                        feature = futures[future]
                        self._logger.error(f"Error in Cox analysis for feature {feature}: {str(e)}")
                        cox_results.append({
                            'feature': feature,
                            'p_value': 1.0,
                            'hazard_ratio': 1.0,
                            'c_index': 0.5
                        })
                
        # Convert results to DataFrame and sort by p-value
        self._cox_results = pd.DataFrame(cox_results)
        self._cox_results = self._cox_results.sort_values('p_value')
        
        # Select features based on criteria
        if self.reverse_selection:
            selected = self._cox_results[self._cox_results['p_value'] > self.p_value_threshold]
        else:
            selected = self._cox_results[self._cox_results['p_value'] <= self.p_value_threshold]
        
        if self.max_features is not None:
            selected = selected.head(self.max_features)
            
        self._selected_features = selected['feature'].tolist()
        self._fitted = True
        
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply the feature selection to the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            X_b: Not used in this selector
            
        Returns:
            X_transformed: The transformed features with only selected features
        """
        self._check_fitted()
        return X[self._selected_features]
        
    @property
    def cox_results(self) -> pd.DataFrame:
        """Get the Cox regression results for all features.
        
        Returns:
            DataFrame containing p-values, hazard ratios and other statistics for all features
        """
        if not self._fitted:
            raise ValueError("Selector must be fitted before accessing results")
        return self._cox_results
    
    @property
    def selected_features_stats(self) -> pd.DataFrame:
        """Get statistics only for the selected features.
        
        Returns:
            DataFrame containing statistics for selected features only
        """
        if not self._fitted:
            raise ValueError("Selector must be fitted before accessing results")
        return self._cox_results[self._cox_results['feature'].isin(self._selected_features)]
    
    def plot_pvalues(self, figsize=(10, 6), top_n: Optional[int] = None):
        """Plot p-values for features.
        
        Args:
            figsize: Figure size (width, height)
            top_n: Number of top features to plot, if None plots all
        
        Returns:
            matplotlib figure and axis objects
        """
        if not self._fitted:
            raise ValueError("Must fit the selector before plotting results")
        
        df = self._cox_results.sort_values('p_value')
        if top_n is not None:
            df = df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bars
        bars = ax.barh(df['feature'], -np.log10(df['p_value']))
        
        # Add significance line
        if self.p_value_threshold > 0:
            ax.axvline(-np.log10(self.p_value_threshold), color='red', linestyle='--', 
                       label=f'p-value threshold ({self.p_value_threshold})')
        
        ax.set_xlabel('-log10(p-value)')
        ax.set_ylabel('Feature')
        ax.set_title('Cox Regression Feature Significance')
        ax.legend()
        
        return fig, ax 