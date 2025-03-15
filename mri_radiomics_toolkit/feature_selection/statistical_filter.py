"""Statistical filters for feature selection.

This module provides feature selection steps that filter features based on statistical tests.
These include t-test for binary classification problems and ANOVA for multi-class problems.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List, Dict

import pingouin as pg
from scipy.stats import shapiro, levene, f_oneway

from .base import FeatureSelectionStep


class TTestFilterStep(FeatureSelectionStep):
    """Filter features based on t-test p-values.
    
    This step is used for binary classification problems to select features that show
    statistically significant differences between the two classes.
    """
    
    def __init__(self, threshold: float = 0.05, name: Optional[str] = "TTestFilter"):
        """Initialize the t-test filter step.
        
        Args:
            threshold: p-value threshold for filtering features. Features with p-values
                less than or equal to this threshold will be kept. Default is 0.05.
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.p_values = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> 'TTestFilterStep':
        """Fit the t-test filter on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, 1)
            X_b: Optional second set of features, if provided both feature sets must pass
                the t-test threshold to be selected
                
        Returns:
            self: The fitted instance
            
        Raises:
            ValueError: If y contains more than two classes.
        """
        # Check if the target has exactly two classes
        try:
            target_col = 'Status' if 'Status' in y.columns else y.columns[0]
            y_values = y[target_col]
        except:
            y_values = y
            
        unique_classes = np.unique(y_values)
        if len(unique_classes) != 2:
            raise ValueError(f"TTestFilter requires exactly two classes in the target. Found {len(unique_classes)} classes.")
            
        self._logger.info(f"Filtering features with t-test p-value threshold {self.threshold}")
        
        # Get indices of samples in each class
        class_indices = {c: y_values[y_values == c].index.tolist() for c in unique_classes}
        
        # Calculate t-test p-values for each feature
        p_values = []
        test_names = []
        
        for feature in X.columns:
            # Extract feature values for each class
            values_class1 = X.loc[class_indices[unique_classes[0]], feature].astype('float32')
            values_class2 = X.loc[class_indices[unique_classes[1]], feature].astype('float32')
            
            # Check normality of the feature in each class
            normality_pvals = [pg.normality(values)['pval'].values[0] for values in [values_class1, values_class2]]
            is_normal = min(normality_pvals) > 0.05
            
            # Choose the appropriate test based on normality
            if is_normal:
                # Use Student's t-test for normally distributed data
                test_result = pg.ttest(values_class1, values_class2)
                p_value = test_result['p-val'].values[0].astype('float')
                test_name = 'Student t-test'
            else:
                # Use Mann-Whitney U test for non-normally distributed data
                test_result = pg.mwu(values_class1, values_class2)
                p_value = test_result['p-val'].values[0].astype('float')
                test_name = 'Mann-Whitney U'
                
            p_values.append(p_value)
            test_names.append(test_name)
            
        # Create DataFrame with results
        self.p_values = pd.DataFrame({
            'p_value': p_values,
            'test': test_names
        }, index=X.columns)
        
        # Select features that pass the p-value threshold
        self._selected_features = self.p_values[self.p_values['p_value'] <= self.threshold].index
        
        # If X_b is provided, apply the same test and keep only features that pass in both sets
        if X_b is not None:
            # Calculate t-test p-values for X_b
            p_values_b = []
            test_names_b = []
            
            for feature in self._selected_features:  # Only test features selected from X
                # Extract feature values for each class
                values_class1 = X_b.loc[class_indices[unique_classes[0]], feature].astype('float32')
                values_class2 = X_b.loc[class_indices[unique_classes[1]], feature].astype('float32')
                
                # Check normality
                normality_pvals = [pg.normality(values)['pval'].values[0] for values in [values_class1, values_class2]]
                is_normal = min(normality_pvals) > 0.05
                
                if is_normal:
                    test_result = pg.ttest(values_class1, values_class2)
                    p_value = test_result['p-val'].values[0].astype('float')
                    test_name = 'Student t-test'
                else:
                    test_result = pg.mwu(values_class1, values_class2)
                    p_value = test_result['p-val'].values[0].astype('float')
                    test_name = 'Mann-Whitney U'
                    
                p_values_b.append(p_value)
                test_names_b.append(test_name)
                
            # Create DataFrame with results for X_b
            if p_values_b:
                p_values_df_b = pd.DataFrame({
                    'p_value': p_values_b,
                    'test': test_names_b
                }, index=self._selected_features)
                
                # Update selected features to only include those that pass in both X and X_b
                self._selected_features = p_values_df_b[p_values_df_b['p_value'] <= self.threshold].index
        
        n_total = X.shape[1]
        n_selected = len(self._selected_features)
        self._logger.info(f"Selected {n_selected}/{n_total} features with p-value <= {self.threshold}")
        
        self._fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the t-test filter to the input data.
        
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


class ANOVAFilterStep(FeatureSelectionStep):
    """Filter features based on ANOVA p-values.
    
    This step is used for multi-class problems to select features that show
    statistically significant differences between three or more classes.
    """
    
    def __init__(self, threshold: float = 0.05, name: Optional[str] = "ANOVAFilter"):
        """Initialize the ANOVA filter step.
        
        Args:
            threshold: p-value threshold for filtering features. Features with p-values
                less than or equal to this threshold will be kept. Default is 0.05.
            name: Optional name for this step.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.p_values = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> 'ANOVAFilterStep':
        """Fit the ANOVA filter on the input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, ) or (n_samples, 1)
            X_b: Optional second set of features, if provided both feature sets must pass
                the ANOVA threshold to be selected
                
        Returns:
            self: The fitted instance
            
        Raises:
            ValueError: If y contains fewer than three classes.
        """
        # Check if the target has at least three classes
        try:
            target_col = 'Status' if 'Status' in y.columns else y.columns[0]
            y_values = y[target_col]
        except:
            y_values = y
            
        unique_classes = np.unique(y_values)
        if len(unique_classes) < 3:
            raise ValueError(f"ANOVAFilter requires at least three classes in the target. Found {len(unique_classes)} classes.")
            
        self._logger.info(f"Filtering features with ANOVA p-value threshold {self.threshold}")
        
        # Group data by class
        groups = {}
        for cls in unique_classes:
            class_indices = y_values[y_values == cls].index
            groups[cls] = X.loc[class_indices]
            
        # Initialize results storage
        p_values = []
        test_names = []
        
        # Test each feature
        for feature in X.columns:
            # Check normality assumption for each group
            is_normal = True
            for group_data in groups.values():
                _, p_value = shapiro(group_data[feature])
                if p_value <= 0.05:
                    is_normal = False
                    break
                    
            # Check equal variance assumption
            feature_groups = [group_data[feature] for group_data in groups.values()]
            center = 'median' if not is_normal else 'mean'
            _, var_p_value = levene(*feature_groups, center=center)
            has_equal_var = var_p_value > 0.05
            
            # Choose appropriate test
            if is_normal and has_equal_var:
                # Use one-way ANOVA
                _, p_value = f_oneway(*feature_groups)
                test_name = 'One-way ANOVA'
            else:
                # Use Kruskal-Wallis H-test
                kruskal_data = pd.DataFrame()
                for i, group_data in enumerate(feature_groups):
                    temp_df = pd.DataFrame({'value': group_data.astype('float'), 'group': i})
                    kruskal_data = pd.concat([kruskal_data, temp_df])

                test_result = pg.kruskal(data=kruskal_data, dv='value', between='group')
                p_value = test_result['p-unc'].values[0]
                test_name = 'Kruskal-Wallis H'
                
            p_values.append(p_value)
            test_names.append(test_name)
            
        # Create DataFrame with results
        self.p_values = pd.DataFrame({
            'p_value': p_values,
            'test': test_names
        }, index=X.columns)
        
        # Select features that pass the p-value threshold
        self._selected_features = self.p_values[self.p_values['p_value'] <= self.threshold].index
        
        # If X_b is provided, apply the same test and keep only features that pass in both sets
        if X_b is not None:
            # Group X_b data by class
            groups_b = {}
            for cls in unique_classes:
                class_indices = y_values[y_values == cls].index
                groups_b[cls] = X_b.loc[class_indices]
                
            # Test each selected feature in X_b
            p_values_b = []
            test_names_b = []
            
            for feature in self._selected_features:
                # Check normality assumption for each group
                is_normal = True
                for group_data in groups_b.values():
                    _, p_value = shapiro(group_data[feature])
                    if p_value <= 0.05:
                        is_normal = False
                        break
                        
                # Check equal variance assumption
                feature_groups = [group_data[feature] for group_data in groups_b.values()]
                center = 'median' if not is_normal else 'mean'
                _, var_p_value = levene(*feature_groups, center=center)
                has_equal_var = var_p_value > 0.05
                
                # Choose appropriate test
                if is_normal and has_equal_var:
                    # Use one-way ANOVA
                    _, p_value = f_oneway(*feature_groups)
                    test_name = 'One-way ANOVA'
                else:
                    # Use Kruskal-Wallis H-test
                    test_result = pg.kruskal(*feature_groups)
                    p_value = test_result['p-unc'].values[0]
                    test_name = 'Kruskal-Wallis H'
                    
                p_values_b.append(p_value)
                test_names_b.append(test_name)
                
            # Create DataFrame with results for X_b
            if p_values_b:
                p_values_df_b = pd.DataFrame({
                    'p_value': p_values_b,
                    'test': test_names_b
                }, index=self._selected_features)
                
                # Update selected features to only include those that pass in both X and X_b
                self._selected_features = p_values_df_b[p_values_df_b['p_value'] <= self.threshold].index
        
        n_total = X.shape[1]
        n_selected = len(self._selected_features)
        self._logger.info(f"Selected {n_selected}/{n_total} features with p-value <= {self.threshold}")
        
        self._fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, X_b: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Apply the ANOVA filter to the input data.
        
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