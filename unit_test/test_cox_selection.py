"""Unit tests for Cox regression based feature selection."""

import unittest
import pandas as pd
import numpy as np
from mri_radiomics_toolkit.feature_selection import CoxRegressionSelector

class TestCoxRegressionSelector(unittest.TestCase):
    """Test cases for CoxRegressionSelector."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic survival data
        n_samples = 100
        n_features = 5
        
        # Generate features with different survival associations
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),  # Strong association
            'feature2': np.random.normal(0, 1, n_samples),  # Moderate association
            'feature3': np.random.normal(0, 1, n_samples),  # Weak association
            'feature4': np.random.normal(0, 1, n_samples),  # No association
            'feature5': np.random.normal(0, 1, n_samples)   # No association
        })
        
        # Generate survival times with different associations
        survival_times = np.random.exponential(1, n_samples)
        events = np.random.binomial(1, 0.7, n_samples)
        
        # Add feature effects to survival times
        survival_times *= np.exp(
            0.5 * X['feature1'] +  # Strong effect
            0.3 * X['feature2'] +  # Moderate effect
            0.1 * X['feature3']    # Weak effect
        )
        
        y = pd.DataFrame({
            'duration': survival_times,
            'event': events
        })
        
        self.X = X
        self.y = y
        
    def test_initialization(self):
        """Test selector initialization."""
        selector = CoxRegressionSelector()
        self.assertEqual(selector.p_value_threshold, 0.05)
        self.assertIsNone(selector.max_features)
        self.assertEqual(selector.duration_col, 'duration')
        self.assertEqual(selector.event_col, 'event')
        
    def test_fit_without_y(self):
        """Test fit method without survival data."""
        selector = CoxRegressionSelector()
        with self.assertRaises(ValueError):
            selector.fit(self.X)
            
    def test_fit_with_missing_columns(self):
        """Test fit method with missing survival columns."""
        selector = CoxRegressionSelector()
        y_invalid = pd.DataFrame({'wrong_col': [1, 2, 3]})
        with self.assertRaises(ValueError):
            selector.fit(self.X, y_invalid)
            
    def test_fit_and_transform(self):
        """Test fit and transform methods."""
        selector = CoxRegressionSelector(p_value_threshold=0.1)
        selector.fit(self.X, self.y)
        
        # Check if features were selected
        self.assertIsNotNone(selector._selected_features)
        self.assertGreater(len(selector._selected_features), 0)
        
        # Transform data
        X_transformed = selector.transform(self.X)
        self.assertEqual(X_transformed.shape[1], len(selector._selected_features))
        
    def test_max_features(self):
        """Test feature selection with max_features limit."""
        selector = CoxRegressionSelector(p_value_threshold=0.1, max_features=2)
        selector.fit(self.X, self.y)
        
        self.assertLessEqual(len(selector._selected_features), 2)
        
    def test_cox_results(self):
        """Test access to Cox regression results."""
        selector = CoxRegressionSelector()
        selector.fit(self.X, self.y)
        
        results = selector.cox_results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), len(self.X.columns))
        self.assertTrue(all(col in results.columns for col in ['feature', 'p_value', 'hazard_ratio']))
        
    def test_results_ordering(self):
        """Test if results are properly ordered by p-value."""
        selector = CoxRegressionSelector()
        selector.fit(self.X, self.y)
        
        results = selector.cox_results
        p_values = results['p_value'].values
        self.assertTrue(np.all(p_values[:-1] <= p_values[1:]))  # Check if sorted ascending
        
    def test_transform_without_fit(self):
        """Test transform method without fitting."""
        selector = CoxRegressionSelector()
        with self.assertRaises(ValueError):
            selector.transform(self.X)
            
    def test_custom_column_names(self):
        """Test with custom survival column names."""
        y_custom = self.y.rename(columns={
            'duration': 'time',
            'event': 'status'
        })
        
        selector = CoxRegressionSelector(
            duration_col='time',
            event_col='status'
        )
        selector.fit(self.X, y_custom)
        
        self.assertIsNotNone(selector._selected_features)
        self.assertGreater(len(selector._selected_features), 0)

    def test_save_and_load(self):
        """Test saving and loading the selector."""
        import tempfile
        from pathlib import Path
        
        # Create and fit a selector
        selector = CoxRegressionSelector(p_value_threshold=0.1)
        selector.fit(self.X, self.y)
        
        # Get original selected features
        original_features = selector.selected_features.copy()
        
        # Save to a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "cox_selector.tar.gz"
            selector.save(filepath)
            
            # Load into a new selector
            loaded_selector = CoxRegressionSelector.load(filepath)
            
            # Check that loaded selector has the same properties
            self.assertEqual(loaded_selector.p_value_threshold, selector.p_value_threshold)
            self.assertEqual(loaded_selector.max_features, selector.max_features)
            self.assertEqual(loaded_selector.duration_col, selector.duration_col)
            self.assertEqual(loaded_selector.event_col, selector.event_col)
            
            # Check selected features are the same
            self.assertListEqual(loaded_selector.selected_features, original_features)
            
            # Check if transform works on loaded selector
            X_transformed = loaded_selector.transform(self.X)
            self.assertEqual(X_transformed.shape[1], len(original_features))

    def test_all_features_insignificant(self):
        """Test behavior when all features are insignificant."""
        # Create data with no association
        np.random.seed(42)
        X_random = pd.DataFrame({
            f'rand_feature{i}': np.random.normal(0, 1, len(self.y)) 
            for i in range(5)
        })
        
        # Set very stringent p-value threshold
        selector = CoxRegressionSelector(p_value_threshold=1e-10)
        selector.fit(X_random, self.y)

        # Print Cox results
        selector._logger.debug(f"{selector._cox_results}")
        
        # Should have no features selected
        self.assertEqual(len(selector.selected_features), 0)
        
        # Transform should return empty DataFrame
        X_transformed = selector.transform(X_random)
        self.assertEqual(X_transformed.shape[1], 0)
        
    def test_with_missing_values(self):
        """Test with missing values in the data."""
        # Create data with missing values
        X_missing = self.X.copy()
        # Add some NaN values
        X_missing.loc[0:5, 'feature1'] = np.nan
        
        # Test with different missing value handling methods
        for method in ['drop', 'impute_mean', 'impute_median', 'impute_zero']:
            selector = CoxRegressionSelector(handle_missing=method)
            selector.fit(X_missing, self.y)
            
            # Should still select some features
            self.assertGreater(len(selector.selected_features), 0)

if __name__ == '__main__':
    unittest.main() 