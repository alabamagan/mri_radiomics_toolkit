import unittest
import os
import subprocess
import tempfile
import shutil
import pprint
import time
from pathlib import Path

import sklearn.model_selection
from mnts.mnts_logger import MNTSLogger
from mri_radiomics_toolkit import *
from mri_radiomics_toolkit.model_building import cv_grid_search
from mri_radiomics_toolkit.perf_metric import getStability, confidenceIntervals, hypothesisTestT, \
    hypothesisTestV, feat_list_to_binary_mat
import numpy as np
import pandas as pd


class Test_ModelBuilding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cls_logger = MNTSLogger(".", "Test_ModelBuilding", verbose=True, keep_file=False,
                                    log_level='debug')
        MNTSLogger.set_global_log_level('debug')
        time.sleep(1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()

    def setUp(self) -> None:
        self._logger = MNTSLogger['Test_ModelBuilding']
        self._globber = "^[0-9]+"
        self._p_setting = Path('test_data/assets/sample_pyrad_settings.yml')
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

        # Check if the setting file exists
        self.assertTrue(self._p_setting.exists(), f"Settings file not found: {self._p_setting}")

    @unittest.skip("Long test")
    def test_model_building(self):
        from sklearn.model_selection import train_test_split

        p_feat_a, p_gt, p_fss = self._prepare_samples()

        features = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)
        cases = list(set(features.index) & set(gt.index))
        gt = gt.loc[cases]
        features = features.loc[cases]

        with tempfile.NamedTemporaryFile('wb', suffix='.pkl') as f:
            from mri_radiomics_toolkit.feature_selection import FeatureSelectionPipeline
            fs = FeatureSelectionPipeline.load(p_fss)
            features = fs.transform(features)
            self._logger.info(f"Selected features are: {features.T}")

            # Random train unit_test split
            splitter = train_test_split(features.index, test_size=0.2)
            train_feats, test_feats = splitter
            self._logger.info(f"Training group: {train_feats}")
            self._logger.info(f"Testing group: {test_feats}")

            self._logger.info("{:-^50s}".format(" Building model "))
            model = ModelBuilder()
            # Test model building with testing data
            try:
                results, predict_table = model.fit(features.loc[train_feats], gt.loc[train_feats],
                                                   features.loc[test_feats], gt.loc[test_feats])
            except:
                self._logger.warning("Fitting with testing data failed!")
                predict_table = "Error"
            # Test model building without testing data
            try:
                results, _ = model.fit(features.loc[train_feats], gt.loc[train_feats])
            except Exception as e:
                self._logger.warning("Fitting without testing data failed!")
                self._logger.exception(f"{e}")
            self._logger.info(f"Results: {pprint.pformat(results)}")
            self._logger.info(f"Predict_table: {predict_table}")
            self._logger.info(f"Best params: {pprint.pformat(model.saved_state)}")

            # Test save functionality
            self._logger.info("{:-^50s}".format(" Testing model save/load "))
            model.save(Path(f.name))
            # Test load functionality
            _model = ModelBuilder()
            _model.load(Path(f.name))
            self._logger.debug(f"Saved state: {pprint.pformat(_model.saved_state)}")
            _predict_table = _model.predict(features.loc[test_feats])

            self._logger.info(f"Left:\n {_predict_table}")
            self._logger.info(f"Right:\n {predict_table}")
        pass

    def test_cv_grid_search(self):
        # Setup the configurations
        p_feat_a, p_gt, p_fss = self._prepare_samples()

        # Load the data
        features = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)
        cases = list(set(features.index) & set(gt.index))
        gt = gt.loc[cases]
        features = features.loc[cases]

        # Drop diagnostics
        features.drop(['diagnostics'], axis=1, inplace=True)

        # Call the function
        best_params, results, predict_table, best_estimators = (
            cv_grid_search(features.iloc[:20], gt.iloc[:20], None, None, scoring='roc_auc', n_jobs=8))

    def _prepare_samples(self):
        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_gt = Path('test_data/assets/sample_datasheet_complex.csv')
        p_fss = Path('test_data/assets/fs_saved_state.fss')

        # Check if all required files exist
        self.assertTrue(p_feat_a.exists(), f"Feature file not found: {p_feat_a}")
        self.assertTrue(p_gt.exists(), f"Ground truth file not found: {p_gt}")
        self.assertTrue(p_fss.exists(), f"Feature selector state file not found: {p_fss}")

        return p_feat_a, p_gt, p_fss

    @unittest.skip("This is broken now.")
    def test_cv_grid_search_with_smote(self):
        r"""Test CV grid search with SMOTE"""
        from mri_radiomics_toolkit.data_split import generate_cross_validation_samples
        # Setup the configurations
        p_feat_a, p_gt, p_fss = self._prepare_samples()

        # Load the data
        features = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)
        cases = list(set(features.index) & set(gt.index))
        gt = gt.loc[cases]
        features = features.loc[cases]

        # Drop diagnostics
        features.drop(['diagnostics'], axis=1, inplace=True)

        # Call the function
        best_params, results, predict_table, best_estimators = (
            cv_grid_search(features.iloc[:20], gt.iloc[:20], None, None,
                           cv=generate_cross_validation_samples))

    def test_cv_grid_search_multi_class(self):
        from mri_radiomics_toolkit.models.cards import multi_class_cv_grid_search_card
        from mri_radiomics_toolkit.model_building import neg_log_loss
        from functools import partial
        # Setup the configurations
        features = pd.DataFrame(np.random.random(size=(40, 100)))
        gt = pd.Series(np.random.randint(low=0, high=3, size=40))

        # Call the function
        best_params, results, predict_table, best_estimators = (
            cv_grid_search(features, gt, None, None,
                           param_grid_dict=multi_class_cv_grid_search_card, scoring=neg_log_loss))

    def test_save_load_mechanism(self):
        """Test the save/load mechanism for ModelBuilder"""
        import tempfile
        from pathlib import Path
        
        # Setup the configurations
        features = pd.DataFrame(np.random.random(size=(40, 10)))
        gt = pd.Series(np.random.randint(low=0, high=2, size=40))
        
        # Create a model builder and fit it
        model = ModelBuilder()
        results, _ = model.fit(features, gt)
        
        # Test save/load with the new mechanism
        with tempfile.TemporaryDirectory() as tempdir:
            save_dir = Path(tempdir) / 'model_state'
            
            # Save the model
            model.save(save_dir)
            
            # Create a new model and load the state
            new_model = ModelBuilder()
            new_model.load(save_dir)
            
            # Check if states match
            self.assertEqual(model.saved_state['best_params'], new_model.saved_state['best_params'])
            
            # Check if the loaded model can predict
            predictions1 = model.predict(features)
            predictions2 = new_model.predict(features)
            pd.testing.assert_frame_equal(predictions1, predictions2)


if __name__ == '__main__':
    unittest.main()
