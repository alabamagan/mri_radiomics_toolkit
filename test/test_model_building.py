import unittest
import os
import subprocess
import tempfile
import shutil
import pprint
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


    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()

    def setUp(self) -> None:
        self._logger = MNTSLogger['Test_ModelBuilding']
        self._globber = "^[0-9]+"
        self._p_setting = Path('test_data/assets/sample_pyrad_settings.yml')
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

    def test_model_building(self):
        from sklearn.model_selection import train_test_split

        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        p_fss = Path('test_data/assets/fs_saved_state.fss')

        features = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)
        cases = list(set(features.index) & set(gt.index))
        gt = gt.loc[cases]
        features = features.loc[cases]

        with tempfile.NamedTemporaryFile('wb', suffix='.pkl') as f:
            fs = FeatureSelector()
            fs.load(p_fss)
            features = fs.predict(features)
            self._logger.info(f"Selected features are: {features.T}")

            # Random train test split
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
        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        p_fss = Path('test_data/assets/fs_saved_state.fss')

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
            cv_grid_search(features.iloc[:20], gt.iloc[:20], None, None))

    def test_cv_grid_search_multi_class(self):
        from mri_radiomics_toolkit.models.cards import multi_class_cv_grid_search_card
        from functools import partial
        # Setup the configurations
        features = pd.DataFrame(np.random.random(size=(40, 100)))
        gt = pd.Series(np.random.randint(low=0, high=3, size=40))

        # scoring function
        def scoring(estimator, X, y):
            try:
                pred = estimator.predict_proba(X)
            except AttributeError:
                pred = estimator.predict(X)
            return -sklearn.metrics.log_loss(y, pred, labels=np.unique(y))

        # Call the function
        best_params, results, predict_table, best_estimators = (
            cv_grid_search(features, gt, None, None,
                           param_grid_dict=multi_class_cv_grid_search_card, scoring=scoring))


if __name__ == '__main__':
    unittest.main()
