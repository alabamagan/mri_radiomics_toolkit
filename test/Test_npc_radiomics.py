import unittest
import os
import subprocess
import tempfile
import shutil
import pprint
from pathlib import Path

import numpy as np


class Test_pipeline(unittest.TestCase):
    def test_feature_extractor_with_norm(self):
        from npc_radiomics.feature_extractor import FeatureExtractor
        from mnts.mnts_logger import MNTSLogger
        globber = "^[0-9]+"
        p_im = Path('../samples/images_not_normalized/')
        p_seg = Path('../samples/segment/')
        p_setting = Path('../samples/sample_pyrad_settings.yml')

        with MNTSLogger('./', keep_file=False, verbose=True) as logger, \
             tempfile.TemporaryDirectory() as f:
            # Create feature extractor
            logger.info("{:-^50s}".format(" Testing feature extraction "))
            fe = FeatureExtractor(id_globber=globber)
            fe.param_file = p_setting
            df = fe.extract_features_with_norm(p_im, p_seg, param_file = p_setting)
            fe.save_features(p_setting.with_name('sample_features.xlsx'))
            logger.info("\n" + df.to_string())
            self.assertTrue(len(df) > 0)
            self.assertTrue(p_setting.with_name('sample_features.xlsx').is_file())
            logger.info("Feature extraction pass...")

            # test save state
            logger.info("{:-^50s}".format(" Testing save state "))
            fe.save(Path('../samples/fe_saved_state.fe'))
            fe.save(Path(f))
            self.assertTrue(Path(f).joinpath('saved_state.fe').is_file())

            # test load state
            logger.info("{:-^50s}".format(" Testing load state "))
            _fe = FeatureExtractor(id_globber=globber)
            _fe.load(Path(f).joinpath('saved_state.fe'))
            _df = fe.extract_features_with_norm(p_im, p_seg)

            logger.info(f"Left:\n {_df.to_string()}")
            logger.info(f"Right:\n {df.to_string()}")

    def test_feature_extractor(self):
        from npc_radiomics.feature_extractor import FeatureExtractor
        from mnts.mnts_logger import MNTSLogger
        globber = "^[0-9]+"
        p_im = Path('../samples/images/')
        p_seg = Path('../samples/segment')
        p_setting = Path('../samples/sample_pyrad_settings.yml')

        logger = MNTSLogger('./', keep_file=False, verbose=True, log_level='debug')
        with tempfile.TemporaryDirectory() as f:
            # Create feature extractor
            logger.info("{:-^50s}".format(" Testing feature extraction "))
            fe = FeatureExtractor(id_globber=globber)
            fe.param_file = p_setting
            df = fe.extract_features(p_im, p_seg, p_setting)
            fe.save_features(p_setting.with_name('sample_features.xlsx'))
            logger.info("\n" + df.to_string())
            self.assertTrue(len(df) > 0)
            self.assertTrue(p_setting.with_name('sample_features.xlsx').is_file())
            logger.info("Feature extraction pass...")

            # test save state
            logger.info("{:-^50s}".format(" Testing save state "))
            fe.save(Path(f))
            self.assertTrue(Path(f).joinpath('saved_state.fe').is_file())

            # test load state
            logger.info("{:-^50s}".format(" Testing load state "))
            _fe = FeatureExtractor(id_globber=globber)
            _fe.load(Path(f).joinpath('saved_state.fe'))
            _df = fe.extract_features(p_im, p_seg, p_setting)

            logger.info(f"Left:\n {_df.to_string()}")
            logger.info(f"Right:\n {df.to_string()}")

    def test_feature_extractor_param_file_load(self):
        from npc_radiomics.feature_extractor import FeatureExtractor
        from mnts.mnts_logger import MNTSLogger
        globber = "^[0-9]+"
        p_im = Path('../samples/images/')
        p_seg = Path('../samples/segment')
        p_setting = Path('../samples/sample_pyrad_settings.yml')

        with MNTSLogger('./', keep_file=False, verbose=True, log_level='debug') as logger:
            fe = FeatureExtractor(id_globber="^[0-9]+")
            fe.param_file = p_setting

            logger.info(f"Processed setting: {fe.param_file}")
            self.assertFalse(fe.param_file == p_setting)
            self.assertTrue(fe.param_file == p_setting.read_text())

    def test_feature_selection(self):
        from npc_radiomics.feature_selection import FeatureSelector
        from mnts.mnts_logger import MNTSLogger
        import pandas as pd

        globber = "^[0-9]+"
        p_feat_a = Path('../samples/samples_feat_1st.xlsx')
        p_feat_b = Path('../samples/samples_feat_2nd.xlsx')
        p_gt = Path('../samples/sample_datasheet.csv')

        features_a = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        features_b = pd.read_excel(str(p_feat_b), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)

        cases = set(features_a.index) & set(gt.index)
        gt = gt.loc[cases]

        with MNTSLogger('./default.log', keep_file=False, verbose=True) as logger,\
                tempfile.NamedTemporaryFile('wb', suffix = '.fss') as f:
            fs = FeatureSelector(n_trials=50, boot_runs=5) # Use default criteria, test with boosting

            # Test one segmentation
            logger.info("{:-^50s}".format(" Testing single feature set "))
            # feats = fs.fit(features_a, gt)

            # Test two segmentation
            logger.info("{:-^50s}".format(" Testing pair feature set "))
            cases = set(features_a.index) & set(features_b.index) & set(gt.index)
            feats = fs.fit(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])

            # Testing save and load function
            logger.info("{:-^50s}".format(" Testing state save/load "))
            fs.save(Path(f.name))
            _new_fs = FeatureSelector()
            _new_fs.load(Path(f.name))
            _feats = _new_fs.predict(features_a)

            logger.info(f"Left:\n {_feats.T}")
            logger.info(f"Right:\n {feats[0].T}")
            logger.info(f"Save/load (Passed)")

        pass

    def test_model_building(self):
        from npc_radiomics.feature_selection import FeatureSelector
        from npc_radiomics.model_building import ModelBuilder
        from mnts.mnts_logger import MNTSLogger
        from sklearn.model_selection import train_test_split
        import pandas as pd

        globber = "^[0-9]+"
        p_feat_a = Path('../samples/samples_feat_1st.xlsx')
        p_gt = Path('../samples/sample_datasheet.csv')
        p_fss = Path('../samples/fs_saved_state.fss')

        features = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)
        cases = set(features.index) & set(gt.index)
        gt = gt.loc[cases]
        features = features.loc[cases]

        with MNTSLogger('./default.log', keep_file=False, verbose=True, log_level='debug') as logger, \
             tempfile.NamedTemporaryFile('wb', suffix='.pkl') as f:
            fs = FeatureSelector()
            fs.load(p_fss)
            features = fs.predict(features)
            logger.info(f"Selected features are: {features.T}")

            # Random train test split
            splitter = train_test_split(features.index, test_size=0.2)
            train_feats, test_feats = splitter
            logger.info(f"Training group: {train_feats}")
            logger.info(f"Testing group: {test_feats}")

            logger.info("{:-^50s}".format(" Building model "))
            model = ModelBuilder()
            # Test model building with testing data
            try:
                results, predict_table = model.fit(features.loc[train_feats], gt.loc[train_feats],
                                                   features.loc[test_feats], gt.loc[test_feats])
            except:
                logger.warning("Fitting with testing data failed!")
            # Test model building without testing data
            try:
                results, predict_table = model.fit(features.loc[train_feats], gt.loc[train_feats])
            except Exception as e:
                logger.warning("Fitting without testing data failed!")
                logger.exception(f"{e}")
            logger.info(f"Results: {pprint.pformat(results)}")
            logger.info(f"Predict_table: {predict_table.to_string()}")
            logger.info(f"Best params: {pprint.pformat(model.saved_state)}")

            # Test save functionality
            logger.info("{:-^50s}".format(" Testing model save/load "))
            model.save(Path(f.name))
            # Test load functionality
            _model = ModelBuilder()
            _model.load(Path(f.name))
            logger.info(f"Estimator: {pprint.pformat(_model.saved_state)}")
            _predict_table = _model.predict(features.loc[test_feats])

            logger.info(f"Left:\n {_predict_table}")
            logger.info(f"Right:\n {predict_table}")
        pass


    def test_controller_extraction(self):
        from npc_radiomics.controller import Controller
        from mnts.mnts_logger import MNTSLogger

        p = Path('../samples/sample_controller_settings.yml')
        p_im = Path('../samples/images_not_normalized/')
        p_seg = Path('../samples/segment/')
        p_gt = Path('../samples/sample_datasheet.csv')
        p_pyrad = Path('../samples/sample_pyrad_settings.yml')
        p_fe_state = Path('../samples/fe_saved_state.fe')

        # extract feature was ported to the controller, test it
        with MNTSLogger('./default.log', verbose=True, keep_file=False) as logger:
            ctl = Controller(setting=p, with_norm=True)
            ctl.load_norm_settings(fe_state=p_fe_state)
            df = ctl.extract_feature(p_im, p_seg, py_rad_param_file=p_pyrad)
            logger.info(f"features {df}")
        pass

    def test_controller_load_norm(self):
        from npc_radiomics.controller import Controller
        from mnts.mnts_logger import MNTSLogger

        p = Path('../samples/sample_controller_settings.yml')
        p_norm_state = Path('../assets/t2wfs/')
        p_norm_graph = Path('../assets/t2wfs/norm_graph.yml')
        p_fe_state = Path('../samples/fe_saved_state.fe')

        with MNTSLogger('./default.log', verbose=True, keep_file=False) as logger:
            ctl = Controller(setting=p, with_norm=True)
            ctl.load_norm_settings(norm_graph=p_norm_graph, norm_state_file=p_norm_state)
            logger.info(f"State 1: \n{pprint.pformat(ctl.extractor.saved_state)}")

            ctl.load_norm_settings(fe_state=p_fe_state)
            logger.info(f"State 2: \n{pprint.pformat(ctl.extractor.saved_state)}")

    def test_controller_fit(self):
        from npc_radiomics.controller import Controller
        from mnts.mnts_logger import MNTSLogger

        p = Path('../samples/sample_controller_settings.yml')
        p_im = Path('../samples/images_not_normalized/')
        p_seg = Path('../samples/segment/')
        p_gt = Path('../samples/sample_datasheet.csv')
        p_pyrad = Path('../samples/sample_pyrad_settings.yml')
        p_fe_state = Path('../samples/fe_saved_state.fe')

        # extract feature was ported to the controller, test it
        with MNTSLogger('./default.log', verbose=True, keep_file=True, log_level='debug') as logger:
            ctl = Controller(setting=p, with_norm=True)
            ctl.load_norm_settings(fe_state=p_fe_state)
            ctl.fit(p_im, p_seg, p_gt)

if __name__ == '__main__':
    te = Test_pipeline()
    te.test_controller_fit()
    # te.test_feature_extractor()