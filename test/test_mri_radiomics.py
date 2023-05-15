import unittest
import os
import subprocess
import tempfile
import shutil
import pprint
from pathlib import Path
from mnts.mnts_logger import MNTSLogger
from mri_radiomics_toolkit import *
from mri_radiomics_toolkit.perf_metric import getStability, confidenceIntervals, hypothesisTestT, \
            hypothesisTestV, feat_list_to_binary_mat
import numpy as np
import pandas as pd


class Test_pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cls_logger = MNTSLogger(".", "Test_pipe", verbose=True, keep_file=False, 
                                    log_level='debug')
    
    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()
        
    def setUp(self) -> None:
        self._logger = MNTSLogger[unittest]
        self._globber = "^[0-9]+"
        self._p_setting = Path('test_data/assets/sample_pyrad_settings.yml')
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_feature_extractor_w_norm(self):
        p_im = Path('test_data/images_not_normalized/')
        p_seg = Path('test_data/segment/')

        # Create feature extractor
        self._logger.info("{:-^50s}".format(" Testing feature extraction "))
        fe = FeatureExtractor(id_globber=self._globber)
        fe.param_file = self._p_setting
        df = fe.extract_features_with_norm(p_im, p_seg, param_file = self._p_setting)
        fe.save_features(self._p_setting.with_name('sample_features.xlsx'))
        self._logger.info("\n" + df.to_string())
        self.assertTrue(len(df) > 0)
        self.assertTrue(self._p_setting.with_name('sample_features.xlsx').is_file())
        self._logger.info("Feature extraction pass...")

        # test save state
        self._logger.info("{:-^50s}".format(" Testing save state "))
        fe.save(Path('test_data/assets/fe_saved_state.fe'))
        fe.save(Path(self.temp_dir))
        self.assertTrue(Path(self.temp_dir).joinpath('saved_state.fe').is_file())

        # test load state
        self._logger.info("{:-^50s}".format(" Testing load state "))
        _fe = FeatureExtractor(id_globber=self._globber)
        _fe.load(Path(self.temp_dir).joinpath('saved_state.fe'))
        _df = fe.extract_features_with_norm(p_im, p_seg)

        self._logger.info(f"Left:\n {_df.to_string()}")
        self._logger.info(f"Right:\n {df.to_string()}")

    def test_feature_extractor(self):
        p_im = Path('test_data/images/')
        p_seg = Path('test_data/segment')

        # Create feature extractor
        self._logger.info("{:-^50s}".format(" Testing feature extraction "))
        fe = FeatureExtractor(id_globber=self._globber, idlist=['1130', '1131'])
        fe.param_file = self._p_setting
        df = fe.extract_features(p_im, p_seg, param_file=self._p_setting)
        fe.save_features(self._p_setting.with_name('sample_features.xlsx'))
        self._logger.info("\n" + df.to_string())
        self.assertTrue(len(df) > 0)
        self.assertTrue(self._p_setting.with_name('sample_features.xlsx').is_file())
        self._logger.info("Feature extraction pass...")

        # test save state
        self._logger.info("{:-^50s}".format(" Testing save state "))
        fe.save(Path(self.temp_dir))
        self.assertTrue(Path(self.temp_dir).joinpath('saved_state.fe').is_file())

        # test load state
        self._logger.info("{:-^50s}".format(" Testing load state "))
        _fe = FeatureExtractor(id_globber=self._globber)
        _fe.load(Path(self.temp_dir).joinpath('saved_state.fe'))
        self._logger.debug(f"Loaded state: {_fe.saved_state}")
        _df = fe.extract_features(p_im, p_seg)

        # Display tested itmes
        self._logger.info(f"Left:\n {_df.to_string()}")
        self._logger.info(f"Right:\n {df.to_string()}")

    def test_feature_extractor_param_file_load(self):
        from mri_radiomics_toolkit.feature_extractor import FeatureExtractor
        p_im = Path('test_data/images/')
        p_seg = Path('test_data/segment')
        self._p_setting = Path('test_data/assets/sample_pyrad_settings.yml')

        fe = FeatureExtractor(id_globber="^[0-9]+")
        fe.param_file = self._p_setting

        self._logger.info(f"Processed setting: {fe.param_file}")
        self._logger.info(f"Saved state: {pprint.pformat(fe.saved_state)}")
        self.assertFalse(fe.param_file == self._p_setting)
        self.assertTrue(fe.param_file == self._p_setting.read_text())

    def test_get_radiomics_features_w_aug(self):
        from mnts.utils import get_unique_IDs, get_fnames_by_IDs
        from mri_radiomics_toolkit.feature_extractor import get_radiomics_features
        import os

        p_im = Path('test_data/images/')
        p_seg_A = Path('test_data/segment')
        p_seg_B = Path('test_data/segment')


        dfs = []
        ids = get_unique_IDs(os.listdir(str(p_im)), "^[0-9]+")
        self._logger.info(f"IDs: {ids}")
        im_fs = get_fnames_by_IDs(os.listdir(str(p_im)), ids)
        seg_a_fs = get_fnames_by_IDs(os.listdir(str(p_seg_A)), ids)
        seg_b_fs = get_fnames_by_IDs(os.listdir(str(p_seg_B)), ids)
        for im, seg_a, seg_b in zip(im_fs, seg_a_fs, seg_b_fs):
            self._logger.info(f"Performing on: \n{pprint.pformat([im, seg_a, seg_b])}")
            df = get_radiomics_features(p_im.joinpath(im),
                                        p_seg_A.joinpath(seg_a),
                                        self._p_setting,
                                        p_seg_B.joinpath(seg_b),
                                        id_globber="^(NPC|T1rhoNPC|K|P|RHO)?[0-9]{2,4}",
                                        augmentor=self._transform)
            self._logger.debug(f"df: {df}")
            dfs.append(df)
        dfs = pd.concat(dfs, axis=1)
        new_index = [o.split('_') for o in dfs.index]
        new_index = pd.MultiIndex.from_tuples(new_index, names=('Pre-processing', 'Feature_Group', 'Feature_Name'))
        dfs.index = new_index
        self._logger.debug(f"dfs:\n {dfs.drop('diagnostics').to_string()}")
        pass

    def test_feature_selection(self):
        r"""Test the feature extractor in the pipeline"""
        self._logger.info("This could take a while...")
        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_feat_b = Path('test_data/assets/samples_feat_2nd.xlsx')
        p_gt = Path('test_data/assets/sample_datasheet.csv')

        features_a = pd.read_excel(str(p_feat_a), index_col=[0, 1, 2]).T
        features_b = pd.read_excel(str(p_feat_b), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)

        cases = list(set(features_a.index) & set(gt.index))
        gt = gt.loc[cases]

        passed = False
        with tempfile.NamedTemporaryFile('wb', suffix = '.fss') as f:
            fs = FeatureSelector(n_trials=20, boot_runs=5,
                                 criteria_threshold=[0.1, 0.1, 0.1],
                                 thres_percentage=0.2,
                                 boosting=True) # Use default criteria, test with boosting
            test_result = {x: "Untested" for x in ['Single feature set',
                                                   'Two paired feature sets',
                                                   'Save/load',
                                                   'n_trial = 1',
                                                   'n_trial & boot_run = 1']}
            # Test one segmentation
            self._logger.info("{:-^50s}".format(" Testing single feature set "))
            try:
                feats = fs.fit(features_a, gt)
                test_result['Single feature set'] = "Passed"
            except:
                test_result['Single feature set'] = "Failed"
            self._logger.info("Single feature set: Passed")

            # Test two segmentation
            self._logger.info("{:-^50s}".format(" Testing pair feature set "))
            cases = list(set(features_a.index) & set(features_b.index) & set(gt.index))
            try:
                feats = fs.fit(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])
                test_result['Two paired feature sets'] = "Passed"
            except:
                test_result['Two paired feature sets'] = "Failed"

            # Testing save and load function
            self._logger.info("{:-^50s}".format(" Testing state save/load "))
            try:
                fs.save(Path(f.name))
                _new_fs = FeatureSelector()
                _new_fs.load(Path(f.name))
                _feats = _new_fs.predict(features_a)

                self._logger.info(f"Left:\n {_feats.T}")
                self._logger.info(f"Right:\n {feats[0].T}")
                self._logger.info(f"Save/load (Passed)")
                test_result['Save/load'] = "Passed"
            except:
                test_result['Save/load'] = "Failed"

            # Test single trial (feature selection using enet with frequency threshold)
            self._logger.info("{:-^50s}".format(" Testing n_trial = 1 "))
            try:
                fs.setting['n_trials'] = 1
                feats = fs.fit(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])
                self._logger.info("n_trial = 1: Passed")
                test_result['n_trial = 1'] = "Passed"
            except:
                test_result['n_trial = 1'] = "Failed"


            # Test single boot_run (feature selection using enet without frequency threshold)
            self._logger.info("{:-^50s}".format(" Testing n_trial & boot_run = 1 "))
            try:
                fs.setting['n_trials'] = 1
                fs.setting['boot_runs'] = 1
                feats = fs.fit(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])
                self._logger.info(f"Single Enet run features extracted: {feats[0].columns}")
                self._logger.info("n_trial & boot_run: Passed")
                test_result['n_trial & boot_run = 1'] = "Passed"
            except:
                test_result['n_trial & boot_run = 1'] = "Failed"
            self._logger.info(f"Test results: \n{pd.Series(test_result, name='Test results').to_frame().to_string()}")

        self.assertFalse(all([x == "Passed" for x in test_result.items()]))

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

    def test_controller_extraction(self):
        p          = Path('test_data/assets/sample_controller_settings.yml')
        p_im       = Path('test_data/images_not_normalized/')
        p_seg      = Path('test_data/segment/')
        p_gt       = Path('test_data/assets/sample_datasheet.csv')
        p_pyrad    = Path('test_data/assets/sample_pyrad_settings.yml')
        p_fe_state = Path('test_data/assets/fe_saved_state.fe')

        # extract feature was ported to the controller, test it
        ctl = Controller(setting=p, with_norm=True)
        ctl.load_norm_settings(fe_state=p_fe_state)
        df = ctl.extract_feature(p_im, p_seg, py_rad_param_file=p_pyrad)
        self._logger.info(f"features {df}")

    def test_controller_load_norm(self):
        p = Path('test_data/assets/sample_controller_settings.yml')
        p_norm_state = Path('../assets/t2wfs/')
        p_norm_graph = Path('../assets/t2wfs/norm_graph.yml')
        p_fe_state = Path('test_data/assets/fe_saved_state.fe')

        ctl = Controller(setting=p, with_norm=True)
        ctl.load_norm_settings(norm_graph=p_norm_graph, norm_state_file=p_norm_state)
        self._logger.info(f"State 1: \n{pprint.pformat(ctl.extractor.saved_state)}")

        ctl.load_norm_settings(fe_state=p_fe_state)
        self._logger.info(f"State 2: \n{pprint.pformat(ctl.extractor.saved_state)}")

    def test_controller_fit(self):
        p = Path('test_data/assets/sample_controller_settings.yml')
        p_im = Path('test_data/images_not_normalized/')
        p_seg = Path('test_data/segment/')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        p_pyrad = Path('test_data/assets/sample_pyrad_settings.yml')
        p_fe_state = Path('test_data/assets/fe_saved_state.fe')

        # extract feature was ported to the controller, test it
        with tempfile.NamedTemporaryFile('wb', suffix='.ctl') as f:
            ctl = Controller(setting=p, with_norm=True)
            ctl.load_norm_settings(fe_state=p_fe_state)
            ctl.fit(p_im, p_seg, p_gt)

            ctl.save(f.name)
            _ctl = Controller()
            _ctl.load(f.name)
            self._logger.info(f"Saved state: {_ctl.saved_state}")

    def test_stability_metric(self):
        test_result = {x: "Untested" for x in ['Binary feature map',
                                               'Stability measure',
                                               'Statistical Test'
                                               ]}

        p_sel_1 = Path('test_data/assets/sample_selected_feat_1.xlsx')
        p_sel_2 = Path('test_data/assets/sample_selected_feat_2.xlsx')
        p_feat_list = Path('test_data/assets/samples_feat_1st.xlsx')

        sel_1 = pd.read_excel(p_sel_1, index_col=0).fillna('').astype(str)
        sel_2 = pd.read_excel(p_sel_2, index_col=0).fillna('').astype(str)
        feats = [str(s) for s in pd.read_excel(p_feat_list, index_col=[0, 1, 2]).index.to_list()]

        try:
            Z1 = feat_list_to_binary_mat(sel_1, feats)
            Z2 = feat_list_to_binary_mat(sel_2, feats)
            test_result['Binary feature map'] = "Passed"
        except:
            test_result['Binary feature map'] = "Failed"

        try:
            self._logger.info(f"{getStability(Z1)}, {getStability(Z2)}")
            test_result['Stability measure'] = "Passed"
        except:
            test_result['Stability measure'] = "Failed"

        self._logger.info(f"Test results: \n{pd.Series(test_result, name='Test results').to_frame().to_string()}")
        test_result['Statistical Test'] = "Passed"
        self.assertFalse(all([x == "Passed" for x in test_result.items()]))



if __name__ == '__main__':
    te = Test_pipeline()
    # te.test_controller_fit()
    # te.test_feature_extractor()
    # te.test_get_radiomics_features_w_aug()
    te.test_controller_extraction()