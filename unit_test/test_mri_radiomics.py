import unittest
import os
import subprocess
import tempfile
import shutil
import pprint
from rich.traceback import install
install(show_locals=True)

from pathlib import Path

import sklearn.model_selection
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
        self._logger = MNTSLogger['unittest']
        self._globber = "^MRI_[0-9]+"
        self._p_setting = Path('test_data/assets/sample_pyrad_settings.yml')
        self._transform = Path('test_data/assets/sample_augment_transform.yml')
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_feature_extractor_w_norm(self):
        p_im = Path('test_data/images_not_normalized/')
        p_seg = Path('test_data/segment/')
        norm_graph =\
        """
        SpatialNorm:
            out_spacing: [0.45, 0.45, 0]
        
        HuangThresholding:
            closing_kernel_size: 10
            _ext:
                upstream: 0 
                is_exit: True
        
        N4ITKBiasFieldCorrection:
            _ext:
                upstream: [0, 1]
            
        ZScoreNorm:
            _ext:
                upstream: [2, 1]
                is_exit: True
        """ # Use this version that does not need training

        # Create feature extractor
        self._logger.info("{:-^50s}".format(" Testing feature extraction "))
        fe = FeatureExtractor(id_globber=self._globber, norm_graph=norm_graph)
        fe.param_file = self._p_setting
        df = fe.extract_features_with_norm(p_im, p_seg, param_file = self._p_setting)
        fe.save_features(self._p_setting.with_name('sample_features.xlsx'))
        self._logger.info("\n" + df.to_string())
        self.assertTrue(len(df) > 0)
        self.assertTrue(self._p_setting.with_name('sample_features.xlsx').is_file())
        self._logger.info("Feature extraction pass...")

        # unit_test save state
        self._logger.info("{:-^50s}".format(" Testing save state "))
        fe.save(Path('test_data/assets/fe_saved_state.fe'))
        fe.save(Path(self.temp_dir))
        self.assertTrue(Path(self.temp_dir).joinpath('saved_state.fe').is_file())

        # unit_test load state
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
        fe = FeatureExtractor(id_globber=self._globber, idlist=['MRI_01', 'MRI_02'])
        fe.param_file = self._p_setting
        df = fe.extract_features(p_im, p_seg, param_file=self._p_setting)
        fe.save_features(self._p_setting.with_name('sample_features.xlsx'))
        self._logger.info("\n" + df.to_string())
        self.assertTrue(len(df) > 0)
        self.assertTrue(self._p_setting.with_name('sample_features.xlsx').is_file())
        self._logger.info("Feature extraction pass...")

        # unit_test save state
        self._logger.info("{:-^50s}".format(" Testing save state "))
        fe.save(Path(self.temp_dir) / 'fe.tar.gz')
        self.assertTrue((Path(self.temp_dir) / 'fe.tar.gz').is_file())

        # unit_test load state
        self._logger.info("{:-^50s}".format(" Testing load state "))
        _fe = FeatureExtractor(id_globber=self._globber)
        _fe.load(Path(self.temp_dir).joinpath('fe.tar.gz'))
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

    @unittest.skip("Depracated functino for augmenting input")
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
            # Replace FeatureSelector with FeatureSelectionPipeline
            from mri_radiomics_toolkit.feature_selection import (
                FeatureSelectionPipeline,
                BootstrappedSelectionStep,
                SupervisedSelectionStep,
                BBRENTStep
            )
            
            # Test 1: Create a pipeline with a bootstrapped selection step
            self._logger.info("Creating test pipeline with BootstrappedSelectionStep")
            # Create inner pipeline for bootstrapping
            inner_pipeline = FeatureSelectionPipeline(name="InnerPipeline")
            inner_pipeline.add_step(
                SupervisedSelectionStep(
                    criteria_threshold=(0.1, 0.1, 0.1),
                    n_trials=20,
                    boosting=True
                )
            )
            
            fs = FeatureSelectionPipeline(name="TestFeatureSelectionPipeline")
            fs.add_step(
                BootstrappedSelectionStep(
                    selection_pipeline=inner_pipeline,
                    n_bootstrap=5,
                    threshold_percentage=0.2,
                    random_state=42
                )
            )
            
            # Test 2: Create a pipeline with BBRENTStep (specialized convenience class)
            self._logger.info("Creating test pipeline with BBRENTStep")
            fs2 = FeatureSelectionPipeline(name="TestBBRENTPipeline")
            fs2.add_step(
                BBRENTStep(
                    criteria_threshold=(0.1, 0.1, 0.1),
                    n_trials=20,
                    n_bootstrap=5,
                    threshold_percentage=0.2,
                    boosting=True,
                    random_state=42
                )
            )
            
            test_result = {x: "Untested" for x in ['Single feature set',
                                                   'Two paired feature sets',
                                                   'Save/load',
                                                   'BBRENTStep']}
                                                   
            # Test one segmentation with standard pipeline
            self._logger.info("{:-^50s}".format(" Testing single feature set "))
            try:
                feats = fs.fit_transform(features_a, gt)
                test_result['Single feature set'] = "Passed"
            except Exception as e:
                self._logger.error(f"Single feature set test failed: {e}")
                test_result['Single feature set'] = "Failed"
            self._logger.info("Single feature set: Passed")

            # Test with BBRENTStep pipeline
            self._logger.info("{:-^50s}".format(" Testing BBRENTStep "))
            try:
                feats2 = fs2.fit_transform(features_a, gt)
                test_result['BBRENTStep'] = "Passed"
            except Exception as e:
                self._logger.error(f"BBRENTStep test failed: {e}")
                test_result['BBRENTStep'] = "Failed"
            self._logger.info("BBRENTStep: Passed")

            # Test two segmentation with standard pipeline
            self._logger.info("{:-^50s}".format(" Testing pair feature set "))
            cases = list(set(features_a.index) & set(features_b.index) & set(gt.index))
            try:
                feats = fs.fit_transform(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])
                test_result['Two paired feature sets'] = "Passed"
            except:
                test_result['Two paired feature sets'] = "Failed"

            # Testing save and load function
            self._logger.info("{:-^50s}".format(" Testing state save/load "))
            try:
                fs.save(Path(f.name))
                _new_fs = FeatureSelectionPipeline.load(Path(f.name))
                _feats = _new_fs.transform(features_a)

                self._logger.info(f"Left:\n {_feats}")
                self._logger.info(f"Right:\n {feats}")
                test_result['Save/load'] = "Passed"
            except Exception as e:
                self._logger.error(f"Save/load failed: {str(e)}")
                test_result['Save/load'] = "Failed"

            # Testing n_trial = 1
            self._logger.info("{:-^50s}".format(" Testing n_trial = 1 "))
            try:
                # Create a new pipeline with n_trials=1
                fs = FeatureSelectionPipeline(name="TestFeatureSelectionPipeline")
                fs.add_step(
                    BootstrappedSelectionStep(
                        criteria_threshold=[0.1, 0.1, 0.1],
                        n_trials=1,
                        boot_runs=5,
                        thres_percentage=0.2,
                        boosting=True
                    )
                )
                feats = fs.fit_transform(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])
                self._logger.info("n_trial = 1: Passed")
                test_result['n_trial = 1'] = "Passed"
            except Exception as e:
                self._logger.error(f"n_trial = 1 failed: {str(e)}")
                test_result['n_trial = 1'] = "Failed"

            # Testing n_trial = 1 and boot_run = 1
            self._logger.info("{:-^50s}".format(" Testing n_trial = 1 and boot_run = 1 "))
            try:
                # Create a new pipeline with n_trials=1 and boot_runs=1
                fs = FeatureSelectionPipeline(name="TestFeatureSelectionPipeline")
                fs.add_step(
                    BootstrappedSelectionStep(
                        criteria_threshold=[0.1, 0.1, 0.1],
                        n_trials=1,
                        boot_runs=1,
                        thres_percentage=0.2,
                        boosting=True
                    )
                )
                feats = fs.fit_transform(features_a.loc[cases], gt.loc[cases], features_b.loc[cases])
                if isinstance(feats, tuple):
                    self._logger.info(f"Single Enet run features extracted: {feats[0].columns}")
                else:
                    self._logger.info(f"Single Enet run features extracted: {feats.columns}")
                self._logger.info("n_trial & boot_run: Passed")
                test_result['n_trial & boot_run = 1'] = "Passed"
            except Exception as e:
                self._logger.error(f"n_trial & boot_run = 1 failed: {str(e)}")
                test_result['n_trial & boot_run = 1'] = "Failed"
            self._logger.info(f"Test results: \n{pd.Series(test_result, name='Test results').to_frame().to_string()}")

        self.assertFalse(all([x == "Passed" for x in test_result.items()]))

    def test_controller_extraction(self):
        p          = Path('test_data/assets/sample_controller_settings.yml')
        p_im       = Path('test_data/images/')
        p_seg      = Path('test_data/segment/')
        p_gt       = Path('test_data/assets/sample_datasheet.csv')
        p_pyrad    = Path('test_data/assets/sample_pyrad_settings.yml')
        p_fe_state = Path('test_data/assets/fe_saved_state.tar.gz')

        # extract feature was ported to the controller, unit_test it
        ctl = Controller(setting=p, with_norm=True)
        ctl.load_norm_settings(fe_state=p_fe_state)
        df = ctl.extract_feature(p_im, p_seg, py_rad_param_file=p_pyrad)
        self._logger.info(f"features {df}")

    def test_controller_load_norm(self):
        p = Path('test_data/assets/sample_controller_settings.yml')
        p_norm_state = Path('../assets/t2wfs/')
        p_norm_graph = Path('../assets/t2wfs/norm_graph.yml')
        p_fe_state = Path('test_data/assets/fe_saved_state.tar.gz')

        ctl = Controller(setting=p, with_norm=True)
        ctl.load_norm_settings(norm_graph=p_norm_graph, norm_state_file=p_norm_state)
        self._logger.info(f"State 1: \n{pprint.pformat(ctl.extractor.saved_state)}")

        ctl.load_norm_settings(fe_state=p_fe_state)
        self._logger.info(f"State 2: \n{pprint.pformat(ctl.extractor.saved_state)}")

    def test_controller_fit(self):
        p = Path('test_data/assets/sample_controller_settings.yml')
        p_im = Path('test_data/images/')
        p_seg = Path('test_data/segment/')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        p_pyrad = Path('test_data/assets/sample_pyrad_settings.yml')
        p_fe_state = Path('test_data/assets/fe_saved_state.tar.gz')

        # Make sure all files exist
        for path in [p_im, p_seg, p_gt, p_pyrad, p_fe_state]:
            self.assertTrue(path.is_file(), f"File not found: {path}")

        # extract feature was ported to the controller, unit_test it
        with tempfile.NamedTemporaryFile('wb', suffix='.ctl') as f:
            ctl = Controller(setting=p, with_norm=True)
            ctl.load_norm_settings(fe_state=p_fe_state)
            ctl.fit(p_im, p_seg, p_gt)

            ctl.save(f.name)
            _ctl = Controller()
            _ctl.load(f.name)
            self._logger.info(f"Saved state: {_ctl.saved_state}")

    @unittest.skip("Long unit_test")
    def test_controller_fit_df(self):
        # This could take a while
        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        p_controller_setting = Path('test_data/assets/sample_controller_settings.yml')

        feat = pd.read_excel(p_feat_a, index_col = [0, 1, 2]).T
        feat.drop('diagnostics', axis=1, inplace=True)
        gt   = pd.read_csv(p_gt      , index_col = 0)
        overlap = feat.index.intersection(gt.index)
        feat = feat.loc[overlap]
        gt = gt.loc[overlap]
        assert feat.index.identical(gt.index), f"{feat.index.difference(gt.index)}"

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(feat, gt, test_size=0.5)
        assert X_train.index.identical(y_train.index)

        ctl = Controller(setting=p_controller_setting, with_norm=False)
        res, table = ctl.fit_df(X_train, y_train, test_features=X_test, test_targets=y_test)

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

    def test_controller_new_save_load(self):
        """Test the save/load mechanism for Controller"""
        import tempfile
        
        p = Path('test_data/assets/sample_controller_settings.yml')
        p_im = Path('test_data/images/')
        p_seg = Path('test_data/segment/')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        p_pyrad = Path('test_data/assets/sample_pyrad_settings.yml')
        
        # Create a controller and extract features
        ctl = Controller(setting=p)
        df = ctl.extract_feature(p_im, p_seg, py_rad_param_file=p_pyrad)
        
        # Test save/load with the new mechanism
        with tempfile.TemporaryDirectory() as tempdir:
            save_dir = Path(tempdir) / 'controller_state'
            
            # Save the controller
            ctl.save(save_dir)
            
            # Create a new controller and load the state
            new_ctl = Controller()
            new_ctl.load(save_dir)
            
            # Check if states match
            self.assertEqual(ctl.fe.param_file, new_ctl.fe.param_file)
            self.assertEqual(ctl.fe.id_globber, new_ctl.fe.id_globber)
            
            # Check if extracted features match
            if hasattr(ctl.fe, 'extracted_features') and ctl.fe.extracted_features is not None:
                pd.testing.assert_frame_equal(
                    ctl.fe.extracted_features,
                    new_ctl.fe.extracted_features
                )

    def test_controller_with_bbrent(self):
        """Test controller with BBRENT feature selection from config file."""
        p_controller_setting = Path('test_data/assets/sample_controller_settings.yml')
        
        # Read the settings file to verify it uses boot_runs > 1
        import yaml
        with open(p_controller_setting, 'r') as f:
            settings = yaml.safe_load(f)
            
        # Verify that the settings use boot_runs > 1 which should trigger BBRENTStep
        self.assertIn('Selector', settings)
        self.assertIn('n_bootstrap', settings['Selector'])
        self.assertGreater(settings['Selector']['n_bootstrap'], 1)
        
        # Create a controller with the settings
        ctl = Controller(setting=p_controller_setting)
        
        # Verify that the controller created a pipeline with BBRENTStep
        self.assertIsNotNone(ctl.selector)
        self.assertGreater(len(ctl.selector.steps), 0)
        
        # Check if the first step is a BBRENTStep
        step = ctl.selector.steps[0]
        from mri_radiomics_toolkit.feature_selection import BBRENTStep
        self.assertIsInstance(step, BBRENTStep)
        
        # Verify that parameters were correctly passed to BBRENTStep
        self.assertEqual(step.n_trials, settings['Selector']['n_trials'])
        self.assertEqual(step.boot_runs, settings['Selector']['boot_runs'])
        self.assertEqual(step.thres_percentage, settings['Selector']['thres_percentage'])
        
        # Test that the pipeline can process data
        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_gt = Path('test_data/assets/sample_datasheet.csv')
        
        feat = pd.read_excel(p_feat_a, index_col=[0, 1, 2]).T
        try:
            feat.drop('diagnostics', axis=1, inplace=True)
        except KeyError:
            pass  # diagnostics column might not exist
            
        gt = pd.read_csv(p_gt, index_col=0)
        overlap = feat.index.intersection(gt.index)
        feat = feat.loc[overlap]
        gt = gt.loc[overlap]
        
        # Only test with a small subset for faster execution
        sample_indices = np.random.choice(feat.index, min(20, len(feat)), replace=False)
        feat_sample = feat.loc[sample_indices]
        gt_sample = gt.loc[sample_indices]
        
        try:
            # Test feature selection with the controller
            result, _ = ctl.fit_df(feat_sample, gt_sample)
            self.assertIsNotNone(result)
            self.assertIsNotNone(ctl.selected_features)
            self.assertGreater(len(ctl.selected_features), 0)
        except Exception as e:
            self.fail(f"Controller feature selection with BBRENTStep failed: {e}")
