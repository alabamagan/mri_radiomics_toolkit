import unittest
from pathlib import Path
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore', '.*ConvergenceWarning.*')

from mri_radiomics_toolkit.feature_selection import filter_features_by_T_test, filter_low_var_features, \
    filter_features_by_ICC_thres, FeatureSelector, compute_ICC, bootstrapped_features_selection, \
    preliminary_feature_filtering, supervised_features_selection, filter_features_by_ANOVA, features_normalization, \
    features_selection
from mnts.mnts_logger import MNTSLogger

class Test_selector(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cls_logger = MNTSLogger(".", "Test_selector", verbose=True, keep_file=False,
                                    log_level='debug')
        # Prepare dummy data
        p_feat_a = Path('test_data/assets/samples_feat_1st.xlsx')
        p_feat_b = Path('test_data/assets/samples_feat_2nd.xlsx')
        p_gt     = Path('test_data/assets/sample_datasheet.csv')

        # Load as attribute
        features_a = pd.read_excel(str(p_feat_a), index_col = [0, 1, 2])
        features_b = pd.read_excel(str(p_feat_b), index_col = [0, 1, 2])
        gt         = pd.read_csv(str(p_gt)      , index_col = 0)

        # Drop diagnostics
        features_a.drop('diagnostics', inplace=True)
        features_b.drop('diagnostics', inplace=True)

        # make sure the cases are overlapped between features and gt dataframe
        cases = list(set(features_a.columns) & set(gt.index))
        # limit to 100 cases
        cls.gt    = gt.loc[cases[:100]]
        cls.features_a = features_a[cases[:100]]
        cls.features_b = features_b[cases[:100]]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()

    def setUp(self) -> None:
        self._logger = MNTSLogger('test selector')
        self.features_a = Test_selector.features_a.copy()
        self.features_b = Test_selector.features_b.copy()
        self.gt = Test_selector.gt.copy()
        np.random.seed(4213215)

    def test_filter_by_t_test(self):
        pval_df = filter_features_by_T_test(self.features_a, self.gt)
        self.assertIsInstance(pval_df, pd.DataFrame)

    def test_filter_by_ICC(self):
        # This might take a while
        icc_df = filter_features_by_ICC_thres(self.features_a, self.features_b)
        self.assertIsInstance(icc_df, pd.MultiIndex)

    def test_filter_by_variance(self):
        idx, _ = filter_low_var_features(self.features_a)
        self.assertIsInstance(idx, pd.MultiIndex)

        idx, _ = filter_low_var_features(self.features_b)
        self.assertIsInstance(idx, pd.MultiIndex)

    def test_preliminary_filter_pipeline(self):
        fa, fb = preliminary_feature_filtering(self.features_a, self.features_b, self.gt)
        self.assertIsInstance(fa, pd.DataFrame)
        self.assertIsInstance(fb, pd.DataFrame)
        self.assertTrue(fa.index.identical(fb.index))

    def test_supervised_filter_pipeline(self):
        self._logger.info(f"Testing supervised filter pipeline, this could take a while.")
        sf = supervised_features_selection(self.features_a, self.gt, 0.02, 0.5,
                                           n_trials=5,
                                           n_features=5,
                                           criteria_threshold=(0.1, 0.1, 0.1) # Set a lot threshold to ensure it works
                                           )
        self.assertEqual(5, sf.shape[0])
        self.assertIsInstance(sf, (pd.DataFrame, pd.Index, pd.MultiIndex))

    def test_supervised_filter_pipeline_ENet(self):
        self._logger.info(f"Testing supervised filter pipeline, this could take a while.")
        sf = supervised_features_selection(self.features_a, self.gt, 0.02, 0.5,
                                           n_trials=1, # n_trials == 1 means a sigle elastic net run
                                           n_features=5,
                                           )
        self.assertEqual(5, sf.shape[0])
        self.assertIsInstance(sf, (pd.DataFrame, pd.Index, pd.MultiIndex))

    def test_supervised_filter_pipeline_multiclass(self):
        self._logger.info(f"Testing supervised filter pipeline, this could take a while.")
        index_to_change = np.random.randint(0, 100, size=10)
        new_gt = self.gt.copy()
        new_gt.iloc[index_to_change] = 2
        sf = supervised_features_selection(self.features_a, new_gt, 0.02, 0.5,
                                           n_trials=1,
                                           n_features=5,
                                           criteria_threshold=(0.1, 0.1, 0.1) # Set a lot threshold to ensure it works
        )
        self.assertIsInstance(sf, (pd.DataFrame, pd.Index, pd.MultiIndex))
        self.assertLessEqual(10, len(sf))

    def test_Selector_IO(self):
        r"""Test creating the selector"""
        s = FeatureSelector()
        pass

    def test_Selector_inference(self):
        pass

    @unittest.skip("This takes too long")
    def test_Selector_fit(self):
        # This might take a while
        self._logger.info(f"Testing select fit pipeline, this could take a while.")
        s = FeatureSelector(boot_runs=3)
        s.fit(self.features_a.T, self.gt, self.features_b.T)

    def test_filter_by_ANOVA(self):
        # artificially add more classes to gt
        np.random.seed(4213215)
        index_to_change = np.random.randint(0, 100, size=10)
        new_gt = self.gt.copy()
        new_gt.iloc[index_to_change] = 2

        # run filter
        sf = filter_features_by_ANOVA(self.features_a, new_gt)
        self.assertIsInstance(sf, pd.DataFrame)
        pass

    def test_feature_selection(self):
        np.random.seed(4213215)
        # Run feature selection
        sf = features_selection(self.features_a, self.gt, features_b=self.features_b, n_trials=5,
                                criteria_threshold=(0.9, 0.1, 0))

    def test_feature_selection_multiclass(self):
        gt_mc = np.random.randint(0, 3, size=len(self.gt))
        gt_mc = pd.DataFrame(gt_mc, index=self.gt.index, columns=self.gt.columns)
        # Run feature selection
        sf = features_selection(self.features_a, gt_mc, features_b=self.features_b, n_trials=1,
                                criteria_threshold=(0.9, 0.1, 0), ICC_p_thres=0.9)

    def test_preliminary_feature_filtering(self):
        # Run feature selection
        sf = preliminary_feature_filtering(self.features_a, self.features_b, self.gt)

    def test_preliminary_feature_filtering_multiclass(self):
        gt_mc = np.random.randint(0, 3, size=len(self.gt))
        gt_mc = pd.DataFrame(gt_mc, index=self.gt.index, columns=self.gt.columns)
        # Run feature selection
        sf = preliminary_feature_filtering(self.features_a, self.features_b, gt_mc)