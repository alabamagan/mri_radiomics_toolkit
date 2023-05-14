import unittest
from pathlib import Path
import pandas as pd

from mri_radiomics_toolkit.feature_selection import filter_features_by_T_test, filter_low_var_features, \
    filter_features_by_ICC_thres, FeatureSelector, compute_ICC, bootstrapped_features_selection, \
    preliminary_feature_filtering, supervised_features_selection
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

        print(features_a)
        # Drop diagnostics
        features_a.drop('diagnostics', inplace=True)
        features_b.drop('diagnostics', inplace=True)

        # make sure the cases are overlapped between features and gt dataframe
        cases = list(set(features_a.columns) & set(gt.index))
        cls.gt    = gt.loc[cases]
        cls.features_a = features_a[cases]
        cls.features_b = features_b[cases]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()

    def setUp(self) -> None:
        self._logger = MNTSLogger('test selector')
        self.features_a = Test_selector.features_a.copy()
        self.features_b = Test_selector.features_b.copy()
        self.gt = Test_selector.gt.copy()

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

    def test_preliminary_filter_pipeline(self):
        preliminary_feature_filtering(self.features_a, self.features_b, self.gt)
        pass

    def test_supervised_filter_pipeline(self):
        pass

    def test_Selector_IO(self):
        pass