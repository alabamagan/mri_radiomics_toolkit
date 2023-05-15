import unittest
import pandas as pd
import radiomics
from pathlib import Path

from mnts.mnts_logger import MNTSLogger
from mri_radiomics_toolkit.feature_extractor import get_radiomics_features, \
    get_radiomics_features_from_folder, FeatureExtractor
from mri_radiomics_toolkit.utils import is_compressed



class Test_feature_extractor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_feature_extractor, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls) -> None:
        cls.cls_logger = MNTSLogger(".", "Test_FE", verbose=True, keep_file=False, log_level='debug')
        # hijack radiomics logger
        radiomics.setVerbosity(20) # INFO
        radiomics.logger = MNTSLogger['radiomics']

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()

    def setUp(self):
        self._logger = self.cls_logger

        # set up test data
        self.sample_img_dir = Path("test_data/image_mul_label")
        self.sample_seg_dir = Path("test_data/segment_mul_label")

        # single sample pair
        self.sample_img_1 = Path("test_data/image_mul_label/TEST01_img_t2wfs.nii.gz")
        self.sample_seg_1 = Path("test_data/segment_mul_label/TEST01_segment.nii.gz")
        self.sample_seg_1_bin = Path("test_data/segment_mul_label/TEST01_segment_binary.nii.gz")

        # pyradiomics settings
        self.settings_1 = Path("./test_data/test_pyrad_setting_1.yml")


    def test_get_radiomics_features(self):
        extracted_features: pd.DataFrame = get_radiomics_features(self.sample_img_1,
                                                                  self.sample_seg_1,
                                                                  self.settings_1)
        self.assertIsInstance(extracted_features, pd.DataFrame)

    def test_get_radiomics_features_by_slice(self):
        extracted_features: pd.DataFrame = get_radiomics_features(self.sample_img_1,
                                                                  self.sample_seg_1,
                                                                  self.settings_1,
                                                                  by_slice = 2)
        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertEqual(3, extracted_features.columns.nlevels)

    def test_get_radiomics_features_conn(self):
        extracted_features: pd.DataFrame = get_radiomics_features(self.sample_img_1,
                                                                  self.sample_seg_1_bin,
                                                                  self.settings_1,
                                                                  by_slice = 2,
                                                                  connected_components=True)
        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertEqual(3, extracted_features.columns.nlevels)

    def test_set_compressed_pyradsettings(self):
        extractor = FeatureExtractor()
        extractor.saved_state['param_file'] = \
            'H4sIAG4MYmQC/31RwVLCMBC98xW5cUEGCx7oTS3DOMLIAOrB8bC023bHNImbRGEc/92UtqIXTpv3XvZt8' \
            'pYqKHB7MBj3hHhgKkiBjMXXd4CLm1WU1LwQO1LPlLkyFqNhdHWk5M5EyRoy8rZmpydyia7UWSz6Gebgpe' \
            'sfpVxzisFPbNlj4z4+6z5O7lJtTYmMzRjRzVnoedNoqaggFi+j4WQyjQbiciCi16DMOTSgcu1HZnujVYD' \
            'Ufq2XIzjPeCvB2topJ7ZOc4bc+F6ImUIuDr/AsTYduvfstCXbwiXsqfJVhxDU9c5q6R0m+EHgSKs/Wntc' \
            '65237tzltdZHffPugTFr2c0bfiq03exHRSHWilz3tidgApXW+doSmq0WMq2ayjIcehadI1XUFKOFykjMV' \
            'rRHuTGQ1sIpz/+1Tva0qsuATMgwJNPu9AeiUQAbTAIAAA=='
        self.assertFalse(is_compressed(extractor.param_file))

