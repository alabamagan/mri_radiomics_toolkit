import unittest
import pandas as pd
import radiomics
import tempfile
from pathlib import Path

from mnts.mnts_logger import MNTSLogger
from mri_radiomics_toolkit.feature_extractor import get_radiomics_features, \
    get_radiomics_features_from_folder, FeatureExtractor
from mri_radiomics_toolkit.utils import is_compressed, compress, decompress, ExcelWriterProcess


class Test_feature_extractor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_feature_extractor, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls) -> None:
        cls.cls_logger = MNTSLogger(".", "Test_FE", verbose=True, keep_file=False, log_level='info')
        # hijack radiomics logger
        radiomics.setVerbosity(10) # INFO

        # removing its handlers to suppress wild messages
        ori_rad_logger = radiomics.logger
        h = ori_rad_logger.handlers
        for hh in h:
            ori_rad_logger.removeHandler(hh)
        radiomics.logger = MNTSLogger['radiomics']

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cls_logger.cleanup()

    def setUp(self):
        self._logger = self.cls_logger

        # set up unit_test data
        self.sample_img_dir = Path("./test_data/images").absolute()
        self.sample_seg_dir = Path("./test_data/segment").absolute()
        self.sample_img_dir_mpi = Path("./test_data/images").absolute()
        self.sample_seg_dir_mpi = Path("./test_data/segment").absolute()

        # single sample pair
        self.sample_img_1 = Path("./test_data/images/MRI_01.nii.gz").absolute()
        self.sample_seg_1 = Path("./test_data/segment/MRI_01.nii.gz").absolute()
        self.sample_seg_1_bin = Path("./test_data/segment/MRI_01.nii.gz").absolute()

        # pyradiomics settings
        self.settings_1 = Path("./test_data/test_pyrad_setting_1.yml").absolute()


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

    def test_mpi_extract_features(self):
        df = get_radiomics_features_from_folder(self.sample_img_dir_mpi,
                                               self.sample_seg_dir_mpi,
                                               self.settings_1,
                                               id_globber="MRI_\d+")
        self.assertGreater(len(df), 0)


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

    def test_saved_state_is_compressed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            extractor = FeatureExtractor()
            compressed_param_file = \
                'H4sIAG4MYmQC/31RwVLCMBC98xW5cUEGCx7oTS3DOMLIAOrB8bC023bHNImbRGEc/92UtqIXTpv3XvZt8' \
                'pYqKHB7MBj3hHhgKkiBjMXXd4CLm1WU1LwQO1LPlLkyFqNhdHWk5M5EyRoy8rZmpydyia7UWSz6Gebgpe' \
                'sfpVxzisFPbNlj4z4+6z5O7lJtTYmMzRjRzVnoedNoqaggFi+j4WQyjQbiciCi16DMOTSgcu1HZnujVYD' \
                'Ufq2XIzjPeCvB2topJ7ZOc4bc+F6ImUIuDr/AsTYduvfstCXbwiXsqfJVhxDU9c5q6R0m+EHgSKs/Wntc' \
                '65237tzltdZHffPugTFr2c0bfiq03exHRSHWilz3tidgApXW+doSmq0WMq2ayjIcehadI1XUFKOFykjMV' \
                'rRHuTGQ1sIpz/+1Tva0qsuATMgwJNPu9AeiUQAbTAIAAA=='
            extractor.param_file = compressed_param_file
            extractor.save(Path(tempdir) / 'unit_test')
            extractor.load(Path(tempdir) / 'unit_test.fe')
            self.assertEqual(extractor.param_file,
                             decompress(compressed_param_file))

    def test_excel_writer_process(self):
        with tempfile.TemporaryDirectory() as tempdir:
            writer = ExcelWriterProcess(tempdir + "/temp.xlsx")
            writer.start()

            test_series1 = pd.Series(data=[1, 2, 3], index=['A', 'B', 'C'], name='test1')
            test_series2 = pd.Series(data=[1, 2, 3], index=['A', 'B', 'C'], name='test2')
            writer.write(test_series1)
            writer.stop()

            # check if an excel is created
            self.assertTrue(Path(tempdir + "/temp.xlsx").is_file())

            # append another series
            writer = ExcelWriterProcess(tempdir + "/temp.xlsx")
            writer.start()
            writer.write(test_series2)
            writer.stop()

            # Check if the program is running correctly so far
            df_expected = pd.concat([test_series1, test_series2], axis=1)
            df = pd.read_excel(tempdir + "/temp.xlsx", index_col=0, engine='openpyxl')
            self.assertTrue(df.equals(df_expected))

            # unit_test if writing dataframe works
            df.columns = ['test3', 'test4']
            writer = ExcelWriterProcess(tempdir + "/temp.xlsx")
            writer.start()
            writer.write(df)
            writer.stop()
            df_expected = pd.concat([df_expected, df], axis=1)
            df = pd.read_excel(tempdir + "/temp.xlsx", index_col=0)
            self.assertTrue(df.equals(df_expected))

    def test_extract_feature_with_stream_writing(self):
        with tempfile.TemporaryDirectory() as f:
            temp_excelfile = f + '/temp.xlsx'
            writer = ExcelWriterProcess(temp_excelfile)
            writer.start()
            df = get_radiomics_features_from_folder(self.sample_img_dir_mpi.absolute(),
                                                   self.sample_seg_dir_mpi.absolute(),
                                                   self.settings_1,
                                                   id_globber="MRI_\d+",
                                                   writer_func=ExcelWriterProcess.write)
            writer.stop()
            saved_df = pd.read_excel(temp_excelfile, index_col=[0, 1, 2], engine='openpyxl')
            df.equals(saved_df)
