import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from mri_radiomics_toolkit.data_split import *
from imblearn.datasets import make_imbalance, fetch_datasets
import unittest


class Test_DataSplit(unittest.TestCase):
    def test_stratified_smote_kfold(self):
        r"""Test StratifiedSmoteKFold"""
        # make imbalance dataset
        df = fetch_datasets(data_home='/tmp/imblearn_dataset')['ecoli']
        X = df['data']
        y = df['target']

        X = pd.DataFrame(data=X)
        y = pd.Series(data=y)

        smote_splitter = StratifiedSMOTEKFold(n_splits=5)
        split = smote_splitter.split(X, y.to_frame())

        for _X_train, _X_test, _y_train, _y_test in split:
            # check if positive and negative sample ratio are the similar
            self.assertAlmostEquals(*_y_train.value_counts())

