import unittest
import pandas as pd
import pytest
from mri_radiomics_toolkit.perf_metric import *
import numpy as np


class Test_Top_k(unittest.TestCase):
    def test_same_unique_values(self):
        y_true = [0, 2, 3, 3, 2, 1]
        y_score = np.random.random(size=[6, 3])
        y_score_int = [2, 2, 1, 1, 1, 1]
        kwargs = {}

        # Test multi-class with float score
        result = top_k_accuracy_score_(y_true, y_score, k = 1, **kwargs)

        # Test multi-class with integer prediction (score)
        result = top_k_accuracy_score_(y_true, y_score_int, k = 1, **kwargs)
        self.assertEqual(2 / 6., result)

    def test_specify_uniqe_labels(self):
        y_true = [0, 1, 2, 4, 1, 2]
        y_score = np.random.random(size=[6, 3])
        y_score_int = [2, 2, 1, 1, 1, 1]

        result = top_k_accuracy_score_(y_true, y_score, k = 1, labels=[0, 1, 2, 3, 4])
        result = top_k_accuracy_score_(y_true, y_score_int, k = 1, labels=[0, 1, 2, 3, 4])
        self.assertEqual(1 / 6., result)

    def test_warning_raised(self):
        r"""Warn when input score is integer and k > 1"""
        y_true = [0, 1, 2, 3, 1, 2]
        y_score_int = [2, 2, 1, 1, 1, 1]

        with self.assertWarns(UserWarning):
            result = top_k_accuracy_score_(y_true, y_score_int, k = 2, labels=[0, 1, 2, 3, 4])