import matplotlib.pyplot as plt
import pandas as pd
from mnts.mnts_logger import MNTSLogger
from sklearn import *
from sklearn.model_selection import *
from imblearn.over_sampling import SMOTE
from typing import Iterable, Union, Optional, Tuple, Any

def generate_cross_validation_samples(X: pd.DataFrame,
                                      y: pd.Series,
                                      n_splits: Optional[int] = 5,
                                      shuffle: Optional[bool] =True,
                                      random_state=None,
                                      stratification=True):
    r"""Generates cross-validation samples with balanced classes using SMOTE.

    This function splits the input data into training and testing sets, balances the classes in the
    training set using SMOTE, and then generates `n_splits` cross-validation samples.

    Args:
        X (numpy.ndarray):
            The input features. This should be a 2D array where each row is a sample and each column
            is a feature.
        y (numpy.ndarray):
            The target variable. This should be a 1D array with the same number of elements as the
            number of rows in `X`.
        n_splits (int, Optional):
            The number of cross-validation splits. Default is 5.
        shuffle (bool, Optional):
            Whether to shuffle the data before splitting. Default is True.
        random_state (int, RandomState instance or None, Optional):
            Determines random number generation for dataset shuffling and SMOTE.
            Pass an int for reproducible output across multiple function calls.
            Default is None.

    Returns:
        Iterator[tuple[np.ndarray, np.ndarray]]:
            A generator that yields pairs of numpy arrays. Each pair corresponds to the indices of
            the training and validation data for a cross-validation split.

    .. note::
        The class balancing is done on the training data only. The original class proportions are
        maintained in the testing data.
    """
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Use SMOTE to balance the classes in the training set
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        yield X_train_resampled, X_test, y_train_resampled, y_test