import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mnts.mnts_logger import MNTSLogger
from sklearn import *
from sklearn.model_selection import *
from imblearn.over_sampling import SMOTE

from typing import Iterable, Union, Optional, Tuple, Any
from numpy.typing import NDArray

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
        Performing SMOTE before cross-validation split introduce data leakage becuase CV does not consider
        the synthesized data and its synthesize source. Therefore, many synthesized data points and their
        sources could exist simultaneously in the training and testing set, i.e., the testing data or its
        sytnehsizing sources are already seen during training. This is inapproprieate and cannot accurately
        reflect the performance of a model. Therefore, this function yields Kfold splits and only performs
        SMOTE *after* the cross-validation split.
    """
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Use SMOTE to balance the classes in the training set
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        yield X_train_resampled, X_test, y_train_resampled, y_test


class StratifiedSMOTEKFold(StratifiedKFold):
    r"""Implements cross-validation with stratification and SMOTE.

    This class provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of StratifiedKFold that returns
    stratified and SMOTE-resampled folds. The folds are made by preserving the
    percentage of samples for each class.

    This implementation inherits the behaviour of StratifiedKFold from sklearn.
    For each split, it applies SMOTE to the training data only, leaving the
    testing data untouched.

    Args are the same as those of StratifiedKFold from sklearn.

    Note:
        Inherits from StratifiedKFold.
        The class balancing using SMOTE is done on the training data only.

    See Also:
        :class:`sklearn.model_selection.StratifiedKFold`

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def split(self,
              X: Union[NDArray, pd.DataFrame],
              y: Union[NDArray, pd.Series],
              groups=None) -> Iterable[Union[Tuple[NDArray, NDArray, NDArray, NDArray]]]:
        r"""Generates indices to split data into training and test set.

        Overriding the split method to apply SMOTE on the training data for each fold.

        Args:
            X (Union[NDArray, pd.DataFrame]):
                The training input samples. Can be a numpy array or pandas DataFrame.
            y (Union[NDArray, pd.Series]):
                The target values (class labels). Can be a numpy array or pandas Series.
            groups (None):
                Always None, exists for compatibility with GroupKFold.

        Yields:
            tuple (Union[Tuple[NDArray, NDArray, NDArray, NDArray]]):
                For each split, yield a tuple containing the resampled training features,
                the testing features, the resampled training target, and the testing target.

        Raises:
            TypeError: If X and y are not both numpy arrays or both pandas DataFrame/Series.
        """
        for train_index, test_index in super().split(X, y, groups):
                if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                elif isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                else:
                    raise TypeError("X and y must be numpy arrays or pandas DataFrames/Series .")


                smote = SMOTE(random_state=self.random_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                yield X_train_resampled, X_test, y_train_resampled, y_test

