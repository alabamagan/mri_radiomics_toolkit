import mnts.mnts_logger
import pandas as pd
import pingouin as pg

from pathlib import Path

import sklearn.preprocessing
from mnts.mnts_logger import MNTSLogger
from tqdm.auto import *
from typing import Union, Optional, Iterable, List, Callable, Tuple, Dict

from sklearn.model_selection import *
from sklearn import *
from scipy.stats import *

import joblib
import numpy as np
import multiprocessing as mpi
from tqdm.auto import *
from functools import partial
from RENT import RENT, stability

global logger

__all__ = ['cv_grid_search', 'model_building', 'ModelBuilder']

def cv_grid_search(train_features: pd.DataFrame,
                   train_targets: pd.DataFrame,
                   test_features: Optional[pd.DataFrame] = None,
                   test_targets: Optional[pd.DataFrame] = None,
                   verbose=False) -> List[Union[pd.DataFrame, Dict]]:
    r"""
    Grid search for best hyper-parameters for the following linear models:
      * SVM
      * Logistic regresion
      * Random forest
      * K-nearest-neighbour
      * Elastic Net


    Args:
        train_features (pd.DataFrame):
            Input training features. Columns and rows should be features and sample respectively.
        train_targets (pd.DataFrame):
            Input training ground-truth. Should only have one column and each row should be one sample.
        test_features (pd.DataFrame, Optional):
            Input testing features. Columns and rows should be features and sample respectively. Default to None.
        test_targets:
            Input testing ground-truth. Default to None.

    Returns:
        best_params (dict):
            Key are methods, values are the best hyper-parameters for training/
        results (pd.DataFrame):
            AUC of the trainng and testing set of the best_estimator/
        predict_table (pd.DataFrame):
            The collection of predicted scores by each trained model.
        best_estimator (Any):
            The sklearn estimator.

    """
    clf = pipeline.Pipeline([
        ('standardization', preprocessing.StandardScaler()),
        ('classification', 'passthrough')
    ])

    # Construct tests to perform
    param_grid_dict = {
        'Support Vector Machine': {
            'classification': [svm.SVR(tol=1E-4, max_iter=3500)],
            'classification__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classification__C': [1, 10, 100, 1000],
            'classification__degree': [3, 5, 7, 9],
            'classification__epsilon': [1, 0.1, 0.01]
        },
        'Elastic Net': {
            'classification': [linear_model.ElasticNet(tol=1E-4, max_iter=3500)],
            'classification__alpha': [.02, .002],
            'classification__l1_ratio': [0.2, 0.5, 0.8]
        },
        'Logistic Regression': {
            'classification': [linear_model.LogisticRegression(penalty='elasticnet',
                                                               solver='saga', tol=1E-5,
                                                               max_iter=3500,
                                                               verbose=verbose)],
            'classification__C': [0.1, 1, 10, 100, 1000],
            'classification__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        'Random Forest': {
            'classification': [ensemble.RandomForestRegressor(n_estimators=50)],
            'classification__criterion': ['squared_error', 'poisson']
        },
        'Perceptron': {
            'classification': [neural_network.MLPRegressor(learning_rate='adaptive',
                                                           tol=1E-4,
                                                           max_iter=3000,
                                                           verbose=verbose)],
            'classification__hidden_layer_sizes': [(100), (20, 50, 100), (100, 50, 20)],
            'classification__learning_rate_init': [1, 0.1, 1E-2, 1E-3, 1E-4]
        },
        'KNN': {
            'classification': [neighbors.KNeighborsRegressor(n_jobs=5)],
            'classification__n_neighbors': [3, 5, 10, 20],
        }
    }

    best_params = {}
    best_estimators = {}
    results = {}
    predict_table = [test_targets]
    splitter = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    for key, param_grid in param_grid_dict.items():
        split = splitter.split(train_targets, train_targets.values.ravel())
        grid = GridSearchCV(clf, n_jobs=10, param_grid=param_grid, scoring='roc_auc',
                            cv=split)
        grid.fit(train_features.values, train_targets.values.ravel())

        # Isolate best estimator
        best_estimators[f"{key}"] = grid.best_estimator_
        best_params[key] = grid.best_params_

        # If testing set exist, evalute performance on the testing set
        if not (test_features is None or test_targets is None):
            y = grid.predict(test_features.values)
            train_y = grid.predict(train_features.values)
            train_score = metrics.roc_auc_score(train_targets.values.ravel(),
                                                train_y)
            test_score = metrics.roc_auc_score(test_targets.values.ravel(),
                                               y)
            results[f'{key}'] = test_score
            results[f'{key} (train)'] = train_score
            predict_table.append(pd.Series(y, index=test_targets.index, name=f"{key}"))

    # Collect the predictions of the testing sets
    if not test_targets is None and not test_features is None:
        predict_table = pd.concat(predict_table, axis=1)
    else:
        predict_table = pd.DataFrame() # Return empty dataframe if something went wrong
    return best_params, results, predict_table, best_estimators


def model_building(train_features: pd.DataFrame,
                   train_targets: pd.DataFrame,
                   test_features: Optional[pd.DataFrame] = None,
                   test_targets: Optional[pd.DataFrame] = None) -> Tuple[dict]:
    r"""
    Train model without grid-search. If test set data is provided, also evalute the performance of the built model using
    the test set data. This function is essentially same as `cv_grid_search` except there are no grid search.

    Args:
        See :func:`cv_grid_search`
    """

    logger = MNTSLogger['model-building']
    models = {
        'Logistic Regression': linear_model.LogisticRegression(penalty='elasticnet',
                                                               solver='saga',
                                                               C = 1/.02,
                                                               l1_ratio = .5,
                                                               max_iter=3000,
                                                               verbose=True),
        'Support Vector Machine': svm.SVR(kernel='linear',
                                          tol=5E-4,
                                          C = 10,
                                          epsilon=0.01,
                                          max_iter=3000,
                                          verbose=True),
        'Random Forest': ensemble.RandomForestRegressor(n_estimators=50,
                                                        verbose=True),
        'Perceptron': neural_network.MLPRegressor(hidden_layer_sizes=(20, 50, 100),
                                                  learning_rate='adaptive',
                                                  learning_rate_init=1E-4,
                                                  tol=5E-4,
                                                  max_iter=3000,
                                                  verbose=True),
        'Boosted SVM': ensemble.AdaBoostRegressor(svm.SVR(kernel='linear',
                                                          tol=5E-4,
                                                          C = 10,
                                                          epsilon=0.01,
                                                          max_iter=3000),
                                                  n_estimators=50)
    }

    results = {}
    for key in models:
        logger.info(f"Traning for {key}")
        m = models[key]
        m.fit(train_features, train_targets)

        if not (test_features is None or test_targets is None):
            logger.info(f"Performing tests...")
            y = m.predict(test_features)
            score = metrics.roc_auc_score(test_targets, y)
            score_trian = metrics.roc_auc_score(train_targets, m.predict(train_features))
            results[key] = score
            results[f"{key} (train)"] = score
        logger.info(f"Done for {key}.")

    return models, results

class ModelBuilder(object):
    def __init__(self, *args, verbose=False, **kwargs):
        super(ModelBuilder, self).__init__()

        self.saved_state = {
            'estimators': None,
            'best_params': None
        }
        self.verbose = verbose
        self._logger = MNTSLogger[__class__.__name__]

    def load(self, f: Path):
        r"""
        Load att `self.save_state`. The file saved should be a dictionary containing key 'selected_features', which
        points to a list of features in the format of pd.MultiIndex or tuple
        """
        assert Path(f).is_file(), f"Cannot open file {f}"
        d = joblib.load(f)
        if not isinstance(d, dict):
            raise TypeError("State loaded is incorrect!")
        self.saved_state.update(d)

    def save(self, f: Path):
        if any([v is None for v in self.saved_state.values()]):
            raise ArithmeticError("There are nothing to save.")
        joblib.dump(self.saved_state, filename=f.with_suffix('.pkl'))

    def fit(self,
            train_features: pd.DataFrame,
            train_targets: pd.DataFrame,
            test_features: pd.DataFrame = None,
            test_targets: pd.DataFrame = None) -> Tuple[pd.DataFrame]:
        if not self.check_dimension(train_features, train_targets):
            # Try to match dimensions for convenience
            self._logger.warning("Miss-match found for train data.")
            train_features, train_targets = self.match_dimension(train_features, train_targets)
        if not (test_features is None or test_targets is None):
            self._logger.warning("Miss-match found for test data.")
            test_features, test_targets = self.match_dimension(test_features, test_targets)

        best_params, results, predict_table, best_estimators = cv_grid_search(train_features,
                                                                              train_targets,
                                                                              test_features,
                                                                              test_targets,
                                                                              verbose=self.verbose)
        self.saved_state['estimators'] = best_estimators
        self.saved_state['best_params'] = best_params
        return results, predict_table

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        r"""
        Predict the class supplied features.

        Args:
            features (pd.DataFrame):
                Selected features.
        Returns:
            (dict)
        """
        assert self.saved_state['estimators'] is not None, "Call fit() or load() first."
        estimators = self.saved_state['estimators']
        d = {v: estimators[v].predict(features) for v in estimators}
        df = pd.DataFrame.from_dict(d)
        df['ID'] = features.index
        df.set_index('ID', drop=True, inplace=True)
        return df

    def check_dimension(self, X: pd.DataFrame, y: pd.DataFrame):
        r"""
        Rows of X must match length of y.
        """
        if not len(X) == len(y):
            self._logger.warning(f"Warning shape of X ({X.shape}) and y ({y.shape}) mis-match! ")
            return False
        else:
            return True

    def match_dimension(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame]:
        r"""
        Match the rows of X and y
        """
        overlap = set(X.index.astype(str)) & set(y.index.astype(str))
        missing_feat = set(X.index.astype(str)) - overlap
        missing_targ = set(y.index.astype(str)) - overlap
        if len(missing_feat) > 0:
            msg = ','.join(missing_feat)
            self._logger.info(f"Rows removed from features: {msg}")
        if len(missing_targ) > 0:
            msg = ','.join(missing_targ)
            self._logger.info(f"Rows removed from targets: {msg}")
        return X.loc[overlap], y.loc[overlap]