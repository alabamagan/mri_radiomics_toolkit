from pathlib import Path
from typing import List, Optional, Tuple, Union, Any

import joblib
import pandas as pd
import numpy as np
from mnts.mnts_logger import MNTSLogger
from sklearn import *
from sklearn.model_selection import *
from .models.cards import default_cv_grid_search_card

__all__ = ['cv_grid_search', 'ModelBuilder', 'neg_log_loss']


def neg_log_loss(estimator: Any,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]):
    r"""Use this scoring function for multi-class classification problem for GridSearchCV.

    Because the default `sklearn` implementation uses `predict_proba` for GridSearchCV scoring functions,
    this method provides an escape for regressors. Note that the function :func:`sklearn.metrics.log_loss` gives
    the negative log likelihood that should be minimized. An addition negative sign is needed to turn this to
    maximization problem.

    Args:
        estimator (object):
            The estimator object used for prediction.
        X (pd.DataFrame or np.ndarray):
            The input features for prediction.
        y (pd.Series or np.ndarray):
            The target class for prediction. Should have a shape of (1, -1) and of type integer.

    Returns:
        float: The score value calculated based on the estimator's predictions.
    """
    try:
        pred = estimator.predict_proba(X)
    except AttributeError:
        pred = estimator.predict(X)
    return -metrics.log_loss(y, pred, labels=np.unique(y))


def cv_grid_search(train_features: pd.DataFrame,
                   train_targets: pd.DataFrame,
                   test_features: Optional[pd.DataFrame] = None,
                   test_targets: Optional[pd.DataFrame] = None,
                   verbose: Optional[bool] = False,
                   classifiers: Optional[str] = None,
                   **kwargs) -> Tuple[dict, pd.DataFrame, pd.DataFrame, Any]:
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
        test_targets (pd.DataFrame, Optional):
            Input testing ground-truth. Default to None.
        verbose (bool, Optional):
            Deprecated.
        classifiers (str, Optional):
            This can be one of the keys of `param_grid_dict`: ['Support Vector Regression'|'Elastic Net'|'Logistic
            Regression'|'Random Forest'|'Perceptron'|'KNN']. If specified, only that method will be trained. Default
            to None
        **kwargs (dict):
            Keyword arguments for GridSearchCV.

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
    logger = MNTSLogger['model-building']

    clf = pipeline.Pipeline([
        ('standardization', preprocessing.StandardScaler()),
        ('classification', 'passthrough')
    ])

    if verbose:
        raise DeprecationWarning(f"This option is deprecated.")

    # Construct tests to perform
    param_grid_dict = kwargs.pop('param_grid_dict', default_cv_grid_search_card)

    if not classifiers is None:
        if not isinstance(classifiers, (list, tuple, str)):
            raise TypeError(f"classifiers must be a list, a tuple or a string, got {type(classifiers)} instead.")
        if isinstance(classifiers, str):
            classifiers = [classifiers]
        param_keys = list(param_grid_dict.keys())
        for key in param_keys:
            if key not in classifiers:
                param_grid_dict.pop(key)
        if len(param_grid_dict) == 0:
            raise ArithmeticError("No classifiers were selected for model building.")

    # Pop kwargs for GridSearchCV
    scoring = kwargs.pop('scoring', 'roc_auc')
    n_jobs = kwargs.pop('n_jobs', 10)

    # # handle for multiclass targets
    # if train_targets.nunique() > 2:
    #     logger.info("Multiclass targets detected. Using one-vs-rest scheme if `scoring` is not specified.")
    #     scoring = 'roc_auc_ovr' if scoring == 'roc_auc' else scoring
    #
    #     available_scorings = (
    #         'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'
    #     )

    best_params = {}
    best_estimators = {}
    results = {}
    predict_table = [test_targets] if not test_targets is None else []
    splitter = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    for key, param_grid in param_grid_dict.items():
        logger.info("{:-^100}".format(f"Fitting for {key}"))
        split = splitter.split(train_targets, train_targets.values.ravel())
        grid = GridSearchCV(clf, n_jobs=n_jobs, param_grid=param_grid, scoring=scoring,
                            cv=split, verbose=MNTSLogger.is_verbose, **kwargs
                            )
        grid.fit(train_features.values, train_targets.values.ravel())

        # Isolate best estimator
        best_estimators[f"{key}"] = grid.best_estimator_
        best_params[key] = grid.best_params_

        # Get training scores for the best estimator
        train_score = grid.score(train_features.values,
                                 y = train_targets.values)
        results[f'{key} (train)'] = train_score

        # If testing set exist, evalute performance on the testing set
        if not (test_features is None or test_targets is None):
            y = grid.predict(test_features.values)
            test_score = metrics.roc_auc_score(test_targets.values.ravel(),
                                               y)
            results[f'{key}'] = test_score
            predict_table.append(pd.Series(y, index=test_targets.index, name=f"{key}"))
        logger.info("{:-^100}".format(f"Done for {key}"))

    # Collect the predictions of the testing sets
    if not test_targets is None and not test_features is None:
        predict_table = pd.concat(predict_table, axis=1)
    else:
        predict_table = pd.DataFrame()  # Return empty dataframe if something went wrong
    results = pd.Series(results)
    return best_params, results, predict_table, best_estimators


def model_building_(train_features: pd.DataFrame,
                    train_targets: pd.DataFrame,
                    test_features: Optional[pd.DataFrame] = None,
                    test_targets: Optional[pd.DataFrame] = None) -> Tuple[dict]:
    r"""
    Train model without grid-search. If test set data is provided, also evalute the performance of the built model using
    the test set data. This function is essentially same as `cv_grid_search` except there are no grid search.

    Args:
        See :func:`cv_grid_search`
    """
    raise NotImplementedError("This function shouldn't be used")
    # TODO: This function is not finished and require adding standard scalar here.
    logger = MNTSLogger['model-building']
    models = {
        'Logistic Regression': linear_model.LogisticRegression(penalty='elasticnet',
                                                               solver='saga',
                                                               C=1 / .02,
                                                               l1_ratio=.5,
                                                               max_iter=3000,
                                                               verbose=True),
        'Support Vector Regression': svm.SVR(kernel='linear',
                                             tol=5E-4,
                                             C=10,
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
                                                          C=10,
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

    def cv_fit(self,
               train_features: pd.DataFrame,
               train_targets: pd.DataFrame,
               test_features: pd.DataFrame = None,
               test_targets: pd.DataFrame = None,
               **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
        r"""
        Perform cv_grid search to determine the best estimator and the best hyper-parameters

        Args:
            See :func:`cv_grid_search` for more.
        """
        if not self.check_dimension(train_features, train_targets):
            # Try to match dimensions for convenience
            self._logger.warning("Miss-match found for train data.")
            train_features, train_targets = self.match_dimension(train_features, train_targets)
        if not (test_features is None or test_targets is None):
            if not self.check_dimension(test_features, test_targets):
                self._logger.warning("Miss-match found for test data.")
                test_features, test_targets = self.match_dimension(test_features, test_targets)

        best_params, results, predict_table, best_estimators = cv_grid_search(train_features,
                                                                              train_targets,
                                                                              test_features,
                                                                              test_targets,
                                                                              verbose=self.verbose,
                                                                              **kwargs)
        self.saved_state['estimators'] = best_estimators
        self.saved_state['best_params'] = best_params
        return results, predict_table

    def fit(self,
            train_features: pd.DataFrame,
            train_targets : pd.DataFrame,
            test_features : Optional[pd.DataFrame] = None,
            test_targets  : Optional[pd.DataFrame] = None,
            **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
        r"""Fit the estimator with the best performance

        Args:
            train_features (pd.DataFrame):
                Training features. Rows should be data points and columns should be features.
            train_targets (pd.DataFrame):
                Training ground-truth. Each row should be a data point.
            test_features (pd.DataFrame):
                Testing features. Rows should be data points and columns should be features. If this is supplied,
                the performance over the testing set would be computed. Default to `None`
            test_targets (pd.DataFrame):
                Testing ground. Must be supplied with `test_features.

        Returns:
            Tuple[pd.Series, pd.DataFrame]
                The series is the performance, the data frame is the results of the prediction. If `test_features` or
                `test_target` was not supplied, this method returns `None`.

        """
        # if there are differences in the saved state and the requested classifiers
        if kwargs.get("classifiers", None) is not None and self.saved_state['estimators'] is not None:
            # run cv fit again if the classifier is not in the saved state
            if isinstance(kwargs['classifiers'], str):
                if kwargs["classifiers"] not in self.saved_state['estimators']:
                    self._logger.warning(f"Requested classifier is not in saved state, running cv_fit again.")
                    self.saved_state['estimators'] = None
                    self.saved_state['best_params'] = None
            elif isinstance(kwargs['classifiers'], (list, tuple)):
                if len(list(set(kwargs['classifiers']) - set(self.saved_state['estimator'].keys()))) != 0:
                    self._logger.warning(f"Requested classifier mismatch with saved state, running cv_fit again.")
                    self.saved_state['estimators'] = None
                    self.saved_state['best_params'] = None

        # make sure the features selected aligns for testing data
        if not test_features is None:
            test_features = test_features[train_features.columns]
            # assert they are equal
            if not test_features.columns.identical(train_features.columns):
                raise KeyError("Features in `test_features` cannot match `train_features`.")

        if self.saved_state['estimators'] is None or self.saved_state['best_params'] is None:
            return self.cv_fit(train_features, train_targets, test_features, test_targets, **kwargs)
        else:
            self._logger.info("Found trained parameters and estimators, fitting them again...")
            best_params = self.saved_state['best_params']
            best_estimators = self.saved_state['estimators']

            results = {}
            predict_table = [test_targets] if not test_targets is None else []
            for key, estimator in best_estimators.items():
                estimator.fit(train_features, train_targets.values.ravel())
                if not (test_features is None or test_targets is None):
                    train_y = estimator.predict(train_features.values)
                    train_score = metrics.roc_auc_score(y_true=train_targets.values.ravel(),
                                                        y_score=train_y)
                    results[f'{key} (train)'] = train_score
                    if test_targets is not None and test_features is not None:
                        y = estimator.predict(test_features.values)
                        test_score = metrics.roc_auc_score(y_true=test_targets.values.ravel(),
                                                           y_score=y)
                        results[f'{key}'] = test_score
                    predict_table.append(pd.Series(y, index=test_targets.index, name=f"{key}"))
            predict_table = pd.concat(predict_table, axis=1)
            results = pd.Series(results)
            return results, predict_table

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        r"""
        Predict the class supplied features.

        Args:
            features (pd.DataFrame):
                Selected features. The rows should be datapoints and the columns should be features.
        Returns:
            pd.DataFrame
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

    def match_dimension(self,
                        X: pd.DataFrame,
                        y: pd.DataFrame,
                        raise_error: Optional[bool] = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r"""Match the rows of X and y.

        Args:
            X (pd.DataFrame):
                Features. Rows should be data points and columns should be features.
            y (pd.DataFrame):
                Train targets. Each row should be a data point.
            raise_error (bool, optional):
                Raise error if rows do not match

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                X and y with matched indece

        Raises:
            IndexError:
                If `raise_error` is True, an error is raised when the input feature indices do not match
                the target indices.

        """
        overlap = X.index.intersection(y.index)
        missing_feat = X.index.difference(overlap)
        missing_targ = y.index.difference(overlap)

        if len(missing_feat) > 0:
            msg = ','.join([str(r) for r in missing_feat])
            if raise_error:
                raise IndexError(f"Rows removed from features: {msg}")
            else:
                self._logger.info(f"Rows removed from features: {msg}")
        if len(missing_targ) > 0:
            msg = ','.join([str (r) for r in missing_targ])
            if raise_error:
                raise IndexError(f"Rows removed from targets: {msg}")
            else:
                self._logger.info(f"Rows removed from targets: {msg}")

        return X.loc[overlap], y.loc[overlap]
