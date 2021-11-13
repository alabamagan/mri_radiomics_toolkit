import mnts.mnts_logger
import pandas as pd
import pingouin as pg

from pathlib import Path

from mnts.mnts_logger import MNTSLogger
from tqdm.auto import *
from typing import Union, Optional, Iterable, List, Callable, Tuple, Dict

from sklearn.model_selection import *
from sklearn import *
from scipy.stats import *

import numpy as np
import multiprocessing as mpi
from tqdm.auto import *
from functools import partial
from RENT import RENT, stability

global logger


def cv_grid_search(train_features: pd.DataFrame,
                   train_targets: pd.DataFrame,
                   test_features: Optional[pd.DataFrame] = None,
                   test_targets: Optional[pd.DataFrame] = None) -> List[Union[pd.DataFrame, Dict]]:
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
        ('classification', 'passthrough')
    ])

    # Construct tests to perform
    param_grid_dict = {
        'Support Vector Machine': {
            'classification': [svm.SVR(tol=1E-4, max_iter=3500)],
            'classification__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classification__C': [0.1, 1, 10],
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
                                                               solver='saga', tol=1E-4,
                                                               max_iter=3500,
                                                               verbose=True)],
            'classification__C': [0.1, 1, 10, 100],
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
                                                           verbose=True)],
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
    predict_table = pd.concat(predict_table, axis=1)

    return best_params, results, predict_table, best_estimators


    import matplotlib.pyplot as plt
    import seaborn as sns

    df_cvres = []
    for i in range(15):
        grid = GridSearchCV(clf, n_jobs=5, param_grid=param_grid,
                            scoring='roc_auc', cv=split)
        X = grid.fit(feats_a.T.values, train_targets.loc[feats_a.columns].to_numpy().ravel())
        logger.info(f"Best_score: {grid.best_score_}")
        logger.info(f"Best_params: {grid.best_params_}")
        logger.info(f"Best_estimator coef: {grid.best_estimator_}")
        cvres = pd.DataFrame(grid.cv_results_)
        df_cvres.append(cvres)


    df_cvres = pd.concat(df_cvres)
    df_cvres = df_cvres.reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.lineplot(data=df_cvres,
                 x='param_classification__alpha',
                 y='mean_test_score',
                 hue='param_classification__l1_ratio',
                 ax=ax[0])
    sns.lineplot(data=df_cvres,
                 x='param_classification__l1_ratio',
                 y='mean_test_score',
                 hue='param_classification__alpha',
                 ax=ax[1])
    plt.show()
    return grid


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

def main():
    global logger
    logger = MNTSLogger('Log/run_model_building.log', verbose=True)

    # |=========================================|
    # | 1. Feature extraction (done externally) |
    # |=========================================|
    # Load features
    #--------------
    # Features should have their col indices as patient identifier and row as features
    # features_a = Path('./extracted_features_1st.xlsx')
    # features_b = Path('./extracted_features_2nd.xlsx')
    features_a = Path('./pyrad_features_1st_nyul.xlsx')
    features_b = Path('./pyrad_features_2nd_nyul.xlsx')
    features_a = pd.read_excel(str(features_a), index_col=(0, 1, 2))
    features_a.index.rename(['Pre-processing', 'Feature_Group', 'Feature_Name'], inplace=True)
    features_b = pd.read_excel(str(features_b), index_col=(0, 1, 2))
    features_b.index.rename(['Pre-processing', 'Feature_Group', 'Feature_Name'], inplace=True)

    # Target status should have patients identifier as row index and status as the only column
    status = Path('./data/v2-datasheet.csv')
    status = pd.read_csv(str(status), index_col=0)
    status.columns = ['Status']

    # Split the features into 5-folds with stratification to the status of malignancy
    splitter = StratifiedKFold(n_splits=5, shuffle=True)
    outer_dict = {}
    outer_list = []
    for k in range(50):
        logger.info(f"=== Running {k} ===")
        selected_features_list = []
        splits = splitter.split(status.index, status[status.columns[0]])
        fold_configs = {}
        for fold, (train_index, test_index) in enumerate(splits):
            train_ids, test_ids = [str(status.index[i]) for i in train_index], \
                                  [str(status.index[i]) for i in test_index]
            train_ids.sort()
            test_ids.sort()
            fold_configs[fold] = (train_ids, test_ids)

        #!! Loop each fold
        for fold, (train_ids, test_ids) in fold_configs.items():
            # Seperate traing and test features
            train_feat_a = features_a.T.loc[train_ids].T
            train_feat_b = features_b.T.loc[train_ids].T
            test_feat_a = features_a.T.loc[test_ids].T
            test_feat_b = features_b.T.loc[test_ids].T

            # |======================|
            # | 2. Feature selection |
            # |======================|

            selected_features = run_features_selection(train_feat_a, train_feat_b, status, n_trials=500)
            selected_features.to_excel(f'./output/selected_features_{fold}.xlsx')
            #save the features
            selected_features_list.append(['__'.join(i) for i in selected_features.index])

        outer_list.extend(selected_features_list)
        union_features = set.union(*[set(i) for i in selected_features_list])
        features_frequencies = {i: 0 for i in union_features}
        for i in union_features:
            for j in selected_features_list:
                if i in j:
                    features_frequencies[i] += 1
        features_frequencies = pd.Series(features_frequencies, name=f'frequencies_{k}')
        outer_dict[k] = features_frequencies
        logger.info(f"Feature_Summary: {features_frequencies.to_string()}")
        logger.info(f"=== Done {k} === ")

    # For outer loop
    union_features = set.union(*[set(i) for i in outer_list])
    features_frequencies = {i: 0 for i in union_features}
    for i in union_features:
        for j in outer_list:
            if i in j:
                features_frequencies[i] += 1
    features_frequencies = pd.Series(features_frequencies, name='frequencies_all')
    logger.info(f"Feature_Summary: {features_frequencies.to_string()}")

    features_frequencies = features_frequencies.to_frame()
    for k in outer_dict:
        _right = outer_dict[k].to_frame()
        features_frequencies = features_frequencies.join(_right, how='outer')
    features_frequencies.fillna(0, inplace=True)
    features_frequencies.to_excel(f"./output/selected_feat_freq.xlsx")


        # |===================|
        # | 3. Model building |
        # |===================|

        # Build models out of the training group

        # Test model using the testing group

        # Compute fold-wise results

if __name__ == '__main__':
    main()
