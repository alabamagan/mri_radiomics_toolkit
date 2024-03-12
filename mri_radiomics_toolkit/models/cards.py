"""
Filename: cards.py
Author: lwong
Date Created: 2023/8/7
Date Last Modified: 2023/8/7
Description:
    This file recorded configurations used for model training as well as hyperparameter
    tuning. They are manually selected and
"""

from sklearn import linear_model, svm, neural_network, ensemble, neighbors
from mnts.mnts_logger import MNTSLogger

verbose = MNTSLogger.is_verbose

# default setting for grid search
default_cv_grid_search_card = {
    'Support Vector Regression': {
        'classification': [svm.SVR(tol=1E-4, max_iter=-1)],
        'classification__kernel': ['linear', 'poly', 'rbf'],
        'classification__C': [1, 100, 1000],
        'classification__degree': [3, 5, 7],
        'classification__epsilon': [1, 0.1, 0.01]
    },
    'Logistic Regression': {
        'classification': [linear_model.LogisticRegression(penalty='elasticnet',
                                                           solver='saga', tol=1E-5,
                                                           max_iter=5500,
                                                           verbose=verbose)],
        'classification__C': [0.1, 1, 10, 100, 1000],
        'classification__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'Random Forest': {
        'classification': [ensemble.RandomForestRegressor(n_estimators=50)],
        'classification__criterion': ['squared_error', 'poisson']
    },
    'Perceptron': {
        'classification': [neural_network.MLPRegressor(learning_rate='adaptive',
                                                       tol=1E-4,
                                                       max_iter=5000,
                                                       verbose=verbose)],
        'classification__hidden_layer_sizes': [(100), (20, 50, 100), (100, 50, 20)],
        'classification__learning_rate_init': [1, 0.1, 1E-2, 1E-3, 1E-4]
    },
    'KNN': {
        'classification': [neighbors.KNeighborsRegressor(n_jobs=5)],
        'classification__n_neighbors': [3, 5, 10, 20],
    }
}

# default setting for grid search
multi_class_cv_grid_search_card = {
    'Support Vector Regression': {
        'classification': [svm.SVC(tol=1E-4, max_iter=-1, probability=True)],
        'classification__kernel': ['linear', 'poly', 'rbf'],
        'classification__C': [1, 100, 1000],
        'classification__degree': [3, 5, 7],
    },
    'Logistic Regression': {
        'classification': [linear_model.LogisticRegression(penalty='elasticnet',
                                                           solver='saga', tol=1E-5,
                                                           max_iter=5500,
                                                           verbose=verbose)],
        'classification__C': [0.1, 1, 10, 100, 1000],
        'classification__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'Random Forest': {
        'classification': [ensemble.RandomForestClassifier(verbose=verbose)],
        'classification__n_estimators': [20, 30, 50, 100],
        'classification__criterion': ['gini', 'entropy'],
        'classification__max_depth': [None, 5, 10, 20]
    },
    # 'Perceptron': {
    #     'classification': [neural_network.MLPRegressor(learning_rate='adaptive',
    #                                                    tol=1E-4,
    #                                                    max_iter=5000,
    #                                                    verbose=verbose)],
    #     'classification__hidden_layer_sizes': [(100), (20, 50, 100), (100, 50, 20)],
    #     'classification__learning_rate_init': [1, 0.1, 1E-2, 1E-3, 1E-4]
    # },
    # 'KNN': {
    #     'classification': [neighbors.KNeighborsRegressor(n_jobs=5)],
    #     'classification__n_neighbors': [3, 5, 10, 20],
    # }
}


