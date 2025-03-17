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

# default setting for grid search, this param grid is designed for classification.
default_cv_grid_search_card = {
    'Support Vector Machine': {
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
        'classification': [ensemble.RandomForestClassifier()],
        'classification__n_estimators': [10, 20, 50, 100],
        'classification__criterion': ['entropy', 'log_loss']
    },
}

multi_class_cv_grid_search_card = {
    'Support Vector Machine': {
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
        'classification': [ensemble.RandomForestClassifier()],
        'classification__n_estimators': [10, 20, 50, 100],
        'classification__criterion': ['entropy', 'log_loss']
    },

}