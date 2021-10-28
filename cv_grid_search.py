from sklearn import *
from sklearn.model_selection import *
import numpy as np
import matplotlib.pyplot as plt
from mnts.mnts_logger import MNTSLogger
import pandas as pd

global logger


def plot_performance_vs_hyperparams(clf,
                                    param_grid,
                                    features,
                                    targets,
                                    n_split=7,
                                    n_fold=15):
    splitter = StratifiedKFold(n_splits=n_split, shuffle=True)
    logger = MNTSLogger['cv_grid_search_plot']
    df_cvres = []
    for i in range(n_fold):
        split = splitter.split(features.columns, targets.loc[features.columns][targets.columns[0]])
        grid = GridSearchCV(clf, n_jobs=5, param_grid=param_grid,
                            scoring='roc_auc', cv=split)
        X = grid.fit(features.T.to_numpy(), targets.loc[features.columns].to_numpy().ravel())
        cvres = pd.DataFrame(grid.cv_results_)
        df_curves.append(cvres)

