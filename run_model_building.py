import mnts.mnts_logger
import pandas as pd
import pingouin as pg

from pathlib import Path

from mnts.mnts_logger import MNTSLogger
from tqdm.auto import *
from typing import Union, Optional, Iterable, List, Callable

from sklearn.model_selection import *
from sklearn import *
from scipy.stats import *

import numpy as np
import multiprocessing as mpi
from tqdm.auto import *
from functools import partial
from RENT import RENT, stability

global logger

def compute_ICC(featset_A: pd.DataFrame,
                featset_B: pd.DataFrame) -> pd.DataFrame:
    r"""
    Compute the ICC of feature calculated from two sets of data. Note that this function does not check the senity of
    the provided dataframe. The dataframe should look like this:

    +----------------+---------------+----------------------------------+-----------+-----------+-----------+
    | Pre-processing | Feature_Group |           Feature_Name           | Patient 1 | Patient 2 | Patient 3 |
    +================+===============+==================================+===========+===========+===========+
    |    original    |     shape     | Elongation                       | 0.736071  | 0.583376  | 0.842203  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | Maximum2DDiameterSlice           | 38.69644  | 40.13999  | 42.83211  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | Maximum3DDiameter                | 39.47085  | 53.00941  | 44.86157  |
    +                +---------------+----------------------------------+-----------+-----------+-----------+
    |                |   firstorder  | 10Percentile                     | 80        | 84        | 131       |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | 90Percentile                     | 167       | 198       | 221       |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | Mean                             | 125.0034  | 141.5715  | 177.9713  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | Median                           | 126       | 143       | 182       |
    +                +---------------+----------------------------------+-----------+-----------+-----------+
    |                |      glcm     | Autocorrelation                  | 32.9823   | 42.24437  | 60.84951  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | ClusterProminence                | 201.2033  | 370.5453  | 213.6482  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | ClusterShade                     | 3.085583  | -2.73874  | -7.56395  |
    +                +---------------+----------------------------------+-----------+-----------+-----------+
    |                |     glrlm     | GrayLevelNonUniformity           | 3828.433  | 5173.855  | 6484.706  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | RunVariance                      | 5.13809   | 2.925426  | 5.239695  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | ShortRunEmphasis                 | 0.574203  | 0.629602  | 0.545728  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | ShortRunHighGrayLevelEmphasis    | 18.43039  | 25.55685  | 32.46986  |
    +                +               +----------------------------------+-----------+-----------+-----------+
    |                |               | ShortRunLowGrayLevelEmphasis     | 0.03399   | 0.030848  | 0.012844  |
    +----------------+---------------+----------------------------------+-----------+-----------+-----------+

    .. note::
        * This function requires that the first three columns are `pd.MultiIndex`. The name of the headers should be
          exactly "Pre-processing", "Feature_Group" and "Feature_Name".
        * This function takes only ICC1 from the pacakge `pingouin`.

    Args:
        featset_A (pd.DataFrame):
            Dataframe that contains features computed and to be compared with the other set `featset_B`.
        featset_B (pd.DataFrame):
            Dataframe that should have the exact same row indices as `featset_A`.

    Returns:
        pd.DataFrame
    """
    assert featset_A.index.nlevels == featset_B.index.nlevels == 3, \
        "The dataframe should be arranged with multi-index: ['Pre-processing', 'Feature_Group', 'Feature_Name']"
    logger = MNTSLogger['ICC']

    featset_A['Segmentation'] = "A"
    featset_B['Segmentation'] = "B"
    features = pd.concat([featset_A, featset_B])
    features.set_index('Segmentation', append=True, inplace=True)
    features = features.reorder_levels([3, 0, 1, 2])

    # Drop some useless attributes:
    for i in ["A", "B"]:
        try:
            features.drop((i, 'diagnostics'), inplace=True)
        except:
            pass

    # Make each feature the unique index
    df = features.reset_index()
    df = df.melt(['Segmentation', 'Feature_Name', 'Feature_Group', 'Pre-processing'], var_name='Patient')
    df['Feature_Name'] = ['_'.join([a, b, c]) for a, b, c in
                          zip(df['Pre-processing'], df['Feature_Group'], df['Feature_Name'])]
    df.drop(columns='Pre-processing', inplace=True)
    df.set_index('Feature_Name', drop=True, inplace=True)

    # Make sure values are float numbers
    df['value'] = df['value'].astype('float')
    df['Patient'] = df['Patient'].astype(str)
    df['Segmentation'] = df['Segmentation'].astype(str)

    # Compute ICC of features
    outted_features = []
    feature_names = list(set(df.index))
    icc_df = pd.DataFrame()
    pool = mpi.Pool(mpi.cpu_count())
    res = pool.starmap_async(partial(pg.intraclass_corr, targets='Patient', raters='Segmentation', ratings='value'),
                             [[df.loc[f]] for f in feature_names])
    pool.close()
    pool.join()

    results = res.get()
    for i, f in enumerate(tqdm(results)):
        logger.info(f"Computing: {feature_names[i]}")
        icc = f
        icc['Feature_Name'] = feature_names[i]
        icc.set_index('Feature_Name', inplace=True)
        icc.set_index('Type', append=True, inplace=True)
        icc_df = icc_df.append(icc)

    # Select only ICC1
    KK = icc_df.drop(["ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"], level=1)
    KK = KK.reset_index()

    # Obtain the index back
    MIndex = pd.MultiIndex.from_tuples([a.split('_') for a in KK['Feature_Name']])
    KK.index = MIndex
    KK = KK.drop("Feature_Name", axis=1)
    KK.index.set_names(['Pre-processing', 'Feature_Group', 'Feature_Name'], inplace=True)
    KK = KK.sort_index(level=0)
    return KK, outted_features

def ICC_thres_filter(featset_A: pd.DataFrame,
                     featset_B: pd.DataFrame,
                     ICC_threshold: Optional[float] = 0.9) -> pd.DataFrame:
    r"""
    Wrapper function for convinience
    `featset_A/B` columns should be patient identifiers, and rows should be features.
    """
    ICCs = compute_ICC(featset_A, featset_B)[0]
    ICCs = ICCs.loc[ICCs['ICC'] >= 0.9].index
    return ICCs

def T_test_filter(features: pd.DataFrame,
                  target: pd.DataFrame) -> pd.DataFrame:
    r"""
    Return the p-values of the t-test.

    * `features` should follow the same convention as that in `get_feat_list_thres_w_ICC`
    * `target` should be a dataframe with status column labelled as 'Status' with value `0/1` or `True/False` and
      patients identifiers as row index.

    Args:
        features (pd.DataFrame):
            The features to be tested. Should fallow the format in :func:`get_feat_list_thres_w_ICC`
        target (pd.DataFrame):
            The status of the patients.

    Returns:
        pd.DataFrame
    """
    classes = list(set(target['Status']))

    # Get a list of ids for patients with different classes
    patients_lists = {c: target.loc[target['Status'] == c].index.tolist() for c in classes}

    T_test_pvals = []
    for f in features.index:
        _patients_A = features[features.columns.intersection(patients_lists[classes[0]])]
        _patients_B = features[features.columns.intersection(patients_lists[classes[1]])]
        _f_A = _patients_A.loc[f].astype('float32')
        _f_B = _patients_B.loc[f].astype('float32')

        pval = [pg.normality(_f.to_numpy())['pval'] for _f in [_f_A, _f_B]]
        pval = pd.concat(pval, axis=0).max()

        _is_norm = pval > .05
        if _is_norm:
            t_pval = float(pg.ttest(_f_A, _f_B)['p-val'])
            test_name = 'Student t-test'
        else:
            t_pval = float(pg.mwu(_f_A, _f_B)['p-val'])
            test_name = 'Mann-Whitney U'

        s = pd.DataFrame([[test_name, t_pval]], columns=['test', 'pval'])
        s.index = ['_'.join(f)]
        T_test_pvals.append(s)
    T_test_pvals = pd.concat(T_test_pvals, axis=0)
    T_test_pvals.index = features.index

    return T_test_pvals

def initial_feature_filtering(features_a: pd.DataFrame,
                              features_b: pd.DataFrame,
                              targets: pd.DataFrame,
                              ):
    r"""
    Perform feature selction following the steps:
    1. p-values of features < .001
    2. Features with extremely low variance removed (because they are likely to be errors or useless
    2. ICC of features > 0.9
    3. Dimensional reduction
    This assume features are already normalized.

    .. note::
        * Note 1: Variance threshold remove features with very low variance, probably all has the same values and is
          basically useless (e.g., all are 0 or 1).

    Args:
        features_a:
            Features extracted using segmentation set A
        features_b:
            Features extracted using segmentation set B
        targets:

    Returns:

    """
    logger = MNTSLogger['initial_feature_filtering']

    # Drop features known to be useless
    for features in [features_a, features_b]:
        logger.info("Dropping 'Diganostics' column.")
        features.drop('diagnostics', inplace=True)

    # Drop features with extremely low variance (i.e., most are same value)
    logger.info("Dropping features with low variance...")
    var_filter = feature_selection.VarianceThreshold(threshold=.95*(1-.95))
    var_feats_a = var_filter.fit_transform(features_a.T)
    var_feats_a_index = features_a.index[var_filter.get_support()]
    var_feats_b = var_filter.fit_transform(features_b.T)
    var_feats_b_index = features_b.index[var_filter.get_support()]
    # Only those that fulfilled the variance threshold in both set of features are to be included
    mutual_features = set(var_feats_a_index) & set(var_feats_b_index)
    logger.info(f"{len(mutual_features)} features kept: \n{mutual_features}")
    logger.info(f"{len(set(features_a.index) - set(mutual_features))} features discarded: \n"
                f"{set(features_a.index) - set(mutual_features)}")

    # Filter features by ICC
    logger.info("Dropping features by their intra-observer segmentation ICC")
    _icc90_feats = ICC_thres_filter(features_a.loc[mutual_features],
                                    features_b.loc[mutual_features], 0.9)

    # Compute p-values of features
    icc90_feats_a = features_a.loc[_icc90_feats]
    icc90_feats_b = features_b.loc[_icc90_feats]
    #!! Temp, enable upper commented lines when done, the lower two line saves time by skipping ICC
    # icc90_feats_a = features_a.loc[mutual_features]
    # icc90_feats_b = features_b.loc[mutual_features]

    # Filter out features with not enough significance
    p_thres = .001
    logger.info(f"Dropping features using T-test with p-value: {p_thres}")
    pvals_feats_a = T_test_filter(icc90_feats_a, targets)
    pvals_feats_b = T_test_filter(icc90_feats_b, targets)
    feats_a = icc90_feats_a.loc[(pvals_feats_a['pval'] < p_thres).index]
    feats_b = icc90_feats_b.loc[(pvals_feats_b['pval'] < p_thres).index]

    return feats_a, feats_b

def supervised_features_selection(features: pd.DataFrame,
                                  targets: pd.DataFrame,
                                  alpha: Union[float, Iterable[float]],
                                  l1_ratio: Union[float,Iterable[float]],
                                  *args,
                                  criteria_threshold: Iterable[float] = (0.8, 0.8, 0.99),
                                  n_features: int = None,
                                  n_splits: int = 5,
                                  n_trials: int = 100,
                                  boosting: bool = True,
                                  **kwargs):
    r"""
    Use RENT [1] to select features. Essentially `n_trials` models were trained using the features and targets, and
    the coefficients of the models for these features were recorded as a [`n_trials` by `features.shape[0]`] matrix.
    The quality of each feature were evaluted through these coefficient using three criteria.


    .. notes::
        * [1] Jenul, Anna, et al. "RENT--Repeated Elastic Net Technique for Feature Selection."
              arXiv preprint arXiv:2009.12780 (2020).


    Args:
        features (pd.DataFrame):
            Columns should be samples and rows should be features.
        targets:
            Row should be samples and a column 'Status' should hold the class.
        n_features:

        n_splits (int):
            Split ratios of train-test group for random bootstrapping.
        n_trials (int):
            Number of runs to select the final model features from. Equivalent to $K$ in the original paper.

    Returns:

    """
    logger = MNTSLogger['sup-featselect']
    #|==========|
    #| Run RENT |
    #|==========|

    # C in RENT is the inverse of regularization term alpha in scipy
    C = 1/alpha

    # Convert alpha and l1_ratio to list for RENT's convention
    C = np.asarray([C]) if type(C) == float else C
    l1_ratio = np.asarray([l1_ratio]) if type(l1_ratio) == float else l1_ratio

    # Align targets and features, assume target.index $\in$ features.columns
    if not targets.index.to_list() == features.columns.to_list():
        logger.warning("Discrepancy found in case identifiers in target and features, trying to align them!")
        _targets = targets.loc[features.columns]
    else:
        _targets = targets

    # Convert model names into strings
    _features_names = ['__'.join(i) for i in features.index]
    _ori_index = features.index
    _map = {_f: o for _f, o in zip(_features_names, _ori_index)}
    features.index = _features_names


    model = RENT.RENT_Regression(data=pd.DataFrame(features.T),
                                 target=_targets[_targets.columns[0]].to_numpy().ravel(),
                                 feat_names=_features_names,
                                 C=C,
                                 l1_ratios=l1_ratio,
                                 autoEnetParSel=False,
                                 poly='OFF',
                                 testsize_range=(1/float(n_splits), 1/float(n_splits)),
                                 K=n_trials,
                                 random_state=0,
                                 verbose=1,
                                 scale=False,
                                 boosting=boosting) # BRENT or RENT
    model.train()
    selected_features = model.select_features(*criteria_threshold)
    selected_features = features.index[selected_features]
    logger.info(f"RENT features: {selected_features}")

    #|=========================|
    #| Feature select features |
    #|=========================|
    # Conver the coeffients to weights
    coefs_df = pd.DataFrame(np.concatenate(model._weight_list, axis=0).T, index=_features_names)
    coefs_df = coefs_df.loc[selected_features]

    # normalize the coefficient vector of each model to normal vector (magnitude = 1)
    coefs_df = coefs_df/ coefs_df.pow(2).sum().pow(.5)

    # rank the coefs based on their mean and variance, large |mean| and small variance is desired.
    mean_ranks = (coefs_df.shape[0] - coefs_df.T.mean().argsort()) # smaller rank is better
    var_ranks = coefs_df.T.std().argsort() # smaller rank is better
    avg_ranks = (mean_ranks + 0.5 * var_ranks) / 1.5

    # remove lower rank features
    _init_features_index = avg_ranks.sort_values()[:n_features]

    # Construct the suggested features
    features.index = _ori_index
    selected_features = pd.MultiIndex.from_tuples([_map[i] for i in _init_features_index.index])
    out_features = features.loc[selected_features]
    return out_features.sort_index(level=0)


def _check_cor_mat(mat, test: Callable = None):
    r"""
    Return true if any non-diagnoal element of the mat return true for the test.
    Default to test if any element > .05
    """
    if test is None:
        test = lambda x: np.abs(x) > .5

    if not isinstance(test, np.ndarray):
        mat = np.asarray(mat)

    return test(mat[~np.eye(mat.shape[0], dtype=bool)].flatten()).any()


def features_normalization(features):
    r"""
    Normalize the features using `StandardScaler`.
    The input columns should be case and the rows should be features
    """
    normed = preprocessing.StandardScaler().fit_transform(features.T)
    normed = pd.DataFrame(normed.T, columns=features.columns, index=features.index)
    return normed

def run_features_selection(features_a: pd.DataFrame,
                           features_b: pd.DataFrame,
                           targets: pd.DataFrame,
                           n_trials = 500):
    r"""
    Features selection wrapper function, execute:
    1. initial feature filtring `initial_feature_filtering`
    2. normalization of features `features_normalization`
    3. RENT/BRENT `supervised_features_selection`

    Args:
        features_a:
            Features extracted using segmentation set A
        features_b:
            Features extracted using segmentation set B
        targets:

    Returns:

    """
    logger = MNTSLogger['model_building']

    # Initial feature filtering using quantitative methods
    feats_a, feats_b = initial_feature_filtering(features_a, features_b, targets)
    # Note: Because initial feature filtering relies on variance and ICC, it is not proper to normalize the data prior
    #       to this because it alters the mean and variance of the features.


    # Normalize remaining features for data-driven feature selection
    feats_a, feats_b = features_normalization(feats_a), features_normalization(feats_b)

    # Supervised feature selection using RENT
    alpha = np.asarray([0.01, 0.1, 1])
    l1_ratio = np.asarray([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    feats_a_out = supervised_features_selection(feats_a, targets,
                                                0.02,
                                                0.5,
                                                criteria_threshold=(0.9, 0.5, 0.99),
                                                n_trials=n_trials,
                                                n_features=25)
    logger.info(f"in features: \n{feats_a.sort_index(level=0)}")
    logger.info(f"The selected features: \n{feats_a_out}")
    # classification models
    return feats_a_out


def run_cv_grid_search(features: pd.DataFrame,
                       targets: pd.DataFrame):
    r"""
    Grid search for best hyper-parameters for the following linear models:
      * SVM
      * Logistic regresion
      * Random forest
      * K-nearest-neighbour
      * Elastic Net

    Args:
        features:
        targets:

    Returns:

    """
    clf = pipeline.Pipeline([
        ('classification', 'passthrough')
    ])
    param_grid =[
        {
            'classification': [svm.SVR(tol=1E-4, max_iter=3500)],
            'classification__C': [0.1, 1, 10],
            'classification__degree': [3, 5, 7, 9],
            'classification__epsilon': [1, 0.1, 0.01]
        },
        {
            'classification': [linear_model.ElasticNet(tol=1E-4, max_iter=3500)],
            'classification__alpha': [.02, .002],
            'classification__l1_ratio': [0.2, 0.5, 0.8]
        },
        {
            'classification': []
        }
    ]


    import matplotlib.pyplot as plt
    import seaborn as sns

    df_cvres = []
    for i in range(15):
        split = splitter.split(feats_a.columns, targets.loc[feats_a.columns][targets.columns[0]])
        grid = GridSearchCV(clf, n_jobs=5, param_grid=param_grid,
                            scoring='roc_auc', cv=split)
        X = grid.fit(feats_a.T.to_numpy(), targets.loc[feats_a.columns].to_numpy().ravel())
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
