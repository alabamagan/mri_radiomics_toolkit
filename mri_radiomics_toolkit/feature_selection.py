import multiprocessing as mpi
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Iterable
from functools import partial

import joblib
import pandas as pd
import pingouin as pg
import sklearn
import warnings
from mnts.mnts_logger import MNTSLogger
from scipy.stats import *
from sklearn import feature_selection as skfs
from sklearn import linear_model, preprocessing
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import *

from RENT import RENT

__all__ = ['FeatureSelector', 'supervised_features_selection', 'preliminary_feature_filtering', 'features_selection',
           'features_normalization']


def compute_ICC(featset_A: pd.DataFrame,
                featset_B: pd.DataFrame,
                ICC_form: Optional[str] = 'ICC2k') -> pd.DataFrame:
    r"""
    Compute the ICC of feature calculated from two sets of segmentation. Note that this function does not check the
    senity of the provided dataframe. The dataframe should look like this:

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
        * This function takes one of the six ICC forms of the package `pingouin`.

    Args:
        featset_A (pd.DataFrame):
            Dataframe that contains features computed and to be compared with the other set `featset_B`.
        featset_B (pd.DataFrame):
            Dataframe that should have the exact same row indices as `featset_A`.
        ICC_form (str, Optional):
            One of the six forms in ["ICC1"|"ICC2"|"ICC3"|"ICC1k"|"ICC2k"|"ICC3k"]. Default to be "ICC2k".

    Returns:
        pd.DataFrame
    """
    assert featset_A.index.nlevels == featset_B.index.nlevels == 3, \
        ("The dataframe should be arranged with multi-index: ['Pre-processing', 'Feature_Group', 'Feature_Name']. Got"
         f" {featset_A.index.nlevels} and {featset_B.index.nlevels}")
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
    outted_features = []    # TODO: depricate this
    feature_names = list(set(df.index))
    icc_df = []
    pool = mpi.Pool(mpi.cpu_count())
    res = pool.starmap_async(partial(pg.intraclass_corr, targets='Patient', raters='Segmentation', ratings='value'),
                             [[df.loc[f]] for f in feature_names])
    pool.close()
    pool.join()
    results = res.get()
    # #! DEBUG
    # results = []
    # for f in feature_names:
    #     icc = pg.intraclass_corr(df.loc[f], targets='Patient', raters='Segmentation', ratings='value')
    #     results.append(icc)
    #     if len(results) > 20:
    #         break


    for i, f in enumerate(tqdm(results)):
        logger.info(f"Computing: {feature_names[i]}")
        icc = f
        icc['Feature_Name'] = feature_names[i]
        icc.set_index('Feature_Name', inplace=True)
        icc.set_index('Type', append=True, inplace=True)
        icc_df.append(icc)
    icc_df = pd.concat(icc_df, axis=0)

    drop_this = ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]
    if not ICC_form in drop_this:
        raise AttributeError(f"ICC form can only be one of the following: {drop_this}")
    drop_this.remove(ICC_form)
    KK = icc_df.drop(drop_this, level=1)
    KK = KK.reset_index()

    # Obtain the index back
    MIndex = pd.MultiIndex.from_tuples([a.split('_') for a in KK['Feature_Name']])
    KK.index = MIndex
    KK = KK.drop("Feature_Name", axis=1)
    KK.index.set_names(['Pre-processing', 'Feature_Group', 'Feature_Name'], inplace=True)
    KK = KK.sort_index(level=0)
    return KK, outted_features


def filter_features_by_ICC_thres(featset_A: pd.DataFrame,
                                 featset_B: pd.DataFrame,
                                 ICC_threshold: Optional[float] = 0.9,
                                 ICC_form: Optional[str] = 'ICC2k') -> pd.MultiIndex:
    r"""Wrapper function for convenience
    `featset_A/B` columns should be patient identifiers, and rows should be features.

    Args:
        featset_A (pd.DataFrame):
            Dataframe that contains features computed and to be compared with the other set `featset_B`.
        featset_B (pd.DataFrame):
            Dataframe that should have the exact same row indices as `featset_A`.

    Retruns:
        pd.MultiIndex:
            Index of features that passed the ICC threshold.
    """
    ICCs = compute_ICC(featset_A, featset_B, ICC_form=ICC_form)[0]
    ICCs = ICCs.loc[ICCs['ICC'] >= 0.9].index
    return ICCs


def filter_features_by_T_test(features: pd.DataFrame,
                              target: pd.DataFrame) -> pd.DataFrame:
    r"""Return the p-values of the t-test.

    * `features` should follow the same convention as that in `get_feat_list_thres_w_ICC`
    * `target` should be a dataframe with status column labelled as 'Status' with value `0/1` or `True/False` and
      patients identifiers as row index.

    Args:
        features (pd.DataFrame):
            The features to be tested. Should follow the format in :func:`get_feat_list_thres_w_ICC`
        target (pd.DataFrame):
            The status of the patients.

    Returns:
        pd.DataFrame
    """
    try:
        T = target['Status']
    except KeyError:
        # If key error use the first column
        T = target[target.columns[0]]
    classes = list(set(T))
    if len(classes) == 1:
        raise ArithmeticError("There is only one class in the data set.")
    elif len(classes) > 2:
        raise ArithmeticError("T-test does not support multi-label classification.")

    # Get a list of ids for patients with different classes
    patients_lists = {c: target.loc[T == c].index.tolist() for c in classes}

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
            # If distribution is normal
            t_pval = pg.ttest(_f_A, _f_B)['p-val'].astype('float')
            test_name = 'Student t-test'
        else:
            # Otherwise
            t_pval = pg.mwu(_f_A, _f_B)['p-val'].astype('float')
            test_name = 'Mann-Whitney U'

        s = pd.Series([test_name, t_pval.item()], index=['test', 'pval'], name='_'.join(f))
        T_test_pvals.append(s)
    T_test_pvals = pd.concat(T_test_pvals, axis=1).T
    T_test_pvals.index = features.index

    return T_test_pvals


def filter_features_by_ANOVA(features: pd.DataFrame,
                             target: Union[pd.Series,pd.DataFrame]) -> pd.DataFrame:
    r"""Performs ANOVA (Analysis of Variance) on the given features and target classes and filters features
    based on the results.

    This function first checks the assumptions of ANOVA on the input data:
    - There are more than 2 classes in the target.
    - The target variable is normally distributed for each group.
    - The variances of the target variable are equal for each group.
    Violations of the normality and equal variance assumptions are reported as warnings.

    Then, it performs ANOVA for each feature and returns a DataFrame containing the p-values of the ANOVA tests.

    Args:
        features (pd.DataFrame):
            The input features as a pandas DataFrame. Each row represents an observation,
            and each column represents a feature. The data should be numeric and not contain any missing values.

        target (pd.Series or pd.DataFrame):
            The target classes as a pandas Series or DataFrame.
            If it is a DataFrame, it should only have one column. The target should not contain any missing values.

    Returns:
        pd.DataFrame:
            A DataFrame with one column named 'pval'. Each row corresponds to a feature in the input
            DataFrame, and the 'pval' column contains the p-value of the ANOVA test for that feature.

    Raises:
        ValueError: If the target has less than 3 unique classes.

    Warnings:
        RuntimeWarning: If the normality assumption is violated for any group in any feature.
        RuntimeWarning: If the equal variance assumption is violated for any feature.

    Examples:
    >>># Assume we have the following DataFrame
    >>>df = pd.DataFrame({
    >>>    'A': [1, 2, 3, 4, 5, 6],
    >>>    'B': [7, 8, 9, 10, 11, 12],
    >>>    'C': [13, 14, 15, 16, 17, 18]
    >>>})
    >>>
    >>># And this target
    >>>target = pd.Series([1, 2, 1, 2, 1, 2])
    >>>
    >>># Perform the feature filtering
    >>>filtered_features = filter_features_by_ANOVA(df, target)
    """
    logger = MNTSLogger['ANOVA']

    # Check ANOVA assumptions
    # Assumption 1: There are more than 2 classes.
    num_classes = target.nunique(dropna=True)[0]
    if num_classes < 3:
        msg = f"Target status has less than 3 classes."
        raise ValueError(msg)

    # Assumption 2: The target variable should be normally distributed for each group
    for group in np.unique(target):
        # filter data that belonged to `group`
        group_indices = target.loc[(target == group).values].index
        group_data = features[group_indices]

        # test normality of each feature in each group
        normality_test = features.apply(shapiro, axis=1)
        normality_test = normality_test.apply(pd.Series)
        normality_test.columns = ["Statistics", "pval"]
        logger.debug(f"Normality Shapiro results: \n{normality_test.to_string()}")

        # those with a significant p-value are not normally distributed
        violated_features = normality_test.loc[normality_test['pval'] <= .05].index

        # warn user about this
        if len(violated_features):
            logger.warning(f"Normality assumption violated for group '{group}', features: {violated_features}")

    # Assumption 3: The variances of the target variable should be equal for each group
    # group data by their classes first
    levene_res = []
    frames = []
    features.groupby(target[target.columns[0]], axis=1).apply(frames.append)

    # for each feature, do the levene test and put the results into pd.Series
    for rows in zip(*[f.iterrows() for f in frames]):
        _feat_name = rows[0][0]
        _feats = [r[1] for r in rows]

        # If features are normally distributed, use mean as center, otherwise use median
        _center = 'median' if _feat_name in violated_features else 'mean'
        _statistics, _pval = levene(*_feats, center=_center)
        levene_res.append(pd.Series([_statistics, _pval], index=["Statistics", "pval"], name=_feat_name))

    # merge the series into a dataframe
    levene_res = pd.concat(levene_res, axis=1).T
    logger.debug(f"Leven test results: \n{levene_res.to_string()}")
    levene_pvals = levene_res['pval']

    # warn about features that violates the equal variance assumption
    if any(levene_pvals.values <= .05):
        non_equal_var_features = levene_pvals.loc[levene_pvals <= .05].index
        logger.warning(f"Equal variance assumption violated for target '{non_equal_var_features}'")

    # Performing ANOVA
    anova_res = []
    for rows in zip(*[f.iterrows() for f in frames]):
        _feat_name = rows[0][0]
        _feats = [r[1] for r in rows]
        f_statistic, p_value = f_oneway(*_feats)
        anova_res.append(pd.Series([f_statistic, p_value], index=["F", "pval"], name=_feat_name))
    anova_res = pd.concat(anova_res, axis=1).T
    logger.debug(f"ANOVA results: \n{anova_res.to_string()}")
    p_values = anova_res['pval']
    return p_values.to_frame()



def filter_low_var_features(features: pd.DataFrame) -> Tuple[pd.Index, skfs.VarianceThreshold]:
    r"""Filter away features with low variance that are likely to be errors.

    The threshold was determined roughly by aiming to remove features that are constant in x% of
    the cases. The variance of such distribution is estimated to be x * (1 - x)


    Args:
        features (pd.DataFrame):
            Input features.

    Returns:
        pd.Index:
            Index of features with variance higher than threshold.
        skfs.VarianceThreshold:
            The filter used for porting attributes.

    """
    var_filter = skfs.VarianceThreshold(threshold=.95 * (1 - .95))
    var_feats_a = var_filter.fit_transform(features.T)
    var_feats_a_index = features.index[var_filter.get_support()]
    return var_feats_a_index, var_filter

def preliminary_feature_filtering(features_a: pd.DataFrame,
                                  features_b: pd.DataFrame,
                                  targets: pd.DataFrame,
                                  ICC_form: Optional[str] = 'ICC2k',
                                  p_thres: Optional[float] = .001,
                                  ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    r"""Preliminary feature selection by statistical methods.

    Perform feature selction following the steps:
    1. p-values of features < .001
    2. Features with extremely low variance removed (because they are likely to be errors or useless
    2. ICC of features > 0.9
    3. Dimensional reduction


    Args:
        features_a (pd.DataFrame):
            Features extracted using segmentation set A.
        features_b (pd.DataFrame):
            Features extracted using segmentation set B. If None, some ICC parts will be skipped.
        targets (pd.DataFrame):
            Ground-truth class with 'Status' column and patient rows.
            If target patients and features patients indices are not the same, an error will be raised.
        ICC_form (str, optional):
            The form of Intraclass Correlation Coefficient (ICC) to use. Defaults to 'ICC2k'.
        p_thres (float, optional):
            The p-value threshold for feature selection. This p-value is calculated from either T-test or ANOVA,
            depending on the number of classes in the target.

    Returns:
        tuple: Tuple of filtered features_a and features_b DataFrames. The rows are features and the columns are cases.

    Notes:
        - Variance threshold removes features with very low variance, which are likely to be useless (e.g., all values
          are 0 or 1).
        - Features known to be useless, such as 'Diagnostics' column, are dropped.
        - If features_b is provided, ICC filtering is applied.
    """
    logger = MNTSLogger['preliminary_feature_filtering']

    # Drop features known to be useless
    for features in [features_a, features_b]:
        try:
            logger.info("Dropping 'Diganostics' column.")
            features.drop('diagnostics', inplace=True)
        except:
            if features is not None:
                logger.error("Diagnostics column not found or error occurs.")

    # Drop features with extremely low variance (i.e., most are same value)
    logger.info("Dropping features with low variance...")
    var_feats_a_index, var_filter = filter_low_var_features(features_a)
    logger.info(f"Dropped {len(features_a.index) - len(var_feats_a_index)} features for first feature set.")

    # if features_b is provided perform ICC filter
    if features_b is not None:
        # Also drop the low zero vairance features for set b
        var_feats_b_index, _ = filter_low_var_features(features_b)
        # Only those that fulfilled the variance threshold in both set of features are to be included
        mutual_features = list(set(var_feats_a_index) & set(var_feats_b_index))
        logger.info(f"{len(mutual_features)} features kept: \n{mutual_features}")
        logger.info(f"{len(set(features_a.index) - set(mutual_features))} features discarded: \n"
                    f"{set(features_a.index) - set(mutual_features)}")

        # Filter features by ICC
        logger.info("Dropping features by their intra-observer segmentation ICC")
        _icc90_feats = filter_features_by_ICC_thres(features_a.loc[mutual_features],
                                                    features_b.loc[mutual_features], 0.9,
                                                    ICC_form=ICC_form)
        feats_a = features_a.loc[_icc90_feats]
        feats_b = features_b.loc[_icc90_feats]
    else:
        logger.info("Second feature set not found. ICC filtering skipped. ")
        feats_a = features_a.loc[var_feats_a_index]
        feats_b = None


    # Filter out features with not enough t-test significance
    logger.info(f"Dropping features using T-test or ANOVA with p-value: {p_thres}")
    if len(feats_a.columns.difference(targets.index)) > 0:
        msg = f"Differene between features and target detected: {feats_a.columns.difference(targets.index)}"
        raise KeyError(msg)
    # If binary targets, use T-test, otherwise, use ANOVA
    if len(np.unique(targets)) <= 2:
        logger.info("Using T-test")
        pvals_feats_a = filter_features_by_T_test(feats_a, targets)
    else:
        logger.info("Using ANOVA")
        pvals_feats_a = filter_features_by_ANOVA(feats_a, targets)

    logger.info(f"Number of features dropped by using T-test or ANOVA for first feature set: "
                f"{len(feats_a.index) - (pvals_feats_a['pval'] < p_thres).sum()}")
    feats_a = feats_a.loc[(pvals_feats_a['pval'] < p_thres)]
    if not features_b is None:
        # If features_b is provided, features are included only it past T-test requirements for both feat_a and feat_b
        if len(np.unique(targets)) <= 2:
            pvals_feats_b = filter_features_by_T_test(feats_b, targets)
        else:
            pvals_feats_b = filter_features_by_ANOVA(feats_b, targets)
        feats_b = feats_b.loc[(pvals_feats_b['pval'] < p_thres).index]
        feats_a = feats_a.loc[feats_a.index.intersection(features_b.index)]
        feats_b = feats_b.loc[feats_a.index]

    return feats_a, feats_b



@ignore_warnings(category=ConvergenceWarning)
def supervised_features_selection(features: pd.DataFrame,
                                  targets: pd.DataFrame,
                                  alpha: Union[float, Iterable[float]],
                                  l1_ratio: Union[float,Iterable[float]],
                                  *args,
                                  criteria_threshold: Iterable[float] = (0.8, 0.8, 0.99),
                                  n_features: int = 25,
                                  n_splits: int = 5,
                                  n_trials: int = 100,
                                  boosting: bool = True,
                                  **kwargs) -> pd.DataFrame:
    r"""
    Use RENT [1] to select features. Essentially `n_trials` models were trained using the features and targets, and
    the coefficients of the models for these features were recorded as a [`n_trials` by `features.shape[0]`] matrix.
    The quality of each feature were evaluted through these coefficient using three criteria.

    This function inherits some arguements from RENT. See :func:`RENT.RENT_Regression` for more.

    .. notes::
        References:
        * [1] Jenul, Anna, et al. "RENT--Repeated Elastic Net Technique for Feature Selection."
              arXiv preprint arXiv:2009.12780 (2020).

    .. notes::
        This methods assumes input is already scalar normalized.

    Args:
        features (pd.DataFrame):
            Columns should be samples and rows should be features.
        targets (pd.DataFrame):
            Row should be samples and a column 'Status' should hold the class.
        alpha (float):
            Regularization term. Its `alpha` in scipy and `1/C` in sklearn. Default to 0.02.
        l1_ratio (float):
            Ratio between l1 and l2 regularization in elastic net. Put it simply, it's the weight between LASSO and
            Ridge regression. If it's 0, the algorithm collapse to become Ridge.
        n_features:
            (Deprecated)
        n_splits (int):
            Split ratios of train-test group for random bootstrapping.
        n_trials (int):
            Number of runs to select the final model features from. Equivalent to $K$ in the original paper. If this is
            set to 1, a single run of elastic net will be executed instead.

    Returns:
        pd.DataFrame
    """
    logger = MNTSLogger['sup-featselect']
    #|===================|
    #| Run RENT or BRENT |
    #|===================|

    # C in RENT is the inverse of regularization term alpha in scipy
    C = 1/alpha

    # Convert alpha and l1_ratio to list for RENT's convention
    C = np.asarray([C]) if type(C) == float else C
    l1_ratio = np.asarray([l1_ratio]) if type(l1_ratio) == float else l1_ratio

    # Align targets and features, assume target.index $\in$ features.columns
    if not targets.index.to_list() == features.columns.to_list():
        logger.warning("Discrepancy found in case identifiers in target and features, trying to align them!")
        logger.debug(f"Target: {','.join(targets.index.to_list())}")
        logger.debug(f"features: {','.join(features.columns.to_list())}")
        _targets = targets.loc[features.columns]
    else:
        _targets = targets

    # Convert model names into strings
    _features_names = ['__'.join(i) for i in features.index]
    _ori_index = features.index
    _map = {_f: o for _f, o in zip(_features_names, _ori_index)}
    features.index = _features_names

    if n_trials <= 1: # One elastic net run
        # Warn if boost is not 0
        if boosting:
            logger.warning("n_trials = 1 but boosting was turned on.")
        # Normalize shape of the target
        if np.unique(_targets.values).sum() > 2:
            logger.info("Multi-class target encountered.")
            if len(_targets.columns) > 1:
                raise ArithmeticError("Cannot handle multi-class multi-task classification.")
            # Change it into one hot vector
            _onehot = pd.get_dummies(_targets.values.ravel())
            _onehot.index = _targets.index
            _targets = _onehot
        _targets = _targets.values

        # if the l1_ratio is an array
        if isinstance(l1_ratio, (list, tuple, np.ndarray)):
            l1_ratio = l1_ratio[0]
        model = linear_model.ElasticNet(alpha = alpha, l1_ratio=l1_ratio, tol=1E-5)
        model.fit(pd.DataFrame(features.T),
                  _targets)

        # Non-zero coef means the feature is selected
        non_zero_coef = np.argwhere(model.coef_ != 0)
        if model.coef_.ndim == 2:
            # For multi-class, features getting non-zero coefficients in any of the class model are included
            non_zero_coef = np.unique(non_zero_coef[:, 1])

        # If there's a limit to the number of features used
        if len(non_zero_coef) > n_features:
            # Rank the features by the magnitude of the coefficients
            ordered_nzc = np.abs(model.coef_[:, non_zero_coef].mean(axis=0)).argsort()[::-1]
            non_zero_coef = ordered_nzc[:n_features]

        # Gather the indices
        selected_features = features.index[non_zero_coef.ravel()]
        logger.debug(f"ENET features: {selected_features}")
        logger.info(f"Features dropped by ENet {len(features.index) - len(selected_features)}")

        # Construct pd index
        selected_features = pd.MultiIndex.from_tuples([_map[i] for i in selected_features])

    else:
        num_classes = np.unique(_targets.values).sum()
        if num_classes > 1:
            # Raise error because this function doesn't support multi-class classification yet.
            raise ArithmeticError("Input target is not binary. Boosted RENT only supports binary classification. "
                                  "Set `n_trials` to 1 to disable Boosted RENT.")
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
        #| Final selected features |
        #|=========================|
        # Convert the coeffients to weights
        coefs_df = pd.DataFrame(np.concatenate(model._weight_list, axis=0).T, index=_features_names)
        coefs_df = coefs_df.loc[selected_features]

        # normalize the coefficient vector of each model to normal vector (magnitude = 1)
        coefs_df = coefs_df / coefs_df.pow(2).sum().pow(.5)

        # rank the coefs based on their mean and variance, large |mean| and small variance is desired.
        mean_ranks = (coefs_df.shape[0] - coefs_df.T.mean().argsort()) # smaller rank is better
                                                                       # (reverse sorted, mean larger = more important)
        var_ranks = coefs_df.T.std().argsort() # smaller rank is better, variance smaller = more stable
        avg_ranks = (mean_ranks + 0.5 * var_ranks) / 1.5 # weight sum of these two average.

        # remove lower rank features
        n_features = min(len(avg_ranks), n_features)    # no lesser than originally proposed features
        _init_features_index = avg_ranks.sort_values()[:n_features]

        # Construct pd index
        selected_features = pd.MultiIndex.from_tuples([_map[i] for i in _init_features_index.index])

    # Construct the suggested features
    features.index = _ori_index
    out_features = features.loc[selected_features]
    return out_features.sort_index(level=0)

def features_normalization(features):
    r"""
    Normalize the features using `StandardScaler`.
    The input columns should be case and the rows should be features
    """
    normed = preprocessing.StandardScaler().fit_transform(features.T)
    normed = pd.DataFrame(normed.T, columns=features.columns, index=features.index)
    return normed

def features_selection(features_a: pd.DataFrame,
                       targets: pd.DataFrame,
                       features_b: pd.DataFrame = None,
                       n_trials=500,
                       criteria_threshold=(0.9, 0.5, 0.99),
                       boosting: bool = True,
                       ICC_form: Optional[str] = 'ICC2k',
                       ICC_p_thres: Optional[float] = 0.01):
    r"""Performs feature selection using a wrapper function that executes preliminary feature filtering,
    normalization of features, and supervised feature selection (RENT/BRENT).

    Args:
        features_a (pd.DataFrame):
            Features extracted using segmentation set A.
        targets (pd.DataFrame):
            Target variable for the feature selection process.
        features_b (pd.DataFrame, optional):
            Features extracted using segmentation set B. If this is not provided, the ICC step will be skipped.
            Defaults to None.
        n_trials (int, optional):
            Number of trials for the feature selection process. Defaults to 500.
        criteria_threshold (tuple, optional):
            Criteria thresholds for feature selection. Defaults to (0.9, 0.5, 0.99).
        boosting (bool, optional):
            If True, use boosting for feature selection. Defaults to True.
        ICC_form (str, optional):
            The form of Intraclass Correlation Coefficient (ICC) to use. Defaults to 'ICC2k'.

    Returns:
        pd.DataFrame: The selected features after the feature selection process.

    Note:
        - Initial feature filtering is done using quantitative methods and depends on variance and ICC.
          Normalizing data prior to this step would be improper, as it alters the mean and variance of the features.
        - The supervised feature selection uses RENT or BRENT, depending on the `boosting` parameter.
    """
    logger = MNTSLogger['model_building']

    # Initial feature filtering using quantitative methods
    feats_a, feats_b = preliminary_feature_filtering(features_a, features_b, targets, ICC_form='ICC2k', p_thres=ICC_p_thres)

    # Note: Because initial feature filtering relies on variance and ICC, it is not proper to normalize the data prior
    #       to this because it alters the mean and variance of the features.

    # Normalize remaining features for data-driven feature selection
    feats_a = features_normalization(feats_a)
    if features_b is not None:
        feats_b = features_normalization(feats_b)

    # Supervised feature selection using RENT
    # TODO: alphas and l1_ratios should be arguments
    alpha = 0.02
    l1_ratio = 0.5
    feats_a_out = supervised_features_selection(feats_a, targets,
                                                alpha,
                                                l1_ratio,
                                                criteria_threshold=criteria_threshold,
                                                n_trials=n_trials,
                                                n_features=25,
                                                boosting=boosting)
    logger.debug(f"in features: \n{feats_a.sort_index(level=0)}")
    logger.info(f"The selected features: \n{feats_a_out.index}")
    # classification models
    return feats_a_out

def bootstrapped_features_selection(features_a: pd.DataFrame,
                                    targets: pd.DataFrame,
                                    features_b: Optional[pd.DataFrame] = None,
                                    criteria_threshold: Optional[Iterable[float]] =(0.9, 0.5, 0.99),
                                    n_trials: Optional[int] = 500,
                                    boot_runs: Optional[int] = 250,
                                    boot_ratio: Optional[Iterable[float]] = (0.8, 1.0),
                                    thres_percentage: Optional[float] = 0.4,
                                    return_freq: Optional[bool] = False,
                                    boosting: Optional[bool] = True,
                                    ICC_form: Optional[str] = 'ICC2k') -> List[pd.DataFrame]:
    r"""Improve feature selection stability using bootstrapping.

    This function uses bootstrapping to create multiple subsets of the input data and applies
    feature selection on each subset. The final selected features are those that appear in the
    results with a frequency greater than a specified threshold.

    Args:
        features_a (pd.DataFrame):
            The first set of features (input variables). Rows should be features and column should be data points.
        targets (pd.DataFrame):
            The target (output) variable.
        features_b (Optional[pd.DataFrame], default=None):
            The second set of features (input variables).
        criteria_threshold (Optional[Iterable[float]], default=(0.9, 0.5, 0.99))
            The criteria thresholds for feature selection.
        n_trials (Optional[int], default=500)
            The number of trials for feature selection.
        boot_runs : (Optional[int], default=250)
            The number of bootstrapping runs.
        boot_ratio : (Optional[Iterable[float]], default=(0.8, 1.0))
            The range for generating random bootstrap ratios.
        thres_percentage (Optional[float], default=0.4)
            The threshold percentage for selecting features based on their appearance frequency.
        return_freq (Optional[bool], default=False)
            If True, return the feature frequencies along with the selected features.
        boosting (Optional[bool], default=True)
            If True, use boosting for feature selection.
        ICC_form (Optional[str], default='ICC2k')
            The form of Intraclass Correlation Coefficient (ICC) to use.

    Returns:
        List[pd.DataFrame]
            A list of selected features from `features_a` and `features_b`.

    .. notes::
        See also the arguments detailed in `func:features_selection`.


    """
    logger = MNTSLogger[__name__]
    bootstrap_l, bootstrap_u = boot_ratio
    bootstrap_ratio = np.random.rand(boot_runs) * (bootstrap_u - bootstrap_l) + bootstrap_l
    features_list = []
    features_dict = {}

    features_name_map ={'__'.join(i): i for i in  features_a.index}
    # Bootstrapping loop
    for i in range(boot_runs):
        logger.info(f"=== Running {i} ===")
        train_y = sklearn.utils.resample(targets, n_samples = int(len(targets) * bootstrap_ratio[i]),
                                         stratify = targets.values.ravel())
        train_xa = features_a.T.loc[train_y.index].T
        train_xb = features_b.T.loc[train_y.index].T if not features_b is None else None

        features = features_selection(train_xa, train_y, train_xb, n_trials=n_trials,
                                      criteria_threshold=criteria_threshold, boosting=boosting, ICC_form=ICC_form)
        features_names = ['__'.join(i) for i in features.index]
        features_list.append(features_names)
        features_dict[i] = features_names
        logger.info(f"=== Done {i} ===")

    # Return features with more than certain percentage of appearance
    # Set thres to 0 if bo bootstrapping (bagging), i.e. all feature selected will be included
    if boot_runs == 1:
        thres_percentage = 0

    # Count features frequencies
    all_features = set.union(*[set(i) for i in features_list])
    features_frequencies = [
        pd.Series({features_name_map[a]: features_dict[i].count(a) for a in features_dict[i]}, name=f'run_{i:02d}')
        for i in range(boot_runs)]
    features_frequencies = pd.concat(features_frequencies, axis=1).fillna(0)
    features_frequencies['Sum'] = features_frequencies.sum(axis=1)
    features_frequencies['Rate'] = features_frequencies['Sum'] / float(boot_runs)

    selected_features = features_frequencies[features_frequencies['Rate'] > thres_percentage].index

    if return_freq:
        return features_frequencies, selected_features
    else:
        return features_a.loc[selected_features], features_b.loc[selected_features]

class FeatureSelector(object):
    r"""
    This class is used for feature selection.

    Args:
        criteria_threshold ([int, int, int], Optional):
            Corresponding to the three threshold described in Anna et al. [1], default to (0.9, 0.5, 0.99)
        n_trials (int, Optional):
            Specify the number times the elastic net is ran in RENT/BRENT.
        boot_runs (int, Optional):
            Number of bootstrapped runs. Each bootstrapped run execute RENT/BRENT once. Features that are nominated in
            more than `thres_percentage` of the runs are included in the final feature subset. Default to 250.
        boot_ratio ([float, float], Optional):
            The lower and upper bound for the ratio of train-test sample size in each bootstrapped runs. Default to
            (0.8, 1.0).
        thres_percentage (float, Optional):
            See `boot_runs`.
        return_freq (bool, Optional):
            If True, `fit` will return the frequency of selection of each feature across the bootsrapped runs.
        boosting (bool, Optional):
            If True, BRENT is used, otherwise RENT is used. Default to True.
        ICC_form (str, Optional):
            One of the six forms in ["ICC1"|"ICC2"|"ICC3"|"ICC1k"|"ICC2k"|"ICC3k"]. Default to be "ICC2k". See also
            :func:`computeICC`

    """
    MC_STRATS = ('ovr')
    def __init__(self,
                 *, # not used
                 criteria_threshold: Optional[Sequence[int]] = (0.9, 0.5, 0.99),
                 n_trials:           Optional[int] = 500,
                 boot_runs:          Optional[int] = 250,
                 boot_ratio:         Optional[Iterable[float]] = (0.8, 1.0),
                 thres_percentage:   Optional[float] = 0.4,
                 return_freq:        Optional[bool] = False,
                 boosting:           Optional[bool] = True,
                 ICC_form:           Optional[str]  = 'ICC2k',
                 multi_class:        Optional[str]  = None,
                 **kwargs # not used
                 ):
        super(FeatureSelector, self).__init__()
        self._logger = MNTSLogger[__class__.__name__]

        # These are controlled by yaml loaded by the controller.
        setting = {
            'criteria_threshold': criteria_threshold,
            'n_trials': n_trials,
            'boot_runs': boot_runs,
            'boot_ratio': boot_ratio,
            'thres_percentage': thres_percentage,
            'return_freq': return_freq,
            'boosting': boosting,
            'ICC_form': ICC_form,
            'multi_class': multi_class
        }
        self.saved_state = {
            'selected_features': None,
            'feat_freq': None,
            'setting': setting,
        }

    @property
    def selected_features(self):
        return self.saved_state['selected_features']

    @selected_features.setter
    def selected_features(self, v):
        raise ArithmeticError("Selected features should not be manually assigned.")

    @property
    def setting(self):
        return self.saved_state['setting']

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
        joblib.dump(self.saved_state, filename=f.with_suffix('.fss'))

    def fit_(self,
            X_a: pd.DataFrame,
            y:   Union[pd.DataFrame, pd.Series],
            X_b: Optional[pd.DataFrame]=None) -> Union[Tuple, pd.DataFrame]:
        r"""Fit to select features from input. This method is for binary classification models only.

        Args:
            X_a (pd.DataFrame):
                Radiomics features. Each row should be a datapoint, each column should be a feature.
            y (pd.DataFrame or pd.Series):
                Class of the data. Each row should be a datapoint for DataFrame. It should only have one value column
                corresponding to the status of each patient.
            X_b (pd.DataFrame, Optional):
                Radiomics features from another segmentation. Aims to filter away features that are susceptable to
                inter-observer changes in the segmentation. Default to None.

        Returns:
            feats (pd.DataFrame):
                feats is the tuple fo (features frequency, features tuple)
        """
        assert y.nunique() == 2, "This method is for binary classification only."

        feats = bootstrapped_features_selection(X_a.T,
                                                y,
                                                X_b.T if X_b is not None else None,
                                                criteria_threshold=self.setting['criteria_threshold'],
                                                n_trials=self.setting['n_trials'],
                                                boot_runs=self.setting['boot_runs'],
                                                boot_ratio=self.setting['boot_ratio'],
                                                thres_percentage=self.setting['thres_percentage'],
                                                return_freq=True,
                                                boosting=self.setting['boosting'],
                                                ICC_form=self.setting['ICC_form'])
        return feats


    def fit(self,
            X_a: pd.DataFrame,
            y:   Union[pd.DataFrame, pd.Series],
            X_b: Optional[pd.DataFrame]=None,
            ) -> Union[Tuple, pd.DataFrame]:
        r"""Wrapper for deciding if the classification is for

        Args:
            X_a (pd.DataFrame):
                Radiomics features. Each row should be a datapoint, each column should be a feature.
            y (pd.DataFrame or pd.Series):
                Class of the data. Each row should be a datapoint for DataFrame. It should only have one value column
                corresponding to the status of each patient.
            X_b (pd.DataFrame, Optional):
                Radiomics features from another segmentation. Aims to filter away features that are susceptable to
                inter-observer changes in the segmentation. Default to None.

        Returns:
            feats (pd.DataFrame):
                feats is the tuple fo (features frequency, features tuple)
        """
        if not self.setting['multi_class']:
            feats = self.fit_(X_a, y, X_b)
            if len(feats[1]) == 0:
                msg = "No features returned. The `thres_percentage` setting might be too high, try tunning it down."
                raise ArithmeticError(msg)

            self._logger.info(f"Selected {len(feats[1])} features: {feats[1]}")
            self.saved_state['selected_features'] = feats[1]
            self.saved_state['feat_freq'] = feats[0]
            if self.setting['return_freq']:
                return feats
            else:
                if X_b is not None:
                    return X_a[feats[1]], X_b[feats[1]]
                else:
                    return X_a[feats[1]], None
        else:
            # check if input is really multi-class
            if y.nunique() <= 2:
                raise ArithmeticError("Multi-class option is specified, but input is binary")
            # check if multi-class strategies is correct
            if not self.setting['multi_class'] in self.__class__.MC_STRATS:
                raise ValueError(f"`multi_class` must be one of the followings: [{','.join(self.__class__.MC_STRATS)}],"
                                 f"got {self.setting['multi_class']} instead.")
            if self.setting['multi_class'] == 'ovr':
                # convert y into OVR binary if its not already it
                _y = pd.get_dummies(_y, columns = y.unique()) if y.ndim < 2 else y

                # each column is a class
                num_class = _y.shape[1]

                feats_list = []
                for i in range(num_class):
                    y_binary = _y[_y.columns[i]]
                    feats_list.append(self.fit_(X_a, y_binary, X_b))
                return out
            else:
                raise NotImplementedError("Required multi-class strategy is not implemented.")

    def predict(self, X_a: pd.DataFrame) -> pd.DataFrame:
        r"""
        Retrurn the feature columns that are selected

        Args:
            X_a:

        Returns:

        """
        if self.saved_state['selected_features'] is None:
            raise ArithmeticError("No information about selected features. Have you run fit()?")

        return X_a[self.saved_state['selected_features']]