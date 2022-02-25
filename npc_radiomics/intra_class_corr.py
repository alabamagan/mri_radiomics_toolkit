"""
This code is an overhaul of the code in [1].

References:
    [1] https://github.com/raphaelvallat/pingouin
"""

import numpy as np
import pandas as pd
from scipy.stats import f
from pingouin.config import options
from pingouin.utils import _postprocess_dataframe


__all__ = ["cronbach_alpha", "intraclass_corr"]

def intraclass_corr(data=None, targets=None, raters=None, ratings=None,
                    nan_policy='raise'):
    """Intraclass correlation.
    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataframe. Data must be fully balanced.
    targets : string
        Name of column in ``data`` containing the targets.
    raters : string
        Name of column in ``data`` containing the raters.
    ratings : string
        Name of column in ``data`` containing the ratings.
    nan_policy : str
        Defines how to handle when input contains missing values (nan).
        `'raise'` (default) throws an error, `'omit'` performs the calculations
        after deleting target(s) with one or more missing values (= listwise
        deletion).
        .. versionadded:: 0.3.0
    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Output dataframe:
        * ``'Type'``: ICC type
        * ``'Description'``: description of the ICC
        * ``'ICC'``: intraclass correlation
        * ``'F'``: F statistic
        * ``'df1'``: numerator degree of freedom
        * ``'df2'``: denominator degree of freedom
        * ``'pval'``: p-value
        * ``'CI95%'``: 95% confidence intervals around the ICC
    Notes
    -----
    The intraclass correlation (ICC, [1]_) assesses the reliability of ratings
    by comparing the variability of different ratings of the same subject to
    the total variation across all ratings and all subjects.
    Shrout and Fleiss (1979) [2]_ describe six cases of reliability of ratings
    done by :math:`k` raters on :math:`n` targets. Pingouin returns all six
    cases with corresponding F and p-values, as well as 95% confidence
    intervals.
    From the documentation of the ICC function in the `psych
    <https://cran.r-project.org/web/packages/psych/psych.pdf>`_ R package:
    - **ICC1**: Each target is rated by a different rater and the raters are
      selected at random. This is a one-way ANOVA fixed effects model.
    - **ICC2**: A random sample of :math:`k` raters rate each target. The
      measure is one of absolute agreement in the ratings. ICC1 is sensitive
      to differences in means between raters and is a measure of absolute
      agreement.
    - **ICC3**: A fixed set of :math:`k` raters rate each target. There is no
      generalization to a larger population of raters. ICC2 and ICC3 remove
      mean differences between raters, but are sensitive to interactions.
      The difference between ICC2 and ICC3 is whether raters are seen as fixed
      or random effects.
    Then, for each of these cases, the reliability can either be estimated for
    a single rating or for the average of :math:`k` ratings. The 1 rating case
    is equivalent to the average intercorrelation, while the :math:`k` rating
    case is equivalent to the Spearman Brown adjusted reliability.
    **ICC1k**, **ICC2k**, **ICC3K** reflect the means of :math:`k` raters.
    This function has been tested against the ICC function of the R psych
    package. Note however that contrarily to the R implementation, the
    current implementation does not use linear mixed effect but regular ANOVA,
    which means that it only works with complete-case data (no missing values).
    References
    ----------
    .. [1] http://www.real-statistics.com/reliability/intraclass-correlation/
    .. [2] Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations:
           uses in assessing rater reliability. Psychological bulletin, 86(2),
           420.
    Examples
    --------
    ICCs of wine quality assessed by 4 judges.
    >>> import pingouin as pg
    >>> data = pg.read_dataset('icc')
    >>> icc = pg.intraclass_corr(data=data, targets='Wine', raters='Judge',
    ...                          ratings='Scores').round(3)
    >>> icc.set_index("Type")
                       Description    ICC       F  df1  df2  pval         CI95%
    Type
    ICC1    Single raters absolute  0.728  11.680    7   24   0.0  [0.43, 0.93]
    ICC2      Single random raters  0.728  11.787    7   21   0.0  [0.43, 0.93]
    ICC3       Single fixed raters  0.729  11.787    7   21   0.0  [0.43, 0.93]
    ICC1k  Average raters absolute  0.914  11.680    7   24   0.0  [0.75, 0.98]
    ICC2k    Average random raters  0.914  11.787    7   21   0.0  [0.75, 0.98]
    ICC3k     Average fixed raters  0.915  11.787    7   21   0.0  [0.75, 0.98]
    """
    from pingouin import anova

    # Safety check
    assert isinstance(data, pd.DataFrame), 'data must be a dataframe.'
    assert all([v is not None for v in [targets, raters, ratings]])
    assert all([v in data.columns for v in [targets, raters, ratings]])
    assert nan_policy in ['omit', 'raise']

    # Convert data to wide-format
    data = data.pivot_table(index=targets, columns=raters, values=ratings, observed=True)

    # Listwise deletion of missing values
    nan_present = data.isna().any().any()
    if nan_present:
        if nan_policy == 'omit':
            data = data.dropna(axis=0, how='any')
        else:
            raise ValueError("Either missing values are present in data or "
                             "data are unbalanced. Please remove them "
                             "manually or use nan_policy='omit'.")

    # Back to long-format
    # data_wide = data.copy()  # Optional, for PCA
    data = data.reset_index().melt(id_vars=targets, value_name=ratings)

    # Check that ratings is a numeric variable
    assert data[ratings].dtype.kind in 'bfiu', 'Ratings must be numeric.'
    # Check that data are fully balanced
    # This behavior is ensured by the long-to-wide-to-long transformation
    # Unbalanced data will result in rows with missing values.
    # assert data.groupby(raters)[ratings].count().nunique() == 1

    # Extract sizes
    k = data[raters].nunique()
    n = data[targets].nunique()

    # Two-way ANOVA
    with np.errstate(invalid='ignore'):
        # For max precision, make sure rounding is disabled
        old_options = options.copy()
        options['round'] = None
        aov = anova(data=data, dv=ratings, between=[targets, raters],
                    ss_type=2)
        options.update(old_options)  # restore options

    # Extract mean squares
    msb = aov.at[0, 'MS']
    msw = (aov.at[1, 'SS'] + aov.at[2, 'SS']) / (aov.at[1, 'DF'] +
                                                 aov.at[2, 'DF'])
    msj = aov.at[1, 'MS']
    mse = aov.at[2, 'MS']

    # Calculate ICCs
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icc2 = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
    icc3 = (msb - mse) / (msb + (k - 1) * mse)
    icc1k = (msb - msw) / msb
    icc2k = (msb - mse) / (msb + (msj - mse) / n)
    icc3k = (msb - mse) / msb

    # Calculate F, df, and p-values
    f1k = msb / msw
    df1 = n - 1
    df1kd = n * (k - 1)
    p1k = f.sf(f1k, df1, df1kd)

    f2k = f3k = msb / mse
    df2kd = (n - 1) * (k - 1)
    p2k = f.sf(f2k, df1, df2kd)

    # Create output dataframe
    stats = {
        'Type': ['ICC1', 'ICC2', 'ICC3', 'ICC1k', 'ICC2k', 'ICC3k'],
        'Description': ['Single raters absolute', 'Single random raters',
                        'Single fixed raters', 'Average raters absolute',
                        'Average random raters', 'Average fixed raters'],
        'ICC': [icc1, icc2, icc3, icc1k, icc2k, icc3k],
        'F': [f1k, f2k, f2k, f1k, f2k, f2k],
        'df1': n - 1,
        'df2': [df1kd, df2kd, df2kd, df1kd, df2kd, df2kd],
        'pval': [p1k, p2k, p2k, p1k, p2k, p2k]
    }

    stats = pd.DataFrame(stats)

    # Calculate confidence intervals
    alpha = 0.05
    # Case 1 and 3
    f1l = f1k / f.ppf(1 - alpha / 2, df1, df1kd)
    f1u = f1k * f.ppf(1 - alpha / 2, df1kd, df1)
    l1 = (f1l - 1) / (f1l + (k - 1))
    u1 = (f1u - 1) / (f1u + (k - 1))
    f3l = f3k / f.ppf(1 - alpha / 2, df1, df2kd)
    f3u = f3k * f.ppf(1 - alpha / 2, df2kd, df1)
    l3 = (f3l - 1) / (f3l + (k - 1))
    u3 = (f3u - 1) / (f3u + (k - 1))
    # Case 2
    fj = msj / mse
    vn = df2kd * ((k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2))**2
    vd = df1 * k**2 * icc2**2 * fj**2 + \
        (n * (1 + (k - 1) * icc2) - k * icc2)**2
    v = vn / vd
    f2u = f.ppf(1 - alpha / 2, n - 1, v)
    f2l = f.ppf(1 - alpha / 2, v, n - 1)
    l2 = n * (msb - f2u * mse) / (f2u * (k * msj + (k * n - k - n) * mse) +
                                  n * msb)
    u2 = n * (f2l * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f2l *
                                  msb)

    stats['CI95%'] = [
        np.array([l1, u1]),
        np.array([l2, u2]),
        np.array([l3, u3]),
        np.array([1 - 1 / f1l, 1 - 1 / f1u]),
        np.array([l2 * k / (1 + l2 * (k - 1)), u2 * k / (1 + u2 * (k - 1))]),
        np.array([1 - 1 / f3l, 1 - 1 / f3u])
    ]

    return _postprocess_dataframe(stats)