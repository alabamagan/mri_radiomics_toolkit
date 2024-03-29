'''
This file was cloned from https://github.com/nogueirs/JMLR2018/blob/master/python/stability/__init__.py
The method was propsoed in Nogueira et al. 2017.

Docstring and signatures format were revised to google style.

Reference:
----------
Nogueira, S., Sechidis, K., & Brown, G. (2017). On the stability of feature selection algorithms.
    J. Mach. Learn. Res., 18(1), 6345-6398.

'''

from typing import Any, Iterable, Optional

import math
import numpy as np
from scipy.stats import norm
from sklearn import metrics
import warnings

def getStability(Z: np.ndarray) -> float:
    r''' 
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].
    
    Args: 
        Z (np.ndarray): 
            A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
            Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
            means the f^th feature has been selected and a 0 means it has not been selected.
           
    Returns:
        (float) - The stability of the feature selection procedure
    ''' # noqa
    Z=checkInputType(Z)
    M,d=Z.shape
    hatPF=np.mean(Z,axis=0)
    kbar=np.sum(hatPF)
    denom=(kbar/d)*(1-kbar/d)
    return 1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom

def getVarianceofStability(Z: np.ndarray) -> float:
    '''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate and its variance as given in [1].
    
    Args: 
        Z: 
            A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception otherwise).
            Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
            means the f^th feature has been selected and a 0 means it has not been selected.
           
    Returns: 
        (dict) - A dictionnary where the key 'stability' provides the corresponding stability value #
                 and where the key 'variance' provides the variance of the stability estimate
    ''' # noqa
    Z=checkInputType(Z) # check the input Z is of the right type
    M,d=Z.shape # M is the number of feature sets and d the total number of features
    hatPF=np.mean(Z,axis=0) # hatPF is a numpy.array with the frequency of selection of each feature
    kbar=np.sum(hatPF) # kbar is the average number of selected features over the M feature sets
    k=np.sum(Z,axis=1) # k is a numpy.array with the number of features selected on each one of the M feature sets
    denom=(kbar/d)*(1-kbar/d) 
    stab=1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom # the stability estimate
    phi=np.zeros(M)
    for i in range(M):
        phi[i]=(1/denom)*(np.mean(np.multiply(Z[i,],hatPF))-(k[i]*kbar)/d**2+(stab/2)*((2*k[i]*kbar)/d**2-k[i]/d-kbar/d+1))
    phiAv=np.mean(phi)
    variance=(4/M**2)*np.sum(np.power(phi-phiAv,2)) # the variance of the stability estimate as given in [1]
    return {'stability':stab,'variance':variance}

def confidenceIntervals(Z: np.ndarray,
                        alpha: Optional[float] = 0.05,
                        res: Optional[dict] = {}) -> dict:
    r'''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function provides the stability estimate and the lower and upper bounds of the (1-alpha)- approximate confidence 
    interval as given by Corollary 9 in [1]
    
    Args:
        Z (np.ndarray):
            A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception otherwise).
            Each row of the binary matrix represents a feature set, where a 1 at the f^th position
            means the f^th feature has been selected and a 0 means it has not been selected.
        alpha (float, Optional):
            `alpha` is an optional argument corresponding to the level of significance for the confidence interval
            (default is 0.05), e.g. alpha=0.05 give the lower and upper bound of for a (1-alpha)=95% confidence interval.
        res (dict, Optional):
            In case you already computed the stability estimate of Z using the function getVarianceofStability(Z),
            you can provide theresult (a dictionnary) as an optional argument to this function for faster computation.
           
    Returns:
        (dict) - A dictionnary where the key 'stability' provides the corresponding stability value, where the key
                'variance' provides the variance of the stability estimate the keys 'lower' and 'upper' respectively
                give the lower and upper bounds of the (1-alpha)-confidence interval.
    ''' # noqa
    Z=checkInputType(Z) # check the input Z is of the right type
    ## we check if values of alpha between ) and 1
    if alpha>=1 or alpha<=0:
        raise ValueError('The level of significance alpha should be a value >0 and <1')
    if len(res)==0: 
        res=getVarianceofStability(Z) # get a dictionnary with the stability estimate and its variance
    lower=res['stability']-norm.ppf(1-alpha/2)*math.sqrt(res['variance']) # lower bound of the confidence interval at a level alpha
    upper=res['stability']+norm.ppf(1-alpha/2)*math.sqrt(res['variance']) # upper bound of the confidence interval 
    return {'stability':res['stability'],'lower':lower,'upper':upper}

## this tests whether the true stability is equal to a given value stab0
def hypothesisTestV(Z: np.ndarray,
                    stab0: float,
                    alpha: Optional[float] = 0.05) -> dict:
    r'''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function implements the null hypothesis test in [1] that test whether the population stability is greater 
    than a given value stab0.
    
    Args:
        Z (np.ndarray):
            A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception
            otherwise). Each row of the binary matrix represents a feature set, where a 1 at the f^th position
            means the f^th feature has been selected and a 0 means it has not been selected.
        stab0 (float):
            `stab0` is the value we want to compare the stability of the feature selection to.
        alpha (float):
            `alpha` is an optional argument corresponding to the level of significance of the null hypothesis test
            (default is 0.05).
           
    Returns:
        (dict)
            A dictionnary with:
            * a boolean value for key 'reject' equal to True if the null hypothesis is rejected and to False otherwise
            * a float for the key 'V' giving the value of the test statistic
            * a float giving for the key 'p-value' giving the p-value of the hypothesis test
    ''' # noqa
    Z=checkInputType(Z) # check the input Z is of the right type
    res=getVarianceofStability(Z)
    V=(res['stability']-stab0)/math.sqrt(res['variance'])
    zCrit=norm.ppf(1-alpha)
    if V>=zCrit: reject=True
    else: reject=False
    pValue=1-norm.cdf(V)
    return {'reject':reject,'V':V,'p-value':pValue}

# this tests the equality of the stability of two algorithms
def hypothesisTestT(Z1: np.ndarray,
                    Z2: np.ndarray,
                    alpha: Optional[float] = 0.05) -> dict:
    '''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function implements the null hypothesis test of Theorem 10 in [1] that test whether 
    two population stabilities are identical.

    Args:
        Z1 & Z2 (np.ndarray)
            Two BINARY matrices Z1 and Z2 (given as lists or as numpy.ndarray objects of size M*d).
            Each row of the binary matrix represents a feature set, where a 1 at the f^th position
            means the f^th feature has been selected and a 0 means it has not been selected.
        alpha (float, Optional)
            alpha is an optional argument corresponding to the level of significance of the null
            hypothesis test (default is 0.05)
           
    Returns:
         (dict)
            A dictionnary with:
            * a boolean value for key 'reject' equal to True if the null hypothesis is rejected and to False otherwise
            * a float for the key 'T' giving the value of the test statistic
            * a float giving for the key 'p-value' giving the p-value of the hypothesis test
    ''' # noqa
    Z1=checkInputType(Z1) # check the input Z1 is of the right type
    Z2=checkInputType(Z2) # check the input Z2 is of the right type
    res1=getVarianceofStability(Z1)
    res2=getVarianceofStability(Z2)
    stab1=res1['stability']
    stab2=res2['stability']
    var1=res1['variance']
    var2=res2['variance']
    T=(stab2-stab1)/math.sqrt(var1+var2)
    zCrit=norm.ppf(1-alpha/2) 
    ## the cumulative inverse of the gaussian at 1-alpha/2
    if(abs(T)>=zCrit):
        reject=True
        #print('Reject H0: the two algorithms have different population stabilities')
    else:
        reject=False
        #print('Do not reject H0')
    pValue=2*(1-norm.cdf(abs(T)))
    return {'reject':reject,'T':T,'p-value':pValue}

def checkInputType(Z):
    ''' This function checks that Z is of the rigt type and dimension.
        It raises an exception if not.
        OUTPUT: The input Z as a numpy.ndarray
    '''
    ### We check that Z is a list or a numpy.array
    if isinstance(Z,list):
        Z=np.asarray(Z)
    elif not isinstance(Z,np.ndarray):
        raise ValueError('The input matrix Z should be of type list or numpy.ndarray')
    ### We check if Z is a matrix (2 dimensions)
    if Z.ndim!=2:
        raise ValueError('The input matrix Z should be of dimension 2')
    return Z


#=== My Code ===
import pandas as pd
import itertools
import multiprocessing as mpi
from sklearn.metrics import jaccard_score

def feat_list_to_binary_mat(selected_feat_list: pd.DataFrame,
                            full_feat_list: Iterable[Any]) -> np.ndarray:
    r"""

    Args:
        selected_feat_list (pd.DataFrame):
            Collection of series, each series is the features selected in one trial. Series can have different length,
            empty string or NA will be ignored. The element should all be a strings.
        full_feat_list (Iterable[Any]):
            The full list of all features.
    Returns:

    """ # noqa
    full_feat_list = list(full_feat_list)

    # Check if all features in the feature_list is in the full_feat_list
    selected_feat_list = selected_feat_list.fillna('')
    sf_union = set.union(*[set(selected_feat_list[s]) for s in selected_feat_list])
    sf_union -= set(['', 'nan']) # remove na and empty string

    fl = set(full_feat_list)
    if len(sf_union - fl) > 0:
        missing = sf_union - set(full_feat_list)
        msg = f"The following features were not in the full_feat_list: {','.join(missing)}"
        raise IndexError(msg)

    d = len(full_feat_list)
    m = len(selected_feat_list.columns)
    Z = np.zeros([m, d], dtype=bool)
    for i, col in enumerate(selected_feat_list):
        # Get the index of the features in the selected list
        index = [full_feat_list.index(s) for s in selected_feat_list[col][selected_feat_list[col] != '']]
        for j in index:
            Z[i, j] = True
    return Z

def jaccard_mean(Z: np.ndarray):
    r"""
    
    Args:
        Z (np.ndarray):
            Each row of the binary matrix represents a feature set, where a 1 at the f^th position
            means the f^th feature has been selected and a 0 means it has not been selected. 

    Returns:
        (float) - Mean of the pair-wise jaccard index
    """ # noqa

    jac_job = [(Z[i], Z[j]) for i, j in itertools.product(range(len(Z)), range(len(Z))) if i != j]

    pool = mpi.Pool()
    res = pool.starmap_async(jaccard_score, jac_job)
    pool.close()
    pool.join()
    jac = res.get()
    jac_bar = np.mean(jac)
    jac_bar_sd = np.std(jac)
    return jac, jac_bar, jac_bar_sd


def top_k_accuracy_score_(y_true, y_score, **kwargs):
    r"""This implementation escapes for scenarios where y_score prediction is a vector that does not 
    consist of all the labels in y_true.
    
    Args:
        y_true (np.ndarray):
            If input dimension doesn't match unique classes in `y_score`
        y_score (np.ndarray):
            Score for prediction. If multi-class, dimension should be (n_class, n_sample)
        **kwargs:
            Passed to :func:`sklearn.metrics.top_k_accuracy_score`
    
    .. notes::
        See :func:`sklearn.metrics.top_k_accuracy_score` for more parameter explainations.
    """ # noqa
    # obtain the unique label
    unique_labels = kwargs.pop('labels', np.unique(y_true))
    k = kwargs.pop('k', 2)

    # convert input to array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)

    # If score is integer, convert it into one-hot probabilities of 100%
    if np.issubdtype(y_score.dtype, np.integer):
        # Warn if k > 1 because accuracy for second guess is also 0
        if k > 1:
            warnings.warn("Setting k > 1 but provided integer y_score. Results could be invalid")
        y_score = pd.get_dummies(y_score)
    else:
        y_score = pd.DataFrame(y_score)
    score_class = y_score.columns

    # check if y_score and y_true has the same unique values
    y_true_set = set(unique_labels)
    if not y_score.shape[1] == len(y_true_set):
        try:
            # If there are labels to reference the missing class
            missing_class = y_true_set - set(score_class)
            for m in missing_class:
                y_score[m] = 0
        except TypeError:
            # otherwise, assume the input is ordered and assign them ordered labels
            for i in range(len(y_true_set) - len(score_class)):
                y_score[i + y_score.shape[1]] = 0
    # order the matrix to be same as the labels
    y_score = y_score[unique_labels]

    # convert score to float for calculation
    y_score = y_score.astype(float)
    return metrics.top_k_accuracy_score(y_true, y_score, labels=unique_labels, k=k, **kwargs)
