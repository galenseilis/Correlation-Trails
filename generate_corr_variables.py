"""
Generates correlated variables.
"""

import numpy as np
from scipy.linalg import eigh, cholesky

def match_cov(X, C, method='cholesky'):
    '''
    Given a dataset, and a covariance matrix,
    this function transforms the dataset to have
    the same covariance structure as the given
    covariance matrix.
    
    PARAMETERS
        X (array-like): Dataset with columns as variables.
        C (array-like): Desired covariance matrix.
        method (str): Decomposition method.
        
    RETURNS
        : array-like
        Transformed data with given covariance structure.
    '''
    if method=='cholesky':
        T = cholesky(C, lower=True)
    else:
        evals, evecs = eigh(C)
        T = np.dot(evecs, np.diag(np.sqrt(evals)))
    return np.dot(T, X)

def cov_to_r(C):
    '''
    Converts a covariance matrix into its corresponding
    correlation matrix.

    PARAMETERS
        C (array-like): Covariance matrix.

    RETURNS
        R (array-like): Correlation matrix.
    '''
    R = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            R[i, j] = C[i, j] / np.sqrt(C[i,i]*C[j,j])
    return R
