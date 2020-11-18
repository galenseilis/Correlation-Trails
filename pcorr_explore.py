import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Definte utility functions
def rand_hyp(x):
    '''
    Generate random covariate structure.
    
    PARAMETERS
        x (array-like): List of variables.
    RETURNS
        d (dict): Random covariance hypothesis.
    '''
    d = {}
    for i, xi in enumerate(x):
        covs = []
        for j, xj in enumerate(x):
            if i > j and np.random.randint(0, 2):
                covs.append(xj)
        d[xi] = covs
    return d

def spcorr(df, cols, cov_dict=None, alpha=0.05):
    '''Calculates a matrix of semi-partial correlation coefficients
        under a given covariance hypothesis.

        PARAMETERS
            df (dataframe): Input data.
            cols (list(str)): Chosen columns for calculation.
            cov_dict (dict): Covariance hypothesis (None).
        RETURNS
            (array-like): Correlation matrix.
            
    '''
    if cov_dict == None:
        cov_dict = {i:[j for j in cols if i != j] for i in cols}
    R = np.zeros((len(cols), len(cols)))
    for i, xi in enumerate(cols):
        for j, xj in enumerate(cols):
            if i > j:
                xi_cov = [i for i in cov_dict[xi] if i != xj]
                xj_cov = [i for i in cov_dict[xj] if i != xi]
                r = pg.partial_corr(df, xi, xj, x_covar=xi_cov, y_covar=xj_cov)
                if r['p-val'][0] < alpha:
                    R[i, j] = r['r'][0]
    return R + R.T + np.identity(R.shape[0])

def cov_hyp_from_corr(X, cols):
    d = {}
    X = X - np.identity(X.shape[0])
    X = np.abs(X) > 0
    for i, row in enumerate(X):
        d[cols[i]] = list(np.array(cols)[row])
    return d

def cov_trail(df, cols, cov_dict):
    '''
    Calculates deterministic directed
    relations between correlation matrices
    from a given starting hypothesis.
    '''
    history_cov = [cov_dict]
    while 1:
        C = spcorr(df, cols, cov_dict)
        cov_dict = cov_hyp_from_corr(C, cols)
        if cov_dict in history_cov:
            history_cov.append(cov_dict)
            break
        else:
            history_cov.append(cov_dict)
    return history_cov

def history_adj_mat(history_cov):
    '''Returns adjacency matrix of a sequence.'''
    Q = np.zeros((len(history_cov), len(history_cov)))
    for i, ci in enumerate(history_cov):
            for j, cj in enumerate(history_cov):
                    if ci == cj:
                            Q[i,j] = 1
    return Q

def plot_mat(R, cols, filename='matrix.png'):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.5, 10.5)
    im = ax.imshow(R, cmap='coolwarm')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
             rotation_mode="anchor")

    for i in range(len(cols)):
        for j in range(len(cols)):
            if abs(R[i, j]) > 0:
                text = ax.text(j, i, R[i, j],
                               ha="center", va="center", color="w")
    plt.savefig(filename)
