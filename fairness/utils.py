import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

### General utility functions ###

# p : X -> [0,1]
# p is any monotonic function
def p(x) -> float:
    return 1 / (1 + np.exp(-x))

def expected(x, plus, minus):
    return p(x) * plus + (1-p(x)) * minus

### Utility functions for threshold selected from samples ###
def opt_step(X, u_plus, u_minus, c_plus, c_minus):
    X = np.asarray(X)

    # Calculate utility and predicted change of all samples
    exp_util = expected(X, u_plus, u_minus) 
    delta_x = expected(X, c_plus, c_minus)

    # Add predicted change to sample if utility is positive else return sample
    max_util = np.where(exp_util > 0, X + delta_x, X) 

    # Smallest threshold where utility is positive
    thresholds = X[exp_util > 0]
    opt_thresh = np.min(thresholds) if thresholds.size > 0 else None

    # array of updated samples, min sample whose util is positive
    return (max_util, opt_thresh)

def vectorize(A, B, threshold_A, threshold_B, c_plus, c_minus, u_plus, u_minus, prob=0.4):
    
    A = np.asarray(A)
    B = np.asarray(B)

    # Calculated predicted change of both sets of samples
    delta_A = expected(A, c_plus, c_minus)
    delta_B = expected(B, c_plus, c_minus)

    A_matrix = np.repeat(threshold_A, threshold_A.shape[0]).reshape((threshold_A.shape[0], threshold_A.shape[0]))
    B_matrix = np.repeat(threshold_B, threshold_B.shape[0]).reshape((threshold_B.shape[0], threshold_B.shape[0]))
    delta_A_matrix = np.repeat(delta_A, A.shape[0]).reshape((A.shape[0], A.shape[0]))
    delta_B_matrix = np.repeat(delta_B, B.shape[0]).reshape((B.shape[0], B.shape[0]))

    '''
    jitter = np.random.choice([1e-8, -1e-8], size=A.shape, p=[0.4, 0.6])
    noise = (A_matrix + delta_A_matrix) == A_matrix.T
    A_matrix = np.where(noise, A_matrix + jitter, A_matrix)
    noise = (B_matrix + delta_B_matrix) == B_matrix.T
    B_matrix = np.where(noise, B_matrix + jitter, B_matrix)
    '''
    
    A_matrix = np.where(A_matrix + delta_A_matrix > A_matrix.T, A_matrix + delta_A_matrix, A_matrix)
    B_matrix = np.where(B_matrix + delta_B_matrix > B_matrix.T, B_matrix + delta_B_matrix, B_matrix)

    mean_A = np.mean(A_matrix, axis=0)
    mean_B = np.mean(B_matrix, axis=0)
    mean_A = np.repeat(mean_A, mean_A.shape[0]).reshape((mean_A.shape[0], mean_A.shape[0]))
    mean_B = np.repeat(mean_B, mean_B.shape[0]).reshape((mean_B.shape[0], mean_B.shape[0])).T

    util_A = np.sum(expected(A_matrix, u_plus, u_minus), axis=0)
    util_B = np.sum(expected(B_matrix, u_plus, u_minus), axis=0)
    util_A = np.repeat(util_A, util_A.shape[0]).reshape((util_A.shape[0], util_A.shape[0]))
    util_B = np.repeat(util_B, util_B.shape[0]).reshape((util_B.shape[0], util_B.shape[0]))

    return mean_A, mean_B, util_A, util_B

def test_fair_opt_step(A, B, u_plus, u_minus, c_plus, c_minus, alpha):
    
    # Working with array-like structures
    A = np.array(A)
    B = np.array(B)
    np.random.seed(1)
    prob = 0.4

    # Calculate population percentages
    w_a = len(A) / (len(A) + len(B))
    w_b = 1 - w_a

    # Build meshgrid for threshold pairs
    mean_A, mean_B = np.meshgrid(A, B, indexing='ij')
    util_A, util_B = np.meshgrid(A, B, indexing='ij')

    # Matrix of every threshold combination
    mean_A, mean_B, util_A, util_B = vectorize(A, B, A, B, c_plus, c_minus, u_plus, u_minus, prob)

    # Calculate fairness difference at each pair
    fairness_diff = np.abs(mean_A - mean_B)

    # Calculate weighted total utility for each pair
    total_util = w_a * util_A + w_b * util_B

    # Mask utilities violating fairness constraint

    total_util[fairness_diff > alpha] = -np.inf

    flat_idx = np.argmax(total_util)
    i_idx, j_idx = np.unravel_index(flat_idx, total_util.shape)

    opt_A = A[i_idx]  # index along A dimension
    opt_B = B[j_idx]  # index along B dimension

    updated_samples = (mean_A[i_idx, j_idx], mean_B[i_idx, j_idx])  # updated after change

    max_util = total_util[i_idx, j_idx]

    return (opt_A, opt_B, max_util, updated_samples, total_util[i_idx][j_idx])

def alt_fair_opt_step(A, B, u_plus, u_minus, c_plus, c_minus, alpha, range, size):
    
    begin = range[0]
    end = range[1]
    thresholds = np.arange(begin, end, size)

    # Calculate population percentages
    w_a = len(A) / (len(A) + len(B))
    w_b = 1 - w_a

    # Build meshgrid for threshold pairs
    mean_A, mean_B = np.meshgrid(A, B, indexing='ij')
    util_A, util_B = np.meshgrid(A, B, indexing='ij')

    # Matrix of every threshold combination
    mean_A, mean_B, util_A, util_B = vectorize(A, B, thresholds, thresholds, c_plus, c_minus, u_plus, u_minus)

    # Calculate fairness difference at each pair
    fairness_diff = np.abs(mean_A - mean_B)

    # Calculate weighted total utility for each pair
    total_util = w_a * util_A + w_b * util_B

    # Mask utilities violating fairness constraint

    total_util[fairness_diff > alpha] = -np.inf

    flat_idx = np.argmax(total_util)
    i_idx, j_idx = np.unravel_index(flat_idx, total_util.shape)

    opt_A = A[i_idx]  # index along A dimension
    opt_B = B[j_idx]  # index along B dimension

    updated_samples = (mean_A[i_idx, j_idx], mean_B[i_idx, j_idx])  # updated after change

    max_util = total_util[i_idx, j_idx]

    return (opt_A, opt_B, max_util, updated_samples, total_util[i_idx][j_idx])
