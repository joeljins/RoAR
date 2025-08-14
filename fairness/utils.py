import numpy as np
from scipy.optimize import fsolve

def p(x) -> float:
    '''
    p : a monotonic function, X -> [0,1], e.g. sigmoid
    '''
    return 1 / (1 + np.exp(-x))

def expected(x, plus, minus):
    '''p(x) * plus + (1-p(x)) * minus'''
    return p(x) * plus + (1-p(x)) * minus

def opt_step(x, u_plus, u_minus, c_plus, c_minus):
    '''Applies change to x if expected utility is positive'''
    '''Return smallest threshold where utility is positive'''
    x = np.asarray(x)

    exp_util = expected(x, u_plus, u_minus) 
    delta_X = expected(x, c_plus, c_minus)

    X = np.where(exp_util > 0, x + delta_X, x) 

    # Smallest threshold where utility is positive
    thresholds = x[exp_util > 0]
    opt_thresh = np.min(thresholds) if thresholds.size > 0 else None

    # array of updated samples, min sample whose util is positive
    return (X, opt_thresh)


def change(A, B, c_plus, c_minus, u_plus, u_minus, prob=0.4):
    A = np.asarray(A)
    B = np.asarray(B)

    delta_A = expected(A, c_plus, c_minus)
    delta_B = expected(B, c_plus, c_minus)

    A_matrix = np.repeat(A, A.shape[0]).reshape((A.shape[0], A.shape[0]))
    B_matrix = np.repeat(B, B.shape[0]).reshape((B.shape[0], B.shape[0]))
    delta_A_matrix = np.repeat(delta_A, A.shape[0]).reshape((A.shape[0], A.shape[0]))
    delta_B_matrix = np.repeat(delta_B, B.shape[0]).reshape((B.shape[0], B.shape[0]))

    jitter = np.random.choice([1e-8, -1e-8], size=A.shape, p=[0.4, 0.6])
    noise = (A_matrix + delta_A_matrix) == A_matrix.T
    A_matrix = np.where(noise, A_matrix + jitter, A_matrix)
    noise = (B_matrix + delta_B_matrix) == B_matrix.T
    B_matrix = np.where(noise, B_matrix + jitter, B_matrix)
    
    A_matrix = np.where(A_matrix > A_matrix.T, A_matrix + delta_A_matrix, A_matrix)
    B_matrix = np.where(B_matrix > B_matrix.T, B_matrix + delta_B_matrix, B_matrix)

    mean_A = np.mean(A_matrix, axis=0)
    mean_B = np.mean(B_matrix, axis=0)
    mean_A = np.repeat(mean_A, mean_A.shape[0]).reshape((mean_A.shape[0], mean_A.shape[0]))
    mean_B = np.repeat(mean_B, mean_B.shape[0]).reshape((mean_B.shape[0], mean_B.shape[0])).T

    util_A = np.sum(expected(A_matrix, u_plus, u_minus), axis=0)
    util_B = np.sum(expected(B_matrix, u_plus, u_minus), axis=0)
    util_A = np.repeat(util_A, util_A.shape[0]).reshape((util_A.shape[0], util_A.shape[0]))
    util_B = np.repeat(util_B, util_B.shape[0]).reshape((util_B.shape[0], util_B.shape[0]))

    return mean_A, mean_B, util_A, util_B, A_matrix, B_matrix

def fair_opt_step(A, B, u_plus, u_minus, c_plus, c_minus, alpha):
    A = np.array(A)
    B = np.array(B)
    np.random.seed(1)
    prob = 0.4

    w_a = len(A) / (len(A) + len(B))
    w_b = 1 - w_a

    # Build meshgrid for threshold pairs
    mean_A, mean_B = np.meshgrid(A, B, indexing='ij')
    util_A, util_B = np.meshgrid(A, B, indexing='ij')

    mean_A, mean_B, util_A, util_B = change(A, B, c_plus, c_minus, u_plus, u_minus, prob)

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

    return (opt_A, opt_B, max_util, updated_samples)

def itvl_fair_opt_step(a, b, u_plus, u_minus, c_plus, c_minus, alpha, thresholds):
    '''
    Vectorized grid search for pair of thresholds(intervals) with max utilty and fairness constraint fulfilled
    '''
    T = len(thresholds)
    
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    
    # Weights
    w_a = n / (n + m)
    w_b = 1 - w_a
    
    # Compute deltas
    delta_A = expected(a, c_plus, c_minus)  # Shape: (n,)
    delta_B = expected(b, c_plus, c_minus)  # Shape: (m,)
    
    # Create threshold meshgrid
    thresh_a, thresh_b = np.meshgrid(thresholds, thresholds, indexing='ij')  # (T, T)
    
    # Reshape for broadcasting
    thresh_a_bc = thresh_a[:, :, np.newaxis]  # (T, T, 1)
    thresh_b_bc = thresh_b[:, :, np.newaxis]  # (T, T, 1)
    
    # Expand populations and deltas for broadcasting
    A_bc = a[np.newaxis, np.newaxis, :]      # (1, 1, n)
    B_bc = b[np.newaxis, np.newaxis, :]      # (1, 1, m)
    delta_A_bc = delta_A[np.newaxis, np.newaxis, :]  # (1, 1, n)
    delta_B_bc = delta_B[np.newaxis, np.newaxis, :]  # (1, 1, m)
    
    '''
    # Apply adjustments conditionally
    # Only apply delta where original + delta > threshold
    condition_A = (A_bc + delta_A_bc) > thresh_a_bc  # (T, T, n)
    condition_B = (B_bc + delta_B_bc) > thresh_b_bc  # (T, T, m)
    '''
    condition_A = A_bc > thresh_a_bc  # (T, T, n)
    condition_B = B_bc > thresh_b_bc  # (T, T, m)

    
    a_adj = np.where(condition_A, A_bc + delta_A_bc, A_bc)  # (T, T, n)
    b_adj = np.where(condition_B, B_bc + delta_B_bc, B_bc)  # (T, T, m)
    
    # Compute utilities for each threshold combination
    util_A_per_sample = expected(a_adj, u_plus, u_minus)  # (T, T, n)
    util_B_per_sample = expected(b_adj, u_plus, u_minus)  # (T, T, m)
    
    util_A_total = np.sum(util_A_per_sample, axis=2)  # (T, T)
    util_B_total = np.sum(util_B_per_sample, axis=2)  # (T, T)
    
    # Combined utility
    util_combined = w_a * util_A_total + w_b * util_B_total  # (T, T)
    
    # Fairness constraint: mean difference
    mean_A = np.mean(a_adj, axis=2)  # (T, T)
    mean_B = np.mean(b_adj, axis=2)  # (T, T)
    mean_diff = np.abs(mean_A - mean_B)  # (T, T)
    
    # Apply fairness mask
    util_masked = np.where(mean_diff <= alpha, util_combined, -np.inf)
    
    
    # Find optimal combination
    if np.all(util_masked == -np.inf):
        # No fair solution exists
        return None, None, None, None, -np.inf
    
    
    i, j = np.unravel_index(np.argmax(util_masked), util_masked.shape)

    '''
    flat = util_masked.flatten()
    epsilon = 0.1
    arg_max = 0
    for k in reversed(range(len(flat))):
        if flat[k] >= flat[arg_max] + epsilon:
            arg_max = k

    i, j = np.unravel_index(arg_max, util_masked.shape)
    '''

    
    # Extract optimal results
    opt_A = a_adj[i, j]
    opt_B = b_adj[i, j]
    thresh_A = thresh_a[i, j]
    thresh_B = thresh_b[i, j]
    max_util = util_masked[i, j]
    
    return opt_A, opt_B, thresh_A, thresh_B, max_util

def sampl_fair_opt_step(A, B, u_plus, u_minus, c_plus, c_minus, alpha):
    '''
    Vectorized grid search for pair of thresholds(samples) with max utilty and fairness constraint fulfilled
    '''
    A = np.asarray(A)
    B = np.asarray(B)
    np.random.seed(1)

    w_a = len(A) / (len(A) + len(B))
    w_b = 1 - w_a

    mean_A, mean_B, util_A, util_B = mat_vect(A, B, c_plus, c_minus, u_plus, u_minus)

    fairness_diff = np.abs(mean_A - mean_B)
    total_util = w_a * util_A + w_b * util_B

    # Apply fairness constraint
    total_util_masked = np.where(fairness_diff <= alpha, total_util, -np.inf)

    # Find best pair (max utility under fairness constraint)

    flat_idx = np.argmax(total_util_masked)
    i, j = np.unravel_index(flat_idx, total_util.shape)

    opt_A = A[i]
    opt_B = B[j]
    updated_samples = (mean_A[i, j], mean_B[i, j])
    max_util = total_util_masked[i, j]

    return (opt_A, opt_B, max_util, updated_samples)

def mat_vect(A, B, c_plus, c_minus, u_plus, u_minus, prob=0.4):
    '''Matrix-vectorization of grid search'''
    A = np.asarray(A)
    B = np.asarray(B)

    delta_A = expected(A, c_plus, c_minus)
    delta_B = expected(B, c_plus, c_minus)

    A_matrix = A[:, None]  # shape (n, 1)
    B_matrix = B[:, None]  # shape (m, 1)

    delta_A_matrix = delta_A[:, None]
    delta_B_matrix = delta_B[:, None]

    # Add small jitter to break ties
    #jitter_A = np.random.choice([1e-8, -1e-8], size=A.shape, p=[0.4, 0.6])
    #jitter_B = np.random.choice([1e-8, -1e-8], size=B.shape, p=[0.4, 0.6])

    A_matrix_adj = np.where(A_matrix > A_matrix.T, A_matrix + delta_A_matrix, A_matrix)
    B_matrix_adj = np.where(B_matrix > B_matrix.T, B_matrix + delta_B_matrix, B_matrix)

    # Break ties
    #A_matrix_adj = np.where(A_matrix + delta_A_matrix == A_matrix.T, A_matrix + jitter_A[:, None], A_matrix_adj)
    #B_matrix_adj = np.where(B_matrix + delta_B_matrix == B_matrix.T, B_matrix + jitter_B[:, None], B_matrix_adj)

    mean_A = np.mean(A_matrix_adj, axis=0)
    mean_B = np.mean(B_matrix_adj, axis=0)

    util_A = np.sum(expected(A_matrix_adj, u_plus, u_minus), axis=0)
    util_B = np.sum(expected(B_matrix_adj, u_plus, u_minus), axis=0)

    # Convert to meshgrids for threshold pairs
    mean_A_grid, mean_B_grid = np.meshgrid(mean_A, mean_B, indexing='ij')
    util_A_grid, util_B_grid = np.meshgrid(util_A, util_B, indexing='ij')

    return mean_A_grid, mean_B_grid, util_A_grid, util_B_grid
