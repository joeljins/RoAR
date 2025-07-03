import numpy as np
from scipy.optimize import fsolve

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

def opt_threshold(domain, u_plus, u_minus):
    """
    Find optimal threshold by solving for where expected value equals zero
    """
    # Define the function to find roots of
    def objective(x):
        return expected(x, u_plus, u_minus)
    
    # Find root within the domain
    if isinstance(domain, (list, tuple)) and len(domain) == 2:
        # If domain is a range, use bounds
        from scipy.optimize import brentq
        try:
            root = brentq(objective, domain[0], domain[1])
        except ValueError:
            # If no root in interval, use fsolve with midpoint
            root = fsolve(objective, x0=(domain[0] + domain[1]) / 2)[0]
    else:
        # Use fsolve with initial guess
        root = fsolve(objective, x0=0)[0]
    
    return root

def change(A, B, c_plus, c_minus, u_plus, u_minus, prob=0.4):
    
    # Working with array-like structures
    A = np.asarray(A)
    B = np.asarray(B)

    # Calculated predicted change of both sets of samples
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

def fair_opt_step(A, B, u_plus, u_minus, c_plus, c_minus, alpha):
    
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

    return (opt_A, opt_B, max_util, updated_samples, total_util[i_idx][j_idx])

def alt_fair_opt_step(pop_A, pop_B, u_plus, u_minus, c_plus, c_minus, alpha, range_param, size):

    begin, end = range_param
    thresholds = np.arange(begin, end, size)
    T = len(thresholds)
    
    pop_A = np.asarray(pop_A)
    pop_B = np.asarray(pop_B)
    n, m = len(pop_A), len(pop_B)
    
    # Weights
    w_a = n / (n + m)
    w_b = 1 - w_a
    
    # Compute deltas once
    delta_A = expected(pop_A, c_plus, c_minus)  # Shape: (n,)
    delta_B = expected(pop_B, c_plus, c_minus)  # Shape: (m,)
    
    # Create threshold meshgrid
    thresh_a, thresh_b = np.meshgrid(thresholds, thresholds, indexing='ij')  # (T, T)
    
    # Reshape for broadcasting
    thresh_a_bc = thresh_a[:, :, np.newaxis]  # (T, T, 1)
    thresh_b_bc = thresh_b[:, :, np.newaxis]  # (T, T, 1)
    
    # Expand populations and deltas for broadcasting
    A_bc = pop_A[np.newaxis, np.newaxis, :]      # (1, 1, n)
    B_bc = pop_B[np.newaxis, np.newaxis, :]      # (1, 1, m)
    delta_A_bc = delta_A[np.newaxis, np.newaxis, :]  # (1, 1, n)
    delta_B_bc = delta_B[np.newaxis, np.newaxis, :]  # (1, 1, m)
    
    # Apply adjustments conditionally
    # Only apply delta where original + delta > threshold
    condition_A = (A_bc + delta_A_bc) > thresh_a_bc  # (T, T, n)
    condition_B = (B_bc + delta_B_bc) > thresh_b_bc  # (T, T, m)
    
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
    util_masked = np.where(mean_diff < alpha, util_combined, -np.inf)
    
    # Find optimal combination
    if np.all(util_masked == -np.inf):
        # No fair solution exists
        return None, None, None, None, -np.inf
    
    i, j = np.unravel_index(np.argmax(util_masked), util_masked.shape)
    
    # Extract optimal results
    opt_A = a_adj[i, j]
    opt_B = b_adj[i, j]
    thresh_A = thresh_a[i, j]
    thresh_B = thresh_b[i, j]
    max_util = util_masked[i, j]
    
    return opt_A, opt_B, thresh_A, thresh_B, max_util

def alt_fair_step(A, B, u_plus, u_minus, c_plus, c_minus, alpha, range, size):
    begin = range[0]
    end = range[1]
    thresholds = np.arange(begin, end, size)

    max_util = -np.inf
    opt_A = None
    opt_B = None
    thresh_A = None
    thresh_B = None 

    w_a = len(A) / (len(A) + len(B))
    w_b = 1 - w_a
    delta_A = expected(A, c_plus, c_minus)
    delta_B = expected(B, c_plus, c_minus)

    for threshold_A in thresholds:
            for threshold_B in thresholds:
                a = np.where(A + delta_A > threshold_A, A + delta_A, A )
                b = np.where(B + delta_B > threshold_B, B + delta_B, B )
                if np.abs( np.mean(a) - np.mean(b) ) >= alpha:
                    continue
                util = w_a * np.sum(expected(a, u_plus, u_minus)) + w_b * np.sum(expected(b, u_plus, u_minus))
                if util >= max_util: 
                    max_util = util
                    opt_A = a
                    opt_B = b
                    thresh_A = threshold_A
                    thresh_B = threshold_B

    return opt_A, opt_B, thresh_A, thresh_B, max_util
