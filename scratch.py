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

    a, b = np.meshgrid(thresholds, thresholds, indexing='ij')  # (T, T)
    a_exp = a[:, :, None]  # (T, T, 1)
    b_exp = b[:, :, None]  # (T, T, 1)

    # Expand A/B and compute deltas
    a_matrix = A[None, None, :]               # (1, 1, n)
    b_matrix = B[None, None, :]               # (1, 1, m)
    delta_A = expected(A, c_plus, c_minus)[None, None, :]  # (1, 1, n)
    delta_B = expected(B, c_plus, c_minus)[None, None, :]  # (1, 1, m)

    # Apply delta conditionally based on thresholds
    a_adj = np.where(a_matrix + delta_A > a_exp, a_matrix + delta_A, a_matrix)  # (T, T, n)
    b_adj = np.where(b_matrix + delta_B > b_exp, b_matrix + delta_B, b_matrix)  # (T, T, m)

    # Compute expected utility for each grid cell
    util_A = expected(a_adj, u_plus, u_minus).sum(axis=2)  # (T, T)
    util_B = expected(b_adj, u_plus, u_minus).sum(axis=2)  # (T, T)
    util = w_a * util_A + w_b * util_B                     # (T, T)

    # Fairness check: mean difference under alpha
    mean_diff = np.abs(np.mean(a_adj) - np.mean(b_adj))  # (T, T)
    util[mean_diff >= alpha] = -np.inf  # Mask unfair combinations

    # Find the best combination
    i, j = np.unravel_index(np.argmax(util), util.shape)
    opt_A = a_adj[i, j]
    opt_B = b_adj[i, j]
    thresh_A = a[i, j]
    thresh_B = b[i, j]
    max_util = util[i, j]

    return opt_A, opt_B, thresh_A, thresh_B, max_util

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
