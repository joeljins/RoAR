from utils import expected, alt_fair_opt_step
import numpy as np

def experiment_1(a, b, u_plus, u_minus, c_plus, c_minus, alpha, domain, interval):
    # B starts out at the optimal threshold regardless of Alpha
    results = alt_fair_opt_step(a, b, u_plus, u_minus, c_plus, c_minus, alpha, domain, interval)

    thresholds_A = []
    thresh_B = results[3] 
    y_means_diff = []

    delta_A = expected(a, c_plus, c_minus)
    delta_B = expected(b, c_plus, c_minus)

    B = np.where(b > thresh_B, b + delta_B, b)
    util_B = np.sum(expected(B, u_plus, u_minus))

    utility_by_alpha = []

    thresholds = np.arange(domain[0], domain[1], interval)
    temp_list = [0.39, 1]
    for alpha in temp_list:
        utility = []
        for threshold in thresholds:
            A = np.where(a > threshold, a + delta_A, a)
            util_A = np.sum(expected(A, u_plus, u_minus))
            utility.append(util_A + util_B)
        thresholds_A.append(threshold)
        utility_by_alpha.append(utility)
    print(utility_by_alpha[0]==utility_by_alpha[1])

    base = utility_by_alpha[0]
    temp_list = np.arange(0.39, 1, 0.1)
    utility_by_alpha = []
    for alpha in temp_list:
        utility = []
        for threshold in thresholds:
            A = np.where(a > threshold, a + delta_A, a)
            util_A = np.sum(expected(A, u_plus, u_minus))
            utility.append(util_A + util_B)
        thresholds_A.append(threshold)
        utility_by_alpha.append(utility)

    for i in utility_by_alpha:
        if i != base:
            print("Not equal")
            break

