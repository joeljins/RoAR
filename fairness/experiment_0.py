from utils import expected
from tqdm import tqdm
import numpy as np

def experiment_0(a, b, u_plus, u_minus, c_plus, c_minus, w_a, w_b, alphas, t_As, t_Bs):
    x_alphas = []
    y_t_A, y_t_B = [], []

    delta_A = expected(a, c_plus, c_minus)
    delta_B = expected(b, c_plus, c_minus)

    for alpha in tqdm(alphas):
        max_util = -np.inf
        thresh_A = None
        thresh_B = None

        for t_A in t_As:
            for t_B in t_Bs:
                A = np.where(a > t_A, a + delta_A, a)
                B = np.where(b > t_B, b + delta_B, b)
                util = w_a * np.sum(expected(A, u_plus, u_minus)) + w_b * np.sum(expected(B, u_plus, u_minus))

                if np.abs(np.mean(A) - np.mean(B)) > alpha:
                    continue

                if util >= max_util:
                    max_util = util
                    thresh_A = t_A
                    thresh_B = t_B

        if thresh_A is not None and thresh_B is not None:
            x_alphas.append(alpha)
            y_t_A.append(thresh_A)
            y_t_B.append(thresh_B)
    
    return x_alphas, y_t_A, y_t_B