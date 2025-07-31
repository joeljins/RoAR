from utils import expected, alt_fair_opt_step
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm

def experiment_2(a, b, u_plus, u_minus, c_plus, c_minus, alpha, w_a, w_b, thresh_B, array, alphas=None):
    # B starts out at the optimal threshold regardless of Alpha

    x_thresh_A = []
    y_util = []
    y_means_diff = []

    delta_A = expected(a, c_plus, c_minus)
    delta_B = expected(b, c_plus, c_minus)

    B = np.where(b > thresh_B, b + delta_B, b)
    util_B = np.sum(expected(B, u_plus, u_minus))

    for item in tqdm(array):
        A = np.where(a > item, a + delta_A, a)
        
        # Mean difference and utils
        diff = np.abs(np.mean(A) - np.mean(B))
        util_A = np.sum(expected(A, u_plus, u_minus))
        total_util = w_a * util_A +  w_b * util_B

        # For plot
        x_thresh_A.append(item)
        y_means_diff.append(diff)
        y_util.append(total_util)

    # Plotting
    alpha_line = go.Scatter(x=x_thresh_A, y=[alpha]*len(x_thresh_A), name='Alpha', mode='lines')
    means_diff = go.Scatter(x=x_thresh_A, name='Mean Diff', mode='markers', y=y_means_diff, fill='tonexty',fillcolor='rgba(0,0,255,0.2)')
    util = go.Scatter(x=x_thresh_A, y=y_util, name='Total Utility', mode='markers', yaxis='y2', marker=dict(color='green'))
    

    fig1 = go.Figure(data=[alpha_line, means_diff, util])
    fig1.update_layout(
        height = 600,
        title='Threshold A with Fixed Threshold B vs Mean Difference',
        xaxis=dict(title="Threshold A"),
        yaxis=dict(title="| mean(A) - mean(B) |"),
        yaxis2=dict(
            title="Utility",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            x=0, 
            #y=1,
            xanchor='right',
            yanchor='top'
        ),
        showlegend=True,
    )

    if alphas is None:
        return fig1

    y_means_diff = np.array(y_means_diff)
    x_alphas = []
    y_thresh_A = []
    prev = 0
    arr = np.argsort(y_means_diff)
    for alpha in alphas:
        arr = arr[prev:]
        condition = y_means_diff[arr] < alpha
        indices = arr[condition]
        if indices.size == 0:
            continue
        max_util_idx = indices.max()

        x_alphas.append(alpha)
        y_thresh_A.append(x_thresh_A[max_util_idx])
        
        prev = np.where(arr == max_util_idx)[0][0] + 1 

    thresholds = go.Scatter(x = x_alphas, y = y_thresh_A, mode = 'markers')
    fig2 = go.Figure(data=[thresholds])
    fig2.update_layout(
        height = 600,
        title='Alpha vs Fair Threshold A',
        xaxis=dict(title="Alpha"),
        yaxis=dict(title="Threshold A"),
        legend=dict(
            x=0, 
            #y=1,
            xanchor='right',
            yanchor='top'
        ),
        showlegend=True,
    )

    return fig1, fig2
