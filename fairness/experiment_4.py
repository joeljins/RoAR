from utils import expected, alt_fair_opt_step
import plotly.graph_objects as go
import numpy as np

# alphas is a list of alpha values
# domain is a tuple (min, max) for the threshold range
# interval is the step size for thresholds
def experiment_4(a, b, u_plus, u_minus, c_plus, c_minus, alphas, thresholds, w_a, w_b, epsilon):
    
    # B starts out at the optimal threshold regardless of Alpha
    results = alt_fair_opt_step(a, b, u_plus, u_minus, c_plus, c_minus, np.inf, thresholds, 0.01)

    x_alphas = []
    thresh_B = results[3] 
    y_util = []
    y_thresh_A = []

    delta_A = expected(a, c_plus, c_minus)
    delta_B = expected(b, c_plus, c_minus)

    B = np.where(b > thresh_B, b + delta_B, b)
    util_B = np.sum(expected(B, u_plus, u_minus))

    for alpha in alphas:
        opt, max_util = -np.inf, -np.inf
        for threshold in thresholds:
            A = np.where(a > threshold, a + delta_A, a)
            
            # Mean difference and utils
            diff = np.abs(np.mean(A) - np.mean(B))
            if diff > alpha:
                continue
            util_A = np.sum(expected(A, u_plus, u_minus))
            total_util = w_a * util_A +  w_b * util_B
            if total_util > max_util + epsilon:
                max_util = total_util
                opt = threshold
            
        # For plot
        if opt == -np.inf:
            continue
        x_alphas.append(alpha)
        y_thresh_A.append(opt)
        y_util.append(max_util)

    # Plotting
    fair_thresholds = go.Scatter(x=x_alphas,  y=y_thresh_A, name='Fair Threshold A', mode='markers')
    util = go.Scatter(x=x_alphas, y=y_util, name='Total Utility', mode='markers', yaxis='y2', marker=dict(color='green'))
    

    fig = go.Figure(data=[fair_thresholds, util])
    fig.update_layout(
        height = 600,
        title='Alpha vs Fair Threshold and Utility (with epsilon={})'.format(epsilon),
        xaxis=dict(title="Alpha"),
        yaxis=dict(title="Threshold A"),
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
    fig.show()
