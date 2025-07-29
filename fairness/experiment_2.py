from utils import expected, alt_fair_opt_step
import plotly.graph_objects as go
import numpy as np

def experiment_2(a, b, u_plus, u_minus, c_plus, c_minus, alpha, thresholds, w_a, w_b):
    # B starts out at the optimal threshold regardless of Alpha
    results = alt_fair_opt_step(a, b, u_plus, u_minus, c_plus, c_minus, alpha, thresholds)

    x_thresh_A = []
    thresh_B = results[3] 
    y_util = []
    y_means_diff = []

    delta_A = expected(a, c_plus, c_minus)
    delta_B = expected(b, c_plus, c_minus)

    B = np.where(b > thresh_B, b + delta_B, b)
    util_B = np.sum(expected(B, u_plus, u_minus))

    for threshold in thresholds:
        A = np.where(a > threshold, a + delta_A, a)
        
        # Mean difference and utils
        diff = np.abs(np.mean(A) - np.mean(B))
        util_A = np.sum(expected(A, u_plus, u_minus))
        total_util = w_a * util_A +  w_b * util_B

        # For plot
        x_thresh_A.append(threshold)
        y_means_diff.append(diff)
        y_util.append(total_util)

    # Plotting
    alpha_line = go.Scatter(x=x_thresh_A, y=[alpha]*len(x_thresh_A), name='Alpha', mode='lines')
    means_diff = go.Scatter(x=x_thresh_A, name='Mean Diff', mode='markers', y=y_means_diff, fill='tonexty',fillcolor='rgba(0,0,255,0.2)')
    util = go.Scatter(x=x_thresh_A, y=y_util, name='Total Utility', mode='markers', yaxis='y2', marker=dict(color='green'))
    

    fig = go.Figure(data=[alpha_line, means_diff, util])
    fig.update_layout(
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
    fig.show()
