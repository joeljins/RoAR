from utils import expected, alt_fair_opt_step
import plotly.graph_objects as go
import numpy as np

def experiment_3(a, b, u_plus, u_minus, c_plus, c_minus, alpha, thresholds, w_a, w_b):
    # B starts out at the optimal threshold regardless of Alpha
    results = alt_fair_opt_step(a, b, u_plus, u_minus, c_plus, c_minus, alpha, thresholds)

    thresh_B = results[3] 

    delta_A = expected(a, c_plus, c_minus)
    delta_B = expected(b, c_plus, c_minus)

    B = np.where(b > thresh_B, b + delta_B, b)
    util_B = w_b * np.sum(expected(B, u_plus, u_minus))

    #c1_all = {}
    all_sets = {}
    greatest = -np.inf

    for thresh_A in thresholds:
        A = np.where(a > thresh_A, a + delta_A, a)
        diff = np.abs(np.mean(A) - np.mean(B))
            
        c1 = np.where((a > thresh_A) & (expected(a, u_plus, u_minus) >= 0) & (delta_A >= 0))[0]
        c2 = np.where((a > thresh_A) & (expected(a, u_plus, u_minus) >= 0) & (delta_A < 0))[0]
        c3 = np.where((a > thresh_A) & (expected(a, u_plus, u_minus) < 0) & (delta_A >= 0))[0]
        c4 = np.where((a > thresh_A) & (expected(a, u_plus, u_minus) < 0) & (delta_A < 0))[0]

        #c1_all[round(thresh_A, 2)] = w_a * expected(A[c1], u_plus, u_minus) 
        #c1_all.append(expected(A[c1], u_plus, u_minus))
        all_sets[round(thresh_A, 2)] = A

        total = np.where(a > thresh_A)[0]
        util_A = w_a * np.sum(expected(A, u_plus, u_minus))

        util_above = w_a * np.sum(expected(A[np.concatenate((c1, c2, c3, c4))], u_plus, u_minus))
        util_below = w_a * np.sum(expected(A[a<=thresh_A], u_plus, u_minus))
        
        if diff > alpha:
            print(f'Violation at threshold {round(thresh_A, 2)}')
        else:
            greatest = max(util_A, greatest)
        print(f'Threshold {round(thresh_A, 2)}: {util_A}')
        #print(f'{np.isclose(util_above + util_below, util_A)}')
        print(f'{util_below} + {util_above}')
        #print(f'Cat 1 Util: {c1_all[round(thresh_A, 2)]}')
        #print(f'{len(total)}:   C1: {len(c1)} C2: {len(c2)} C3: {len(c3)} C4: {len(c4)} \n' )
        print(
        f'Total: {len(total)}\n'
        f'C1:    {len(c1)}\n'
        f'C2:    {len(c2)}\n'
        f'C3:    {len(c3)}\n'
        f'C4:    {len(c4)}\n'
        )
    print(f'Greatest utility: {greatest}')
    return all_sets


    alpha_line = go.Scatter(x=x_thresh_A, y=[alpha]*len(x_thresh_A), name='Alpha Line', mode='lines')
    util = go.Scatter(x=x_thresh_A, y=y_util, name='Total Utility', mode='markers', yaxis='y2')
    means_diff = go.Scatter(x=x_thresh_A, name='Mean Diff', y=y_means_diff)
    
    fig = go.Figure(data=[util, means_diff, alpha_line])
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
