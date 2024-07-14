import plotly.graph_objs as go
import numpy as np
import math
import time
import argparse

def collatz(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

def do_collatz(n):
    sols = [n]
    while True:
        n = collatz(n)
        sols.append(n)
        if n == 1:
            return sols

def plot_collatz(sols):
    steps = np.arange(len(sols))
    fig = go.Figure()
    
    # Plotting the Collatz sequence
    fig.add_trace(go.Scatter(x=steps, y=sols, mode='lines+markers', name='Collatz Sequence', line=dict(color='blue')))
    
    # Highlighting the starting point (green) and sequence points (red)
    fig.add_trace(go.Scatter(x=[0], y=[sols[0]], mode='markers', name='Start', marker=dict(color='green', size=10)))
    fig.add_trace(go.Scatter(x=steps, y=sols, mode='markers', name='Sequence Points', marker=dict(color='red', size=5)))
    
    # Customizing layout
    fig.update_layout(
        title=f'Collatz Numbers for {sols[0]}',
        xaxis_title='Step Number',
        yaxis_title='Number',
        xaxis=dict(tickmode='linear'),
        showlegend=True
    )
    
    fig.show()

def main(n):
    print(f'Computing Collatz Numbers for {n}')
    start = time.time()
    sols = do_collatz(n)
    end = time.time()
    print(f'Time Taken: {end-start} seconds')
    plot_collatz(sols)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and Plot Collatz Numbers')
    parser.add_argument('-n', '--number', help='input number to compute collatz for', default=4, required=True)
    args = parser.parse_args()
    num = int(args.number)
    main(num)

