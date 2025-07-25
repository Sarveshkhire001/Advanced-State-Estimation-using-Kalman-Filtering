# plotting_utils.py

import numpy as np
import plotly.graph_objects as go

def plot_ellipse(fig, mean, cov, color, label):
    # Function to plot uncertainty ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov[:2, :2])

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)

    t = np.linspace(0, 2 * np.pi, 100)

    x_points = mean[0] + np.sqrt(eigenvalues[0]) * np.cos(angle) * np.cos(t) - np.sqrt(eigenvalues[1]) * np.sin(angle) * np.sin(t)
    y_points = mean[1] + np.sqrt(eigenvalues[0]) * np.sin(angle) * np.cos(t) + np.sqrt(eigenvalues[1]) * np.cos(angle)*np.sin(t)

    ellipse = go.Scatter(
        x=x_points,
        y=y_points,
        mode='lines',
        line=dict(color=color),
        name=label
    )

    fig.add_trace(ellipse)

