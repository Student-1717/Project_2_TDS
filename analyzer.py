import pandas as pd
import numpy as np
from scipy.stats import linregress
from util import plot_scatter_base64

def compute_count(df, column, threshold=None):
    if threshold is not None:
        return int((df[column] > threshold).sum())
    return len(df)

def compute_correlation(df, col1, col2):
    return float(df[col1].corr(df[col2]))

def compute_regression_slope(df, x_col, y_col):
    x = df[x_col].values
    y = df[y_col].values
    slope, _, _, _, _ = linregress(x, y)
    return float(slope)

def generate_scatter_plot(df, x_col, y_col, regression=True):
    return plot_scatter_base64(df, x_col, y_col, regression)
