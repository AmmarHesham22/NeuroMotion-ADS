import torch
import numpy as np
from sklearn.metrics import r2_score, silhouette_score

def compute_pearson_r(y_true, y_pred):
    """
    Computes Pearson Correlation Coefficient.
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    # Handle single batch scalar case
    if len(y_true) < 2:
        return 0.0
        
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    # Handle NaN if variance is zero
    if np.isnan(correlation):
        return 0.0
    return correlation

def compute_r2(y_true, y_pred):
    """
    Computes R-squared metric.
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    if len(y_true) < 2:
        return 0.0
    return r2_score(y_true, y_pred)

def compute_silhouette(z, labels):
    """
    Computes silhouette score for embedding quality evaluation.
    Requires at least 2 clusters.
    """
    z_np = z.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    unique_labels = np.unique(labels_np)
    
    if len(unique_labels) < 2:
        return 0.0
        
    return silhouette_score(z_np, labels_np)
