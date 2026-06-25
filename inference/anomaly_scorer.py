import numpy as np
from sklearn.covariance import EmpiricalCovariance

class AnomalyScorer:
    """
    Calculates Mahalanobis distance from a normative PInSoRo baseline cluster.
    """
    def __init__(self):
        self.cov_estimator = EmpiricalCovariance()
        self.is_fitted = False

    def fit_baseline(self, z_baseline_matrix: np.ndarray):
        """
        Fits the covariance matrix using embeddings from the normative dataset.
        z_baseline_matrix: [N, 256]
        """
        self.cov_estimator.fit(z_baseline_matrix)
        self.is_fitted = True

    def score_clip(self, z_vector: np.ndarray) -> float:
        """
        Returns the Mahalanobis distance.
        z_vector: [1, 256]
        """
        if not self.is_fitted:
            raise ValueError("Baseline must be fitted before scoring.")
            
        dist = self.cov_estimator.mahalanobis(z_vector)
        # Normalize to [0, 1] range conceptually via sigmoid or max scaling downstream
        return dist[0]
