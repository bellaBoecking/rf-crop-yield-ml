"""
target_similarity.py

Local target instability diagnostics with mixed-feature neighborhoods.

This module computes a local variance measure of the target variable using a custom
mixed numeric-categorical distance designed to preserve numeric smoothness. Unlike Gower 
distance, which discretizes categorical mismatches and can disrupt continuity in
numeric space, this formulation maintains smooth local neighborhoods for numeric
features while still accounting for categorical differences.

Numeric features are MinMax-scaled and compared using L1 distance, while categorical
features contribute a normalized Hamming distance. The combined distance induces a geometry 
that is better aligned with local smoothness assumptions.

These diagnostics quantify local target instability under a distance designed specifically
for smoothness-aware analysis and are intended for exploratory evaluation rather than model
training.
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np
import pandas as pd
from .feature_similarity import gower_distance

logger = logging.getLogger(__name__)

def compute_local_y_variance(X, y, numeric_features, categorical_features, k = 10):
    """
    Compute local target variance using smooth mixed-feature neighborhoods.

    Distances are combined using weights proportional to the number of numeric and 
    categorical features, yielding neighborhoods that reflect both smooth numeric evaluation and 
    categorical structure. Local variance of targets is then computed within each neighborhood
    as a diagnostic of local instability.

    Args: 
        X: Training feature DataFrame
        y: Validation feature DataFrame
        numeric_features: Numeric feature names
        categorical_features: Categorical feature names

    Returns:
        local_vars: Calculated local variances
    """

    scaler = MinMaxScaler()

    X_num = scaler.fit_transform(X[numeric_features])
    X_cat = X[categorical_features].astype(str).values
    

    def mixed_distance(i, j):
        """
        Custom metric distance function, computes mixed-feature distance between two observations.

        Numeric features contribute a normalized L1 distance computed as the mean absolute difference across
        numeric features. Categorical features contribute a normalized Hamming distance, computed
        as the fraction of mismatched categories.

        The final distance is a weighted sum of numerical and categorical components, with weights
        proportional to the number of features of each type.

        Args:
            i: Index of first observation
            j: Index of second observation

        Returns:
            mixed_dist: Mixed-feature distance between observations i and j. Lower values indicate
            greater similarity.
        """
        num_dist = np.nanmean(np.abs(X_num[i] - X_num[j]))
        cat_dist = np.mean(X_cat[i] != X_cat[j])
        feature_sum = len(numeric_features) + len(categorical_features)

        mixed_dist = (len(numeric_features) / feature_sum) * num_dist + (len(categorical_features) / feature_sum) * cat_dist

        return mixed_dist

    n = len(X)
    local_vars = []

    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dists.append((mixed_distance(i, j), j)) 

        dists.sort(key = lambda x: x[0]) 
        neighbors = [j for d, j in dists if d > 0][:k] 
        local_vars.append(np.var(y.iloc[neighbors], ddof = 1))

    local_vars = pd.Series(local_vars, index = X.index) 

    return local_vars
