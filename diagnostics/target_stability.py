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

CROP_DISTANCE = {
    frozenset(("CORN", "SOYBEANS")): 0.950,
    frozenset(("CORN", "WHEAT")): 0.948,
    frozenset(("CORN", "BARLEY")): 0.768,
    frozenset(("BARLEY", "SOYBEANS")): 0.232,
    frozenset(("BARLEY", "WHEAT")): 0.230,
    frozenset(("SOYBEANS", "WHEAT")): 0.052,
}

def compute_local_y_variance(
    X,
    y,
    numeric_features,
    categorical_features,
    k = 10,
    crop_feature = "commodity_desc",
    numeric_weight = 0.8,
):
    """
    Compute local target variance using a smoothness-aware mixed-feature distance.

    Numeric features are min-max scaled and compared using mean absolute L1 distance.
    Unlike standard Gower distance, which treats categorical mismatches as binary 0/1
    penalties, the crop categorical feature uses a graded crop-distance matrix. This
    allows crop mismatches to contribute different distances depending on their
    empirical/domain-informed yield similarity.

    Other categorical features are still treated as standard Hamming mismatches unless
    a custom distance is defined for them.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.
        k: Number of nearest neighbors.
        crop_feature: Name of the crop/commodity categorical column.
        numeric_weight: Weight assigned to numeric distance. The remaining weight is
            assigned to categorical distance.

    Returns:
        local_vars: Local target variances indexed like X.
    """

    scaler = MinMaxScaler()

    X_num = scaler.fit_transform(X[numeric_features])
    X_cat = X[categorical_features].astype(str).values

    categorical_weight = 1.0 - numeric_weight

    def crop_distance(a, b):
        """
        Graded crop distance.

        Same crop has distance 0. Different crops use the predefined crop-distance
        matrix. Unknown crop pairs fall back to 1.0.
        """
        a = str(a).upper().strip()
        b = str(b).upper().strip()

        if a == b:
            return 0.0

        return CROP_DISTANCE.get(frozenset((a, b)), 1.0)

    def categorical_distance(col_name, a, b):
        """
        Feature-specific categorical distance.

        Crop/commodity uses the graded crop distance matrix.
        Other categorical features use standard 0/1 mismatch distance.
        """
        if col_name == crop_feature:
            return crop_distance(a, b)

        return float(str(a) != str(b))

    def mixed_distance(i, j):
        """
        Smoothness-aware mixed-feature distance between observations i and j.

        Numeric features contribute scaled L1 distance. Categorical features contribute
        the average feature-specific categorical distance. For crop identity, this is
        a graded distance rather than a binary mismatch.
        """
        num_dist = np.nanmean(np.abs(X_num[i] - X_num[j]))

        cat_dists = []

        for cat_idx, col_name in enumerate(categorical_features):
            a = X_cat[i, cat_idx]
            b = X_cat[j, cat_idx]

            cat_dists.append(
                categorical_distance(col_name, a, b)
            )

        cat_dist = np.mean(cat_dists) if cat_dists else 0.0
        mixed_dist = numeric_weight * num_dist + categorical_weight * cat_dist

        return mixed_dist

    n = len(X)
    local_vars = []

    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue

            dists.append((mixed_distance(i, j), j))

        dists.sort(key=lambda x: x[0])
        neighbors = [j for d, j in dists if d > 0][:k]
        local_vars.append(np.var(y.iloc[neighbors], ddof=1))

    local_vars = pd.Series(local_vars, index=X.index)

    return local_vars
