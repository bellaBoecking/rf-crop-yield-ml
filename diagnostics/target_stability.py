"""
target_similarity.py

Local target instability diagnostics with mixed-feature neighborhoods.

This module computes a local variance measure of the target variable using a
custom smoothness-aware mixed numeric-categorical distance.

Numeric features are MinMax-scaled and compared using mean absolute L1 distance,
so small numeric changes produce small changes in distance. Unlike standard
Gower distance, which treats nominal categorical mismatches as binary 0/1
penalties, the crop categorical feature uses a graded crop yield-dissimilarity
penalty. This avoids treating all crop mismatches as equally distant.

Other categorical features use standard Gower-style nominal mismatch distance:
0 for a match and 1 for a mismatch.

These diagnostics quantify local target instability under a distance designed
for exploratory smoothness-aware analysis. They are intended for diagnostic
evaluation rather than model training.
"""

from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CROP_YIELD_DISSIMILARITY = {
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
    k=10,
    crop_feature="commodity_desc",
    numeric_weight=None,
):
    """
    Compute local target variance using a smoothness-aware mixed-feature distance.

    Numeric features are MinMax-scaled and compared using mean absolute L1
    distance. The crop categorical feature uses a graded crop yield-dissimilarity
    penalty instead of a binary Gower-style mismatch. Other categorical features
    use standard Gower-style nominal mismatch distance.

    Args:
        X: Feature DataFrame.
        y: Target Series or array-like target values aligned with X.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.
        k: Number of nearest neighbors.
        crop_feature: Name of the crop/commodity categorical column.
        numeric_weight: Weight assigned to numeric distance. Must be between
            0 and 1. If None, defaults to the fraction of features that are numeric.

    Returns:
        local_vars: Local target variances indexed like X.
    """

    n_numeric = len(numeric_features)
    n_categorical = len(categorical_features)
    total_features = n_numeric + n_categorical

    if total_features == 0:
        raise ValueError("At least one numeric or categorical feature is required.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    missing_numeric = [col for col in numeric_features if col not in X.columns]
    missing_categorical = [col for col in categorical_features if col not in X.columns]

    if missing_numeric:
        raise ValueError(f"Missing numeric columns in X: {missing_numeric}")

    if missing_categorical:
        raise ValueError(f"Missing categorical columns in X: {missing_categorical}")

    if numeric_weight is None:
        numeric_weight = n_numeric / total_features

    if not np.isfinite(numeric_weight):
        raise ValueError("numeric_weight must be finite.")

    if not 0 <= numeric_weight <= 1:
        raise ValueError("numeric_weight must be between 0 and 1.")

    categorical_weight = 1.0 - numeric_weight

    if n_categorical > 0 and crop_feature not in categorical_features:
        logger.warning(
            "%s is not in categorical_features, so crop yield-dissimilarity "
            "penalties will not be used.",
            crop_feature,
        )

    if n_numeric > 0:
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(X[numeric_features])
    else:
        X_num = None

    if n_categorical > 0:
        X_cat = X[categorical_features].fillna("MISSING").astype(str).values
    else:
        X_cat = None

    y_values = np.asarray(y)

    def crop_yield_dissimilarity(a, b):
        """
        Return graded crop yield-dissimilarity penalty.

        Same crop has penalty 0. Different crops use the predefined crop
        yield-dissimilarity matrix. Unknown crop pairs fall back to 1.0.
        """
        a = str(a).upper().strip()
        b = str(b).upper().strip()

        if a == b:
            return 0.0

        return CROP_YIELD_DISSIMILARITY.get(frozenset((a, b)), 1.0)

    def categorical_distance(col_name, a, b):
        """
        Return feature-specific categorical distance.

        The crop feature uses the graded crop yield-dissimilarity penalty.
        Other categorical features use standard Gower-style nominal mismatch:
        0 for same category, 1 for different category.
        """
        if col_name == crop_feature:
            return crop_yield_dissimilarity(a, b)

        return float(str(a).strip() != str(b).strip())

    def mixed_distance(i, j):
        """
        Compute smoothness-aware mixed distance between observations i and j.

        Numeric component:
            Mean absolute difference across MinMax-scaled numeric features.

        Categorical component:
            Mean categorical distance across categorical features. Crop uses
            graded yield-dissimilarity; other categoricals use 0/1 mismatch.
        """
        if n_numeric > 0:
            num_dist = np.nanmean(np.abs(X_num[i] - X_num[j]))
        else:
            num_dist = 0.0

        if n_categorical > 0:
            cat_dists = []

            for cat_idx, col_name in enumerate(categorical_features):
                a = X_cat[i, cat_idx]
                b = X_cat[j, cat_idx]
                cat_dists.append(categorical_distance(col_name, a, b))

            cat_dist = np.mean(cat_dists)
        else:
            cat_dist = 0.0

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
        neighbors = [j for _, j in dists][:k]

        if len(neighbors) >= 2:
            local_vars.append(np.var(y_values[neighbors], ddof=1))
        else:
            local_vars.append(np.nan)

    local_vars = pd.Series(local_vars, index=X.index)

    return local_vars
