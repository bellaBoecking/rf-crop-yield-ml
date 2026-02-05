"""
feature_similarity.py

Utilities for computing Gower-based similarity diagnostics in cross-validation.

This module implements a Gower distance function capable of handling mixed numeric
and categorical features, and uses it to measure nearest-neighbor similarity between 
training and validation folds.

The primary use case is diagnostic analysis: quantifying how similar validation samples 
are to their closest training samples under a given cross-validation scheme (e.g. GroupKFold).
High similarity may indicate limited siatributional shift between folds, while low similarity 
can signal extrapolation or generalization challenges.

These diagnostics are intended for model evaluation and data understanding, not for
direct use as a training objective.
"""

import pandas as pd
import numpy as np
from config.paths import DATA_DIR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

def gower_distance(row_i, row_j, numeric_features, categorical_features, ranges):
    """
    Computes Gower distance between two observations with mixed feature types.

    Numeric features are normalized by their observed range in the training data, while 
    categorical features contribute a binary mismatch indicator. Missing values are ignored 
    on a per-feature basis.

    Args:
        row_i : First observation
        row_j: Second observation
        numeric_features: Names of numeric features
        categorical_features: Names of categorical features
        ranges: Mapping from numeric feature name to its observed range (max - min)
        computed from training data

    Returns:
        Dist: Gower distance in [0, 1]. Returns 0.0 if no comparable features available.
    """
    dist_sum = 0.0
    weight_sum = 0.0

    for col in numeric_features:
        xi = row_i[col]
        xj = row_j[col]
        r = ranges[col]

        if pd.notna(xi) and pd.notna(xj) and r > 0:
            dist_sum += abs(xi - xj) / r
            weight_sum += 1
        
    for col in categorical_features:
        xi = row_i[col]
        xj = row_j[col]
        
        if pd.notna(xi) and pd.notna(xj):
            dist_sum += 0.0 if xi == xj else 1
            weight_sum += 1

    dist = dist_sum / weight_sum if weight_sum > 0 else 0.0

    return dist


def compute_gower_nn_similarity(train_df, test_df, numeric_features, categorical_features):
    """
    Computes average nearest-neighbor similarity from test data to training data using Gower distance.

    For each observation in the test set, the maximum similarity (1 - Gower distance) to any training 
    observation is computed. The final statistic is the mean of these nearest-neighbor similarities 
    across all test samples.

    Numeric feature ranges are computed exclusively from the training data to avoid information leakage.

    Args:
        train_df: Training feature DataFrame
        test_df: Validation feature DataFrame
        numeric_features: Names of numeric features
        categorical_features: Names of categorical features
    
    Returns:
        mean_similarities: Mean of calculated similarities
    """
    similarities = []

    ranges = {}
    for col in numeric_features:
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        ranges[col] = col_max - col_min

    # iterate over test rows
    for _, test_row in test_df.iterrows():
        row_sims = []

        for _, train_row in train_df.iterrows():
            similarity = 1 - gower_distance(test_row, train_row, numeric_features, categorical_features, ranges)
            row_sims.append(similarity)

        similarities.append(max(row_sims))
    
    mean_similarities = float(np.mean(similarities))
    
    return mean_similarities


def run_cv_similarity_diagnostics(cv, X, y, groups, numeric_features, categorical_features):
    """
    Run cross-validated Gower similarity diagnostics across folds.

    For each cross-validation split, this function computes the mean nearest-neighbor similarity
    from the validation fold to the corresponding training fold. The result provides a fold-level 
    diagnostic of how similar validation data is to the training data under the chosen cv strategy.

    Created to assess group-based cross-validation schemes where distributional shift between folds
    is a concern.

    Args:
        cv: CV splitter
        X: Feature DataFrame
        y: Target vector
        groups: Group labels used by cv splitter
        numeric_features: Numeric feature names
        categorical_features: categorical feature names

    Returns:
        pd.DataFrame(fold_stats): DataFrame of fold similarity stats
    """
    fold_stats = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        train_df = X.iloc[tr_idx]
        test_df = X.iloc[va_idx]

        sim = compute_gower_nn_similarity(
            train_df = train_df,
            test_df = test_df, 
            numeric_features = numeric_features, 
            categorical_features = categorical_features
        )

        fold_stats.append({
            "fold" : fold, 
            "mean_test_to_train_similarity" : sim
        })

    return pd.DataFrame(fold_stats)

