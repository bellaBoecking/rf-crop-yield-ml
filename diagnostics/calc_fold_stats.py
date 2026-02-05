""" 
calc_fold_stats.py

Compute fold-level regression metrics (RMSE, MAE, R^2, mornalized target variance) for cross-validation, 
including group-aware splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from sklearn.base import clone

logger = logging.getLogger(__name__)

def fold_stats(cv, X_train, y_train, groups, best_model):
    """
    Computes metrics for each fold of a cross-validation split.

    Args:
        cv: CV splitter
        X_train: Training features
        y_train: Training targets
        groups: Group labels for group-aware CV
        best_model: Estimator to clone fit on each fold

    Returns:
        fold_df: Contains columns:
            - fold: fold_index
            - rmse: root mean squared error
            - r2: R^2 score
            - y_var_norm var(y_val) / var(y_train)
            - y_range: Target range in fold
            - val_idx: Validation indicies
    """
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups)):
        model = clone(best_model)

        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        pd.set_option('display.max_columns', None) 
        pd.set_option('display.max_rows', None) 

        fold_metrics.append({
            "fold": fold,
            "rmse": np.sqrt(mean_squared_error(y_va, preds)),
            "mae": mean_absolute_error(y_va, preds),
            "r2": r2_score(y_va, preds),
            "y_var_norm": np.var(y_va, ddof = 1) / np.var(y_train, ddof = 1),
            "y_range": y_va.max() - y_va.min(),
            "val_idx": va_idx
        })

    fold_df = pd.DataFrame(fold_metrics)

    return fold_df

