"""
impute.py

Custom state-aware imputers for scikit-learn pipelines.

This module defines transformers for handling missing data in both numeric and categorical
features, with imputation performed within groups (e.g., states) to account for local heterogeneity
in the data. These transformers are compateble withscikit learn pipelines and support cross-
validation without leaking information across folds.

Classes:
    GroupedMedianImputer: Imputes missing numeric values.
    GroupedMostFrequentImputer: Imputes missing categorical values.
    GroupwiseImputer: Composite transformer, applies grouped imputers to multiple columns.

Classes Inherit from:
    Base Estimator: 
        Provides scikit-learn parameter management, enabling compatibility with
        model selection utilities such as GridSearchCV.
    TransformerMixin:
        Supplies the fit-transformer interface required for integration into scikit-learn
        pipeline.

Key Features:
    - Group-aware: Imputes values within a specialized imputing column.
    - Fallbacks: Uses global statistics when group-level data is missing.
    - Pipeline-compatible: Inherits from BaseEstimator and TransformerMixin.
    - Safe for cross-validation: Does not leak information from holdout folds.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    """
    State-aware median imputer using median statistics.

    Imputes missing values in numeric columns by computing the median within each state
    (or other grouping variable). If a state contains no valid observations for the 
    imputed column, the global median computed over the entire training set is used as a 
    fallback.
    """

    def __init__(self, group_col, impute_col):
        self.group_col = group_col 
        self.impute_col = impute_col
        self.group_medians = {}
        self.global_median = None

    def fit(self, X, y = None): 
        """
        Computes state-level medians for the imputed column.

        Args:
            X: Feature DataFrame
            y: Target vector

        Returns: Fitted transformer with state-level and global medians.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.group_medians_= df.groupby(self.group_col)[self.impute_col].median().to_dict()
        self.global_median_ = df[self.impute_col].median() 

        return self

    def transform(self, X):
        """
        Imputes missing numeric values using state-level or global medians.

        Missing values are filled using the median corresponding to the observation's state.
        If no state-level median is available, global median is used.

        Returns: Transformed DataFrame with missing numeric values imputed.
        """
        df = pd.DataFrame(X).copy() 
        df[self.impute_col] = df[self.impute_col].fillna( 
            df[self.group_col].map(self.group_medians_).fillna(self.global_median_) 
        )

        return df
    

class GroupedMostFrequentImputer(BaseEstimator, TransformerMixin):
    """
    State-aware categorical imputer using model values.

    This transformer imputes missing categorical values by using the most frequent value (mode) 
    observed within each state. If a state has no valid observation, the global model is used.

    Compatible with  scikit-learn pipelines and cross-validation. 
    """
    def __init__(self, group_col, impute_col):
        self.group_col = group_col
        self.impute_col = impute_col
        self.group_modes = {}
        self.global_mode = None

    def fit(self, X, y = None):
        """
        Computes state-level and global modes for the imputed column.

        Args:
            X: Feature DataFrame
            y: Ignored, included for scikit-learn compatibility

        Returns: Fitted transformer with stored state-level and global modes.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.group_modes_ = df.groupby(self.group_col)[self.impute_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
        self.global_mode_ = df[self.impute_col].mode().iloc[0]

        return self

    def transform(self, X):
        """
        Imputes missing categorical values using state-level or global modes. 

        Missing values are replaced with the most frequent category observed within the same
        state. If no state-level mode exists, the global mode is applied.

        Args:
            X: Feature DataFrame
        
        Returns: Transformed DataFrame with missing categorical values imputed.
        """
        df = pd.DataFrame(X).copy()
        df[self.impute_col] = df[self.impute_col].fillna(df[self.group_col].map(self.group_modes_).fillna(self.global_mode_))
        
        return df


class GroupwiseImputer(BaseEstimator, TransformerMixin):
    """
    Custom state-aware composite imputer for mixed feature types. 

    Applied group-based imputation across multiple columns, using state-level medians for 
    numeric features and state-level modes for categorical features. All imputations
    fall back to global statistics when state-level information is unavailable.

    Acts as a lightweight wrapper around individual grouped imputers and is intended for
    use within scikit-learn pipelines.
    """
    def __init__(self, group_col, numeric_cols = None, categorical_cols = None):
        self.group_col = group_col
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self._num_imputers = {} 
        self._cat_imputers = {}

    def fit(self, X, y = None):
        """
        Fits state-aware imputers for all specified numeric and categorical columns.
        
        For each column, a corresponding state-based imputer is fitted using statistics
        computed within each state.

        Args:
            X: Feature DataFrame
            y: Ignored; Included for scikit-learn compatibility.
        
        Returns:
            self: Fitted GroupwiseImputer with column-level imputers.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        assert self.group_col in X.columns

        # Fit numeric imputers
        self._num_imputers = {
            col: GroupedMedianImputer(self.group_col, col).fit(df)
            for col in self.numeric_cols if col in df.columns 
        }
        # Fit categorical imputers
        self._cat_imputers = {
            col: GroupedMostFrequentImputer(self.group_col, col).fit(df)
            for col in self.categorical_cols if col in df.columns
        }

        return self

    def transform(self, X):
        """
        Applies state-aware imputations to numeric and categorical columns.

        Each fitted imputer fills missing values using state-level statistics, falling back
        to global values when necessary. 

        Args:
            X: Feature DataFrame

        Returns: Dataframe with state-wise imputation applied. 
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Apply numeric imputations
        for col, imp in self._num_imputers.items():
            df = imp.transform(df)

        # Apply categorical imputations
        for col, imp in self._cat_imputers.items():
            df = imp.transform(df)
        return df