from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values based on the median of a grouping column
    Falls back to global median if the group has no valid values
    """
    def __init__(self, group_col, impute_col):
        self.group_col = group_col 
        self.impute_col = impute_col
        self.group_medians = {}
        self.global_median = None

    def fit(self, X, y = None): 
        """
        Calculates group-specific and global medians for the impute column
        """
        logger.info(f"Fitting GroupedMedianImputer for '{self.impute_col}' grouped by '{self.group_col}'")
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.group_medians_= df.groupby(self.group_col)[self.impute_col].median().to_dict()
        self.global_median_ = df[self.impute_col].median() 

        logger.debug(f"Computed group medians for {len(self.group_medians_)} groups")
        logger.debug(f"Global median for '{self.impute_col}'  = {self.global_median_}")
        return self

    def transform(self, X):
        """
        Replaces missing values in impute_col using group-specific or global medians
        """
        logger.info(f"transforming '{self.impute_col}' using GroupedMedianImputer")
        df = pd.DataFrame(X).copy() 
        df[self.impute_col] = df[self.impute_col].fillna( 
            df[self.group_col].map(self.group_medians_).fillna(self.global_median_) 
        )

        logger.debug(f"Filled missing continuous values in '{self.impute_col}")
        return df
    

class GroupedMostFrequentImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing categorical values based on the mode of a grouping column
    Falls back to global mode if the group has no valid values
    """
    def __init__(self, group_col, impute_col):
        self.group_col = group_col
        self.impute_col = impute_col
        self.group_modes = {}
        self.global_mode = None

    def fit(self, X, y = None):
        """
        Calculates group-specific and global modes for the impute column
        """
        logger.info(f"Fitting GroupedMostFrequentImputer for '{self.impute_col}' grouped by '{self.group_col}'")
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.group_modes_ = df.groupby(self.group_col)[self.impute_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
        self.global_mode_ = df[self.impute_col].mode().iloc[0]

        logger.debug(f"Computed group modes for {len(self.group_modes_)} groups")
        logger.debug(f"Global mode for '{self.impute_col}'  = {self.global_mode_}")
        return self

    def transform(self, X):
        """
        Replaces missing values in impute_col using group-specifc or global modes
        """
        logger.info(f"transforming '{self.impute_col}' using GroupedMostFrequentImputer")
        df = pd.DataFrame(X).copy()
        df[self.impute_col] = df[self.impute_col].fillna(df[self.group_col].map(self.group_modes_).fillna(self.global_mode_)
        )
        
        logger.debug(f"Filled missing categorical values in '{self.impute_col}")
        return df


class GroupwiseImputer(BaseEstimator, TransformerMixin):
    """
    Wrapper that applies grouped imputers to multiple columns.
    Automatically selects GroupedMedianImputer for numeric col
    and GroupedMostFrequentImputer for categorical col
    """
    def __init__(self, group_col, numeric_cols = None, categorical_cols = None):
        self.group_col = group_col
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self._num_imputers = {} 
        self._cat_imputers = {}

    def fit(self, X, y = None):
        """
        Gets groupwise imputers for each specified column
        """
        logger.info("Fitting GroupwiseImputer")
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
        logger.info(f"Fitted {len(self._num_imputers)} numeric and {len(self._cat_imputers)} categorical imputers")
        return self

    def transform(self, X):
        """
        Applies fitted imputers to fill missing values in numeric and categorical columns
        """
        logger.info("Applying GroupwiseImputer transformations")
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Apply numeric imputations
        for col, imp in self._num_imputers.items():
            df = imp.transform(df)

        # Apply categorical imputations
        for col, imp in self._cat_imputers.items():
            df = imp.transform(df)
        return df