"""
derive_features.py
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DerivedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that derives additional features from raw inputs.

    This transformer augments the input feature set with domain-informed numeric and 
    categorial features derived from raw data. No parameters are learned during fitting, 
    all transformations are deterministic.

    Inherits from:
        Base Estimator: 
            Provides scikit-learn parameter management, enabling compatibility with
            model selection utilities such as GridSearchCV.
        TransformerMixin:
            Supplies the fit-transformer interface required for integration into scikit-learn
            pipeline.
   
    Derived features:
        Numeric:
            - soil_quality_score
            - temp_optimality
            - ca_mg_ratio
        Categorical: 
            - gdd_suitability
    """

    def __init__(self):
        self.new_numeric_ = ['soil_quality_score', 'temp_optimality', 'ca_mg_ratio']
        self.new_categorical_ = ['gdd_suitability']

    def fit(self, X, y = None):
        """
        Fit method is implemented for compatibility with scikit-learn pipelines.
        No state is learned during fitting; all derived features are computed deterministically
        from the input data.
        """
        return self

    def transform(self, X):
        """
        Add derived numeric and categorical features to the input DataFrame.

        This method computes features capturing soil quality, temperature optimality, 
        nutrient balance, and crop-specific growing degree day suitability. Original features are 
        preserved and new columns are appended to the dataframe.

        Args:
            Training DataFrame
        
        Returns: Transformed DataFrame with new features.
        """
        try:
            X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

            ph_vals = X['ph_h2o']
            oc_vals = X['estimated_organic_carbon']
            cec_vals = X['cec_nh4_ph_7']
            clay_vals = X['clay_total']
            avg_temps = X['avg_temperature']
            precip = X['total_precipitation']
            ca_vals = X['ca_nh4_ph_7']
            mg_vals = X['mg_nh4_ph_7']
            gdd = X['growing_degree_days']

            X['soil_quality_score'] = ph_vals + cec_vals + clay_vals
            X['temp_optimality'] = np.maximum(0, 1 - np.abs(avg_temps - 22.0) / 22.0)
            X['ca_mg_ratio'] = np.where(mg_vals > 0, ca_vals / mg_vals, 5.0)
            commodity = X.get('commodity_desc', pd.Series('CORN', index = X.index))
            commodity = commodity.fillna('CORN')
            X['commodity_desc'] = commodity

            gdd_ranges = {
                'CORN' : (1000, 2000),
                'SOYBEANS' : (1200, 1800),
                'WHEAT' : (800, 1400),
                'COTTON' : (1400, 2000),
                'BARLEY' : (900, 1300)
            }

            def check_gdd_suitability(row):
                """
                Determines crop-specific growing degree day suitability.

                Compares observed growing degree days for a sample against predefined optimal
                ranges for each supported commodity.

                Args: 
                    row: Single observation including commodity type

                Returns:
                    'Optimal'/'Suboptimal': String description of commodity desc and gdd_ranges 
                    compatability
                """
                crop = row['commodity_desc']
                if crop in gdd_ranges:
                    low, high = gdd_ranges[crop]
                    return 'Optimal' if low <= row['growing_degree_days'] <= high else 'Suboptimal'
                return 'Suboptimal'

            X['gdd_suitability'] = X.apply(check_gdd_suitability, axis = 1)

            return X        

        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")
            logger.error("Returning input DataFrame without modifications")
            return X