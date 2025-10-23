# Custom transformer to derive additional features for the crop yield model
# Transformer follows the sklearn pattern: fit + transform

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DerivedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for deriving new features:
    - Numeric features: soil_quality_score, temp_optimality, ca_mg_ratio
    - Categorical featrue: gdd_suitability
    """

    def __init__(self):
        # Names of new features
        self.new_numeric_ = ['soil_quality_score', 'temp_optimality', 'ca_mg_ratio']
        self.new_categorical_ = ['gdd_suitability']

    def fit(self, X, y = None):
        """
        Fit method is required for sklearn pipeline compatability
        No fitting needed for this transformer
        """
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by adding derived features
        """
        try:
            X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            logger.info(f"Starting feature derivation for {len(X)} records")

            ph_vals = X['ph_h2o']
            oc_vals = X['estimated_organic_carbon']
            cec_vals = X['cec_nh4_ph_7']
            clay_vals = X['clay_total']
            avg_temps = X['avg_temperature']
            precip = X['total_precipitation']
            ca_vals = X['ca_nh4_ph_7']
            mg_vals = X['mg_nh4_ph_7']
            gdd = X['growing_degree_days']

            # Deriving Numeric Features
            X['soil_quality_score'] = ph_vals + cec_vals + clay_vals
            X['temp_optimality'] = np.maximum(0, 1 - np.abs(avg_temps - 22.0) / 22.0)
            X['ca_mg_ratio'] = np.where(mg_vals > 0, ca_vals / mg_vals, 5.0)
            commodity = X.get('commodity_desc', pd.Series('CORN', index = X.index))
            commodity = commodity.fillna('CORN')
            X['commodity_desc'] = commodity

            # Derive Categorical Features
            # Growing degree day suitability per crop
            gdd_ranges = {
                'CORN' : (1000, 2000),
                'SOYBEANS' : (1200, 1800),
                'WHEAT' : (800, 1400),
                'COTTON' : (1400, 2000),
                'BARLEY' : (900, 1300)
            }

            def check_gdd_suitability(row):
                crop = row['commodity_desc']
                if crop in gdd_ranges:
                    low, high = gdd_ranges[crop]
                    return 'Optimal' if low <= row['growing_degree_days'] <= high else 'Suboptimal'
                return 'Suboptimal'

            X['gdd_suitability'] = X.apply(check_gdd_suitability, axis = 1) # applying among columns

            logger.info("Feature derivation completed successfully")
            return X        

        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")
            logger.error("returning input DataFrame without modifications")
            return X