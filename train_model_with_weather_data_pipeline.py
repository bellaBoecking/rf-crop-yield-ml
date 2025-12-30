#!/usr/bin/env python3


"""
Crop Yield Prediction Pipeline
------------------------------
Connects to Supabase to retrieve weather, soil chemical, and physical data, 
matches soil samples with historical crop yields using approximate spatial 
and temporal heuristics, and trains a Random Forest regressor using a 
scikit-learn pipeline with group-aware imputation and cross-validation.

Assumes:
- Soil samples are independent by pedlabsampnum
- State-level spatial matching is a coarse proxy (to be refined later)
"""

# NOTE (v1):
# Multiple yield rows may correspond to the same soil sample (pedlabsampnum)
# Observations are therefore not independent
# All splitting and CV is grouped by pedlabsampnum to prevent leakage
# Reported performance metrics should be interpreted as optimistic

import os
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    GroupKFold, 
    GroupShuffleSplit,
    GridSearchCV
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from impute import GroupedMedianImputer, GroupedMostFrequentImputer, GroupwiseImputer
from derive_features import DerivedFeaturesTransformer

warnings.filterwarnings("ignore", category = RuntimeWarning)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model Trainer Class
class WeatherBasedModelTrainer:
    """
    Retrieves data from Supabase, processes and merges it, and trains a Random Forest 
    regression model to predict crop yields based on environmental and soil factors.
    """

    def __init__(self):
        """
        Initializes Supabase connection, feature definitions, and
        model-related state.
        """
        logger.info("Initializing WeatherBasedModelTrainer")
        load_dotenv()
        self.supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None

        self.numeric_features = [
        'sample_year', 'sample_latitude', 'sample_longitude',
        'avg_temperature', 'max_temperature', 'min_temperature',
        'total_precipitation', 'avg_relative_humidity', 'growing_degree_days',
        'ph_h2o', 'ph_cacl2', 'estimated_organic_carbon',
        'total_carbon_ncs', 'total_nitrogen_ncs', 'carbon_to_nitrogen_ratio',
        'cec_nh4_ph_7', 'base_sat_nh4oac_ph_7', 'ca_nh4_ph_7',
        'mg_nh4_ph_7', 'k_nh4_ph_7', 'na_nh4_ph_7',
        'clay_total', 'silt_total', 'sand_total',
        'bulk_density_oven_dry', 'water_retention_15_bar', 'particle_density_less_than_2mm'
       ]
        self.categorical_features = [
        'climate_region', 'commodity_desc'
       ]

    def get_weather_and_soil_data(self):
        """
        Fetches weather, soil chemical, and soil physical property data from Supabase
        and merges them into a singlee dataframe.

        Notes:
        - Uses left joins to preserve weather records even if lab data is missing
        - Chemical and physical data tables may be sparse or incomplete
        """

        try:
            logger.info("Fetching Weather Data From Supabase")
            weather_response = self.supabase.table('weather_soil_samples').select('*').execute()
            weather_df = pd.DataFrame(weather_response.data)
            logger.info(f"Found {len(weather_df)} weather records")
            
            sample_nums = weather_df['pedlabsampnum'].tolist()
            logger.info(f"Preparing {len(sample_nums)} sample numbers for merging soil properties")
            
            logger.info("Fetching chemical soil properties")
            chem_response = self.supabase.table('ssurgo_lab_chemical_properties').select('*').execute()
            
            initial_chem_df = pd.DataFrame(chem_response.data) if chem_response.data else pd.DataFrame()
            if initial_chem_df.empty:
                logger.warning("No chemical property data found")
            else:
                logger.info(f"Retrieved {len(initial_chem_df)} chemical property records")
            selected_chem_columns = ['labsampnum', 'ph_h2o', 'ph_cacl2', 'estimated_organic_carbon', 
                'total_carbon_ncs', 'total_nitrogen_ncs', 'carbon_to_nitrogen_ratio', 
                'cec_nh4_ph_7', 'base_sat_nh4oac_ph_7', 'ca_nh4_ph_7', 
                'mg_nh4_ph_7', 'k_nh4_ph_7', 'na_nh4_ph_7']
            chem_df = initial_chem_df[selected_chem_columns]
            
            logger.info("Fetching physical soil properties")
            phys_response = self.supabase.table('ssurgo_lab_physical_properties').select('*').execute()

            selected_phys_columns = ['labsampnum', 'clay_total', 'silt_total', 'sand_total', 
                'bulk_density_oven_dry', 'water_retention_15_bar', 'particle_density_less_than_2mm']
            
            initial_phys_df = pd.DataFrame(phys_response.data) if phys_response.data else pd.DataFrame()
            if initial_phys_df.empty:
                logger.warning("No physical property data found")
            else:
                logger.info(f"Retrieved {len(initial_phys_df)} physical property records")
            phys_df = initial_phys_df[selected_phys_columns]
            
            if not chem_df.empty:
                merged_df = weather_df.merge(chem_df, left_on='pedlabsampnum', right_on='labsampnum', how='left')
                logger.info(f"Joined {len(chem_df)} chemical property records")
            
            if not phys_df.empty:
                merged_df = merged_df.merge(phys_df, left_on='pedlabsampnum', right_on='labsampnum', how='left', suffixes=('', '_phys'))
                logger.info(f"Joined {len(phys_df)} physical property records")
            
            merged_df = merged_df.drop(columns = ['labsampnum', 'labsampnum_phys'])
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error fetching weather or soil data: {e}")
            return pd.DataFrame()


    def match_with_crop_yields(self, enhanced_df):
        """
        Matches soil-weather records with historical crop records from Supabase using 
        approximate spatial (state boundaries) and temporal matching.
        """
        # TODO (v2):
        # - Replace bounding boxes with true state polygons
        # - Resolve overlapping state regions
        # - Reduce geographic misclassification near borders
        # - Prevent error amplification from misassigned states
        
        logger.info("Matching with crop yields using expanded coverage...")
        
        try:
            crop_response = self.supabase.table('nass_crops').select(
                'commodity_desc, year, value, state_name, county_name, unit_desc'
            ).eq('statisticcat_desc', 'YIELD').in_(
                'commodity_desc', ['CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'BARLEY']
            ).gte('year', 1948).lte('year', 2025).execute()
            
            if not crop_response.data:
                logger.error("No crop data found")
                return pd.DataFrame()

            if hasattr(crop_response, 'data') and crop_response.data:
                crop_df = pd.DataFrame(crop_response.data)
            else:
                crop_df = pd.DataFrame()
            logger.info(f"Retrieved {len(crop_df)} crop yield records")
            
            crop_df['yield_value'] = pd.to_numeric(
                crop_df['value'].astype(str).str.replace(',', '').str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce'
                )
            invalid_years = crop_df[crop_df['year'].isna()]
            logger.info(f"invalid years: {invalid_years}")
            crop_df['year'] = pd.to_numeric(crop_df['year'], errors = 'coerce')
            crop_df = crop_df[crop_df['year'].notna()].copy()
            
            crop_df = crop_df[
                (crop_df['yield_value'].notna()) & 
                (crop_df['yield_value'] > 0) & 
                (crop_df['yield_value'] < 1000)
            ].copy()
            logger.info(f"Filtered crop yields, {len(crop_df)} valid records remaining")
            
            state_boundaries = {
                'ALABAMA': {'lat_range': (30.1, 35.1), 'lon_range': (-88.6, -84.8)},
                'ALASKA': {'lat_range': (51.2, 71.5), 'lon_range': (-179.2, -129.9)},
                'ARIZONA': {'lat_range': (31.2, 37.1), 'lon_range': (-114.9, -108.9)},
                'ARKANSAS': {'lat_range': (32.8, 36.6), 'lon_range': (-94.7, -89.4)},
                'CALIFORNIA': {'lat_range': (32.4, 42.1), 'lon_range': (-124.6, -114.0)},
                'COLORADO': {'lat_range': (36.9, 41.1), 'lon_range': (-109.2, -101.9)},
                'CONNECTICUT': {'lat_range': (40.9, 42.1), 'lon_range': (-73.8, -71.7)},
                'DELAWARE': {'lat_range': (38.3, 39.9), 'lon_range': (-75.8, -74.9)},
                'FLORIDA': {'lat_range': (24.3, 31.1), 'lon_range': (-87.8, -79.8)},
                'GEORGIA': {'lat_range': (30.3, 35.1), 'lon_range': (-85.7, -80.7)},
                'HAWAII': {'lat_range': (18.8, 22.4), 'lon_range': (-160.5, -154.5)},
                'IDAHO': {'lat_range': (41.9, 49.1), 'lon_range': (-117.3, -110.8)},
                'ILLINOIS': {'lat_range': (36.9, 42.6), 'lon_range': (-91.6, -87.0)},
                'INDIANA': {'lat_range': (37.7, 41.9), 'lon_range': (-88.2, -84.7)},
                'IOWA': {'lat_range': (40.3, 43.6), 'lon_range': (-96.7, -90.0)},
                'KANSAS': {'lat_range': (36.9, 40.1), 'lon_range': (-102.2, -94.5)},
                'KENTUCKY': {'lat_range': (36.4, 39.2), 'lon_range': (-89.7, -81.8)},
                'LOUISIANA': {'lat_range': (28.8, 33.1), 'lon_range': (-94.1, -88.7)},
                'MAINE': {'lat_range': (42.9, 47.5), 'lon_range': (-71.2, -66.8)},
                'MARYLAND': {'lat_range': (37.8, 39.8), 'lon_range': (-79.6, -75.0)},
                'MASSACHUSETTS': {'lat_range': (41.2, 42.9), 'lon_range': (-73.6, -69.9)},
                'MICHIGAN': {'lat_range': (41.6, 48.3), 'lon_range': (-90.5, -82.3)},
                'MINNESOTA': {'lat_range': (43.4, 49.5), 'lon_range': (-97.3, -89.3)},
                'MISSISSIPPI': {'lat_range': (30.1, 35.1), 'lon_range': (-91.7, -88.0)},
                'MISSOURI': {'lat_range': (35.9, 40.7), 'lon_range': (-95.9, -89.0)},
                'MONTANA': {'lat_range': (44.3, 49.1), 'lon_range': (-116.2, -103.9)},
                'NEBRASKA': {'lat_range': (39.9, 43.1), 'lon_range': (-104.2, -95.2)},
                'NEVADA': {'lat_range': (35.0, 42.1), 'lon_range': (-120.1, -114.0)},
                'NEW HAMPSHIRE': {'lat_range': (42.7, 45.4), 'lon_range': (-72.6, -70.6)},
                'NEW JERSEY': {'lat_range': (38.8, 41.4), 'lon_range': (-75.6, -73.8)},
                'NEW MEXICO': {'lat_range': (31.2, 37.1), 'lon_range': (-109.2, -102.9)},
                'NEW YORK': {'lat_range': (40.4, 45.1), 'lon_range': (-79.9, -71.7)},
                'NORTH CAROLINA': {'lat_range': (33.7, 36.6), 'lon_range': (-84.4, -75.4)},
                'NORTH DAKOTA': {'lat_range': (45.8, 49.1), 'lon_range': (-104.2, -96.5)},
                'OHIO': {'lat_range': (38.3, 42.0), 'lon_range': (-84.9, -80.4)},
                'OKLAHOMA': {'lat_range': (33.5, 37.1), 'lon_range': (-103.1, -94.3)},
                'OREGON': {'lat_range': (41.9, 46.4), 'lon_range': (-124.7, -116.4)},
                'PENNSYLVANIA': {'lat_range': (39.7, 42.4), 'lon_range': (-80.6, -74.5)},
                'RHODE ISLAND': {'lat_range': (41.1, 42.0), 'lon_range': (-71.9, -71.1)},
                'SOUTH CAROLINA': {'lat_range': (32.0, 35.3), 'lon_range': (-83.6, -78.4)},
                'SOUTH DAKOTA': {'lat_range': (42.4, 46.0), 'lon_range': (-104.2, -96.3)},
                'TENNESSEE': {'lat_range': (34.9, 36.8), 'lon_range': (-90.4, -81.6)},
                'TEXAS': {'lat_range': (25.7, 36.6), 'lon_range': (-106.7, -93.3)},
                'UTAH': {'lat_range': (36.9, 42.1), 'lon_range': (-114.2, -109.0)},
                'VERMONT': {'lat_range': (42.7, 45.1), 'lon_range': (-73.5, -71.4)},
                'VIRGINIA': {'lat_range': (36.5, 39.5), 'lon_range': (-83.7, -75.2)},
                'WASHINGTON': {'lat_range': (45.5, 49.1), 'lon_range': (-124.9, -116.8)},
                'WEST VIRGINIA': {'lat_range': (37.1, 40.7), 'lon_range': (-82.8, -77.7)},
                'WISCONSIN': {'lat_range': (42.4, 47.3), 'lon_range': (-93.0, -86.7)},
                'WYOMING': {'lat_range': (40.9, 45.1), 'lon_range': (-111.1, -104.0)}
            }

            matched_records = []
            crop_df = crop_df[crop_df['state_name'].notna()].copy()

            dropped = crop_df['state_name'].isna().sum()
            logger.info(f"Dropped {dropped} crop rows with missing state_name")

            crop_df['state_name'] = crop_df['state_name'].str.upper().str.strip()
            crop_df['county_name'] = crop_df['county_name'].str.upper().str.strip()

            logger.info(
                f"Crop states available ({crop_df['state_name'].nunique()}): "
                f"{sorted(crop_df['state_name'].unique())[:10]} ..."
            )
            no_state_count = 0
            

            logger.info(f"Upper bound for enhanced iterrows: {len(enhanced_df)}")

            for _, soil_row in enhanced_df.iterrows():
                try:
                    lat = float(soil_row['sample_latitude'])
                    lon = float(soil_row['sample_longitude'])
                    soil_year = int(soil_row['sample_year'])

                    if soil_year is None:
                        continue
                    
                    soil_state = None
                    for state, bounds in state_boundaries.items():
                        lat_min, lat_max = bounds['lat_range']
                        lon_min, lon_max = bounds['lon_range']

                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                            soil_state = state
                            break
                    
                    if not soil_state:
                        no_state_count += 1
                        continue
                    
                    matching_crops = crop_df[
                        (crop_df['state_name'] == soil_state) &
                        (crop_df['year'] <= soil_year) &
                        (crop_df['year'] >= soil_year -1)
                    ]

                    if matching_crops.empty:
                        past_crops = crop_df[
                            (crop_df['state_name'] == soil_state)&
                            (crop_df['year'] <= soil_year)
                        ]
                        if not past_crops.empty:
                            closest_idx = (soil_year - past_crops['year']).abs().idxmin()
                            matching_crops = past_crops.loc[[closest_idx]]

                    # Potential problem here because the same soil sample may be matched with multiple rows
                    # Same soil/weather features but different commodities and different yields
                    # Technically not a leak but does cause dependency leakage across CV folds
                    # The same soil sample may appear in both training and validation
                    # Called sample duplication leackage, weakens the validity of cv estimates
                    for _, crop_row in matching_crops.iterrows():
                        matched_record = {
                            **soil_row.to_dict(),
                            'commodity_desc': crop_row['commodity_desc'],
                            'crop_year': crop_row['year'],
                            'yield_value': crop_row['yield_value'],
                            'crop_unit_desc': crop_row['unit_desc'],
                            'state_name' : soil_state,
                            'year_difference': crop_row['year'] - soil_year
                            }
                        matched_records.append(matched_record)
                    
                except Exception as e:
                    logger.warning(f"Error matching crops for sample {soil_row.get('pedlabsampnum')}: {e}")
                    continue
            
            if matched_records:
                matched_df = pd.DataFrame(matched_records)
                if 'state_name' not in self.categorical_features:
                    self.categorical_features.append('state_name')
                logger.info(f"Successfully matched {len(matched_df)} training records")
                logger.info(f"Soil samples with no inferred state: {no_state_count}")
                return matched_df
            else:
                logger.warning("No matches found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error matching crop yields: {e}")
            return pd.DataFrame()
    

    def build_pipeline(self):
        """
        Assembles a preprocessing + modeling pipeline with:
        - Groupwise imputation by state
        - Derived feature generation
        - Scaling and one-hot encoding
        - Random Forest regression model
        """
        logger.info("Building preprocessing and modeling pipeline")

        imputer = GroupwiseImputer( 
            group_col = 'state_name',
            numeric_cols = self.numeric_features,
            categorical_cols = self.categorical_features
        )

        derived = DerivedFeaturesTransformer()

        # Extend feature lists with newly derived features
        for col in derived.new_numeric_:
            if col not in self.numeric_features:
                self.numeric_features.append(col)
        for col in derived.new_categorical_:
            if col not in self.categorical_features:
                self.categorical_features.append(col)

        preprocessor = ColumnTransformer(
            transformers = [
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown = 'ignore'), self.categorical_features)
            ]
        )
    
        model = RandomForestRegressor(random_state = 42, n_jobs = -1)

        pipeline = Pipeline([
            ('imputer', imputer), 
            ('derived', derived),
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        logger.info("Pipeline successfully constructed")
        return pipeline
        

    # Training
    def train_with_grid_search(self, df):
        """
        Splits data into training and test sets, performs grid search for
        hyperparameter tuning, and evaluates model performance.
        """
        logger.info("Splitting data into train/test sets")
        X = df[self.numeric_features + self.categorical_features]
        y = df['yield_value']
        groups = df['pedlabsampnum']

        # Group-aware train-test split to prevent soil sample leakage
        gss = GroupShuffleSplit(test_size = 0.2, random_state = 42)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        pipeline = self.build_pipeline()

        param_grid = {
            "model__n_estimators" : [100, 200, 300],
            "model__max_depth" : [None, 10, 20],
            "model__min_samples_split" : [2, 5, 10],
            "model__min_samples_leaf" : [1, 2, 4]
        }

        logger.info("Starting grid search for hyperparameter tuning")

        cv = GroupKFold(n_splits = 5)
        grid = GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid, 
            cv = cv,
            n_jobs =-1,
            scoring = "r2",
            verbose = 1
        )

        grid.fit(
            X_train, 
            y_train, 
            groups = groups.iloc[train_idx]
        )

        logger.info(f"Best params: {grid.best_params_}")
        logger.info(f"Best CV Score (R^2): {grid.best_score_:.4}")

        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        logger.info(f"Holdout metrics -> RMSE: {rmse:.3f}, MAE: {mae: .3f}, R^2: {r2: .3f}")

        self.model = best_model
        return best_model
        

# Main Pipeline Runner
def main():
    """
    Executes full end-to-end training pipeline.
    
    Fails fast if any upstream data dependency is missing.
    """
    logger.info("Starting full training pipeline")
    trainer = WeatherBasedModelTrainer()
    
    try:
        logger.info("Starting weather-based model training...")
        
        # Step 1: Get weather and soil data
        enhanced_data = trainer.get_weather_and_soil_data()
        if enhanced_data.empty:
            logger.error("No enhanced data available, aborting")
            return
        
        # Step 2: Match with crop yields
        training_data = trainer.match_with_crop_yields(enhanced_data)
        if training_data.empty:
            logger.error("No training data created, aborting")
            return

        # step 3: run pipeline
        model = trainer.train_with_grid_search(df = training_data)
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()