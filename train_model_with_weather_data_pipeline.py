#!/usr/bin/env python3


"""
Crop Yield Prediction Pipeline
------------------------------
This script connects to Supabase to retrieve weather, soil chemical, and physical 
data, matches them with historical crop yields, and trains a Random Forest 
regressor using a preprocessing pipeline with imputation and derived features.
"""

# Imports
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
import json
from datetime import datetime
from impute import GroupedMedianImputer, GroupedMostFrequentImputer, GroupwiseImputer
from derive_features import DerivedFeaturesTransformer
import warnings

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
        logger.info("Initializing WeatherBasedModelTrainer")
        load_dotenv()
        # Connect to supabase using credentials stored in .env
        self.supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None

        # Define numeric and categorical feature lists
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

    # Data Fetching
    def get_weather_and_soil_data(self):
        """
        Fetches weather, soil chemical, and soil physical property data from Supabase
        and merges them into a singlee dataframe.
        """

        try:
            # Get all weather records
            logger.info("Fetching Weather Data From Supabase")
            weather_response = self.supabase.table('weather_soil_samples').select('*').execute()
            weather_df = pd.DataFrame(weather_response.data)
            logger.info(f"Found {len(weather_df)} weather records")
            
            # Get soil sample numbers for joining
            sample_nums = weather_df['pedlabsampnum'].tolist()
            logger.info(f"Preparing {len(sample_nums)} sample numbers for merging soil properties")
            
            # Get chemical properties and select relevant columns
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
            
            # Get physical properties
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
            
            # Merge DataFrames
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


    # Crop Matching
    def match_with_crop_yields(self, enhanced_df):
        """
        Matcches soil-weather records with historical crop records from Supabase using 
        approximate spatial (state boundaries) and temporal matching.
        """

        logger.info("Matching with crop yields using expanded coverage...")
        
        try:
            # Get crop yield data
            crop_response = self.supabase.table('nass_crops').select(
                'commodity_desc, year, value, state_name, county_name, unit_desc'
            ).eq('statisticcat_desc', 'YIELD').in_(
                'commodity_desc', ['CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'BARLEY']
            ).gte('year', 1948).lte('year', 2025).execute()
            
            if not crop_response.data:
                logger.error("No crop data found")
                return pd.DataFrame()

            # Convert to DataFrame
            if hasattr(crop_response, 'data') and crop_response.data:
                crop_df = pd.DataFrame(crop_response.data)
            else:
                crop_df = pd.DataFrame()
            logger.info(f"Retrieved {len(crop_df)} crop yield records")
            
            # Clean crop yields
            crop_df['yield_value'] = pd.to_numeric(
                crop_df['value'].astype(str).str.replace(',', '').str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce'
            )
            
            crop_df = crop_df[
                (crop_df['yield_value'].notna()) & 
                (crop_df['yield_value'] > 0) & 
                (crop_df['yield_value'] < 1000)
            ].copy()
            logger.info(f"Filtered crop yields, {len(crop_df)} valid records remaining")
            
            # Defining state boundaries
            state_boundaries = {
                'ALABAMA': {'lat_range': (30.2, 35.0), 'lon_range': (-88.5, -84.9)},
                'ARIZONA': {'lat_range': (31.3, 37.0), 'lon_range': (-114.8, -109.0)},
                'ARKANSAS': {'lat_range': (33.0, 36.5), 'lon_range': (-94.6, -89.6)},
                'CALIFORNIA': {'lat_range': (32.5, 42.0), 'lon_range': (-124.4, -114.1)},
                'COLORADO': {'lat_range': (37.0, 41.0), 'lon_range': (-109.1, -102.0)},
                'FLORIDA': {'lat_range': (24.5, 31.0), 'lon_range': (-87.6, -80.0)},
                'GEORGIA': {'lat_range': (30.4, 35.0), 'lon_range': (-85.6, -80.8)},
                'IDAHO': {'lat_range': (42.0, 49.0), 'lon_range': (-117.2, -111.0)},
                'ILLINOIS': {'lat_range': (37.0, 42.5), 'lon_range': (-91.5, -87.0)},
                'INDIANA': {'lat_range': (37.8, 41.8), 'lon_range': (-88.1, -84.8)},
                'IOWA': {'lat_range': (40.4, 43.5), 'lon_range': (-96.6, -90.1)},
                'KANSAS': {'lat_range': (37.0, 40.0), 'lon_range': (-102.1, -94.6)},
                'KENTUCKY': {'lat_range': (36.5, 39.1), 'lon_range': (-89.6, -81.9)},
                'LOUISIANA': {'lat_range': (28.9, 33.0), 'lon_range': (-94.0, -88.8)},
                'MICHIGAN': {'lat_range': (41.7, 48.2), 'lon_range': (-90.4, -82.4)},
                'MINNESOTA': {'lat_range': (43.5, 49.4), 'lon_range': (-97.2, -89.5)},
                'MISSOURI': {'lat_range': (36.0, 40.6), 'lon_range': (-95.8, -89.1)},
                'MONTANA': {'lat_range': (45.0, 49.0), 'lon_range': (-116.1, -104.0)},
                'NEBRASKA': {'lat_range': (40.0, 43.0), 'lon_range': (-104.1, -95.3)},
                'NORTH DAKOTA': {'lat_range': (45.9, 49.0), 'lon_range': (-104.1, -96.6)},
                'OHIO': {'lat_range': (38.4, 41.9), 'lon_range': (-84.8, -80.5)},
                'OKLAHOMA': {'lat_range': (33.6, 37.0), 'lon_range': (-103.0, -94.4)},
                'SOUTH DAKOTA': {'lat_range': (42.5, 45.9), 'lon_range': (-104.1, -96.4)},
                'TEXAS': {'lat_range': (25.8, 36.5), 'lon_range': (-106.6, -93.5)},
                'WISCONSIN': {'lat_range': (42.5, 47.1), 'lon_range': (-92.9, -86.8)}
            }
            
            matched_records = []
            # Normalizing state and county names
            # Only filtering by state here, as the data is sparce
            crop_df['state_name'] = crop_df['state_name'].str.upper().str.strip()
            crop_df['county_name'] = crop_df['county_name'].str.upper().str.strip()
            
            # Match each soil record with a corresponding yield record
            for _, soil_row in enhanced_df.iterrows():
                try:
                    lat = float(soil_row['sample_latitude'])
                    lon = float(soil_row['sample_longitude'])
                    soil_year = soil_row['sample_year']
                    
                    # Find matching state
                    soil_state = None
                    for state, bounds in state_boundaries.items():
                        lat_min, lat_max = bounds['lat_range']
                        lon_min, lon_max = bounds['lon_range']
                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                            soil_state = state
                            break
                    
                    if not soil_state:
                        continue
                    
                    # Match crop yields within +- 1 year of soil sample
                    matching_crops = crop_df[
                        (crop_df['state_name'] == soil_state) &
                        (crop_df['year'].between(soil_year - 1, soil_year + 1))
                    ]

                    # Fallback: pick the closest year if none match directly
                    if matching_crops.empty:
                        state_crops = crop_df[crop_df['state_name'] == soil_state]
                        if not state_crops.empty:
                            closest_idx = (state_crops['year'] - soil_year).abs().idxmin()
                            matching_crops = state_crops.loc[[closest_idx]] # double brackets returns a df instead of series
                        
                    # Combine soil and crop data into samples
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
                return matched_df
            else:
                logger.warning("No matches found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error matching crop yields: {e}")
            return pd.DataFrame()
    

    # Pipeline Assembly
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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        pipeline = self.build_pipeline()

        param_grid = {
            "model__n_estimators" : [100, 200, 300],
            "model__max_depth" : [None, 10, 20],
            "model__min_samples_split" : [2, 5, 10],
            "model__min_samples_leaf" : [1, 2, 4]
        }

        logger.info("Starting grid search for hyperparameter tuning")
        grid = GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid, 
            cv = 5,
            n_jobs =-1,
            scoring = "r2",
            verbose = 2
        )

        grid.fit(X_train, y_train)
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
    """Executes full end-to-end training pipeline."""
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