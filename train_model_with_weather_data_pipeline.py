#!/usr/bin/env python3


"""
Crop Yield Prediction Pipeline
------------------------------
Connects to Supabase to retrieve weather, soil chemical, and physical data, 
matches soil samples with historical crop yields using approximate spatial 
and temporal heuristics, and trains a Random Forest regressor using a 
scikit-learn pipeline with group-aware imputation and cross-validation.

v1.1 Updates:
- Improved logging for crop matching and filtering invalid data
- Introduced use of GeoPandas for state polygon spatial joins
- Added dependency on Shapely for geometric operations (Point objects)

Assumes:
- Soil samples are independent by pedlabsampnum
- State-level spatial matching is a coarse proxy (to be refined later)
"""

# NOTE (v1.1):
# Multiple yield rows may correspond to the same soil sample (pedlabsampnum)
# Observations are therefore not independent
# All splitting and CV is grouped by pedlabsampnum to prevent leakage
# Reported performance metrics should be interpreted as optimistic

import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

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
import geopandas as gpd
from shapely.geometry import Point

from impute import GroupedMedianImputer, GroupedMostFrequentImputer, GroupwiseImputer
from derive_features import DerivedFeaturesTransformer

BASE_DIR = Path(__file__).resolve().parent
STATE_SHIP = BASE_DIR / "states" / "tl_2025_us_state.shp"
states = gpd.read_file(STATE_SHIP)

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
        and merges them into a single dataframe.

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


    def load_state_polygons(self):
        """
        Loads TIGER state polygons, normalizes state names, and converts data into a
        lat/lon format for soil sample matching. Cached after first load.
        """
        if hasattr(self, "_state_gdf"):
            return self._state_gdf
        logger.info("Loading TIGER state polygons")

        BASE_DIR = Path(__file__).resolve().parent
        STATE_SHP = BASE_DIR / "states" / "tl_2025_us_state.shp"
        states = gpd.read_file(STATE_SHP)

        states = states.to_crs(epsg = 4326) 

        states['state_name'] = states["NAME"].str.upper().str.strip()
        self._state_gdf = states[["state_name", "geometry"]]
        return self._state_gdf


    def assign_states_to_soil_samples(self, df):
        """
        Assigns state_name to soil sample using TIGER state polygons.
        """
        logger.info("Assigning states using TIGER polygons")
        
        states = self.load_state_polygons()

        soil_gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry = gpd.points_from_xy(
                df['sample_longitude'],
                df['sample_latitude']
            ),
            crs = "EPSG:4326" 
        )

        joined = gpd.sjoin(
            soil_gdf, 
            states, 
            how = 'left', 
            predicate = 'within'
        )

        missing = joined["state_name"].isna().sum()
        logger.info(f"Soil samples with no matched state {missing}")

        joined = joined.drop(columns = ['geometry', 'index_right'])

        return pd.DataFrame(joined)

        

    def match_with_crop_yields(self, enhanced_df):
        """
        Matches soil-weather records with historical crop records from Supabase using 
        approximate spatial (state boundaries) and temporal matching.
        """
        
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
                logger.info(crop_df.columns.to_list())
            else:
                crop_df = pd.DataFrame()
                logger.error("Crop responce does not exist!")
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
            
            matched_records = []
            dropped = crop_df['state_name'].isna().sum()
            logger.info(f"Dropped {dropped} crop rows with missing state_name")
            
            crop_df = crop_df[crop_df['state_name'].notna()].copy()

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
                    soil_state = soil_row['state_name']
                    soil_year = soil_row['sample_year']
                    
                    max_lookback = 10
                    matching_crops = crop_df[
                        (crop_df['state_name'] == soil_state) &
                        (crop_df['year'] <= soil_year) &
                        (crop_df['year'] >= soil_year - max_lookback)
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
                matched_df = matched_df.drop(columns = ['year_difference']) 

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
        
        # Step 2: Assign States
        enhanced_data = trainer.assign_states_to_soil_samples(enhanced_data)
        if enhanced_data.empty:
            logger.error("No state-assigned data available, aborting")
            return
        
        # Step 3: Match with crop yeilds
        training_data = trainer.match_with_crop_yields(enhanced_data)
        if training_data.empty:
            logger.error("No training data created, aborting")
            return

        # step 3: Run pipeline
        model = trainer.train_with_grid_search(df = training_data)
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()