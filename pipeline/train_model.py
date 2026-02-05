#!/usr/bin/env python3

"""
train_model.py

Crop Yield Prediction Pipeline
------------------------------
Project connects to Supabase to retrieve weather, soil chemical, and physical data, and assigns geographic 
state information to soil samples using TIGER polygons. It then matches soil samples with historical
crop yields using approximate spatial and temporal heuristics and trains a Random Forest 
regressor using a scikit-learn pipeline with group-aware imputation, feature derivation, one-hot
encoding, and cross-validation.

Diagnostics Overview:
- Local target variance (Var(Y | X ~ x)) for identifying unstable neighborhoods.
- Fold-level performance vs. high-variance regions.
- Train-test similarity using mixed-feature distances.

v2.0 Updates:
- First implementation of diagnostics computations for local target variance.
- Logging of holdout variance and high-variance fraction.
- Correlation analysis between high-variance regions and R^2.

Assumes:
- Soil samples are treated as independent units, though regions of high feature similarity 
may still exhibit elevated local target variance.
- State-level matching is a coarse proxy to be refined in future versions.
- Diagnostics rely on mixed numeric + categorical distance metrics and are task-dependent.
"""

import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from config.paths import DATA_DIR, PIPELINE_DIR, DIAGNOSTICS_DIR, STATE_SHP

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
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

from .impute import GroupedMedianImputer, GroupedMostFrequentImputer, GroupwiseImputer
from .derive_features import DerivedFeaturesTransformer

# -- Project Imports --
from diagnostics.feature_similarity import run_cv_similarity_diagnostics, compute_gower_nn_similarity
from diagnostics.target_stability import compute_local_y_variance
from diagnostics.calc_fold_stats import fold_stats

warnings.filterwarnings("ignore", category = RuntimeWarning)

# Logging Setup
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force = True
)

for noisy in [
    "httpx",
    "httpcore",
    "httpcore.http2",
    "h2",
    "hpack",
    "urllib3",
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Model Trainer Class
class WeatherBasedModelTrainer:
    """
    Executes full end-to-end training pipeline for crop yield prediction using weather, 
    soil, and geographic data.

    This class is responsible for:
    - Retrieving weather and soil laboratory data from Supabase
    - Enriching samples with geographic state information via TIGER polygons
    - Merging environmental features with historical crop yield outcomes
    - Constructing a preprocessing and modeling pipeline
    - Training a Random Forest model with hyperparameter tuning
    - Running diagnostic analyses to assess model stability and performance

    The design emphasizes data integrity (through left joins and validation), reproducible
    preprocessing, and interpretability of performance variation across folds.

    Intended to be used as a single orchestration object for data loading, 
    feature engineering, model training, and evaluation. Fails fast if any upstream data dependency is missing.
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
        Retrieves weather data and associated soil chemical and physical properties from
        Supabase and merges them into a single DataFrame.

        Weather records form the base table and are left-joined with laboratory chemical and
        physical soil property tables using soil sample identifiers. This prioritises all weather
        observations.

        Returns:
            merged_df: Merged DataFrame indexed by soil sample.
        """
        try:
            logger.debug("Fetching Weather Data From Supabase")
            weather_response = self.supabase.table('weather_soil_samples').select('*').execute()
            weather_df = pd.DataFrame(weather_response.data)
            logger.debug(f"Found {len(weather_df)} weather records")
            
            sample_nums = weather_df['pedlabsampnum'].tolist()
            logger.debug(f"Preparing {len(sample_nums)} sample numbers for merging soil properties")
            
            logger.debug("Fetching chemical soil properties")
            chem_response = self.supabase.table('ssurgo_lab_chemical_properties').select('*').execute()
            
            initial_chem_df = pd.DataFrame(chem_response.data) if chem_response.data else pd.DataFrame()
            if initial_chem_df.empty:
                logger.warning("No chemical property data found")
            else:
                logger.debug(f"Retrieved {len(initial_chem_df)} chemical property records")
            selected_chem_columns = ['labsampnum', 'ph_h2o', 'ph_cacl2', 'estimated_organic_carbon', 
                'total_carbon_ncs', 'total_nitrogen_ncs', 'carbon_to_nitrogen_ratio', 
                'cec_nh4_ph_7', 'base_sat_nh4oac_ph_7', 'ca_nh4_ph_7', 
                'mg_nh4_ph_7', 'k_nh4_ph_7', 'na_nh4_ph_7']
            chem_df = initial_chem_df[selected_chem_columns]
            
            logger.debug("Fetching physical soil properties")
            phys_response = self.supabase.table('ssurgo_lab_physical_properties').select('*').execute()

            selected_phys_columns = ['labsampnum', 'clay_total', 'silt_total', 'sand_total', 
                'bulk_density_oven_dry', 'water_retention_15_bar', 'particle_density_less_than_2mm']
            
            initial_phys_df = pd.DataFrame(phys_response.data) if phys_response.data else pd.DataFrame()
            if initial_phys_df.empty:
                logger.warning("No physical property data found")
            else:
                logger.debug(f"Retrieved {len(initial_phys_df)} physical property records")
            phys_df = initial_phys_df[selected_phys_columns]
            
            if not chem_df.empty:
                merged_df = weather_df.merge(chem_df, left_on='pedlabsampnum', right_on='labsampnum', how='left')
                logger.debug(f"Joined {len(chem_df)} chemical property records")
            
            if not phys_df.empty:
                merged_df = merged_df.merge(phys_df, left_on='pedlabsampnum', right_on='labsampnum', how='left', suffixes=('', '_phys'))
                logger.debug(f"Joined {len(phys_df)} physical property records")
            
            merged_df = merged_df.drop(columns = ['labsampnum', 'labsampnum_phys'])
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error fetching weather or soil data: {e}")
            return pd.DataFrame()


    def load_state_polygons(self):
        """
        Loads and caches U.S. state boundary polygons from TIGER shapefiles.

        The shapefile is reprojected to WGS84 (EPSG: 4326) to ensure compatibility
        with latitude/longitude soil sample coordinates. State names are normalized to 
        uppercase for consistent downstream overhead.

        Returns:
            self._state_gdf: GeoPandas DataFrame with columns:
                - state_name: normalized state name
                - geometry: polygon geometry in EPSG: 4326
        """
        if hasattr(self, "_state_gdf"):
            return self._state_gdf
        logger.debug("Loading TIGER state polygons")

        states = gpd.read_file(STATE_SHP)
        states = states.to_crs(epsg = 4326) 

        states['state_name'] = states["NAME"].str.upper().str.strip()
        self._state_gdf = states[["state_name", "geometry"]]
        return self._state_gdf


    def assign_states_to_soil_samples(self, df):
        """
        Assigns U.S. state names to soil samples using spatial point-in-polygon matching.

        Soil sample coordinates (latitude/longitude) are converted into point geometries 
        and spatially joined against TIGER state boundary polygons. Each soil sample is
        assigned to the state polygon it falls within.

        Samples that do not fall within any state polygon are retained with a missing state_name, 
        and the count of unmatched samples is logged for diagnostics.

        Args:
            df: DataFrame containing soil samples

        Returns:
            pd.DataFrame(joined): Copy of the input DataFrame with an added 'state_name' column.
            Geometry and spatial join columns are removed.
        """
        logger.debug("Assigning states using TIGER polygons")
        
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
        logger.debug(f"Soil samples with no matched state: {missing}")

        joined = joined.drop(columns = ['geometry', 'index_right'])

        return pd.DataFrame(joined)

        

    def match_with_crop_yields(self, enhanced_df):
        """
        Matches soil-weather records in 'enhanced_df' with historical crop yield records from Supabase.
        Matching is performed using ap[proximate spatial (state boundaries) and temporal criteria, 
        with a lookback window to capture crop years when exact matches are not available.

        For each soil sample, multiple crop-year observations may be matched. The resulting matched
        records include both soil/weather features and crop information such as commodity type, 
        year, yield value, and unit. Missing or invalid crop records are filtered out.

        Matching logic: 
            - Only considers crops in ['CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'BARLEY']
            - Years between 1948 and 2025
            - Yield values must be numeric, positive, and < 1000
            - Primary match: same state and year <= sample tear within a 10-year lookback
            - If no crops found within lookback, uses closest prior year
            - Adds matched crop info to the soil sample, preserving state_name

        Args:
            enhanced_df: DataFrame containing soil rows constructed by merging Supabase tables

        Returns:
            pd.DataFrame: Matched soil-weather-crop records. Each row contains:
                - Original soil-weather features
                - 'commodity_desc' : Crop commodity
                - 'crop_year' : Observation year
                - 'yield_value' : Crop yield
                - 'crop_unit_desc' : Yield unit
                - 'state_name' : Uppercased state name
            Returns an empty DataFrame if no matches are found or an error.

        Notes: 
            - Adds 'state_name' to self.categorical_features if it isn't already present.
            - Ensures downstream group-aware splitting by 'pedlabsampnum' does not leak across folds.
        """
        logger.debug("Matching with crop yields using expanded coverage...")
        
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
                logger.error("Crop response does not exist!")
            logger.debug(f"Retrieved {len(crop_df)} crop yield records")
            
            crop_df['yield_value'] = pd.to_numeric(
                crop_df['value'].astype(str).str.replace(',', '').str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce'
                )
            invalid_years = crop_df[crop_df['year'].isna()]
            logger.debug(f"invalid years: {invalid_years}")
            crop_df['year'] = pd.to_numeric(crop_df['year'], errors = 'coerce')
            crop_df = crop_df[crop_df['year'].notna()].copy()
            
            crop_df = crop_df[
                (crop_df['yield_value'].notna()) & 
                (crop_df['yield_value'] > 0) & 
                (crop_df['yield_value'] < 1000)
            ].copy()
            logger.debug(f"Filtered crop yields, {len(crop_df)} valid records remaining")
            
            matched_records = []
            dropped = crop_df['state_name'].isna().sum()
            logger.debug(f"Dropped {dropped} crop rows with missing state_name")
            
            crop_df = crop_df[crop_df['state_name'].notna()].copy()

            crop_df['state_name'] = crop_df['state_name'].str.upper().str.strip()
            crop_df['county_name'] = crop_df['county_name'].str.upper().str.strip()

            logger.debug(
                f"Crop states available ({crop_df['state_name'].nunique()}): "
                f"{sorted(crop_df['state_name'].unique())[:10]} ..."
            )
            
            logger.debug(f"Upper bound for enhanced iterrows: {len(enhanced_df)}")

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

                    # A single soil sample may be matched to multiple crop-year observations.
                    # Group-aware splitting by 'pedlabsampnum' is used to prevent cross-fold leakage
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

                logger.debug(f"Successfully matched {len(matched_df)} training records")
                return matched_df
            else:
                logger.warning("No matches found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error matching crop yields: {e}")
            return pd.DataFrame()
    

    def build_pipeline(self):
        """
        Constructs the full preprocessing and modeling pipeline used for training and evaluation.

        The pipeline performs group-aware imputation by state, generates derived
        features, applies numeric scaling and categorical one-hot encoding, and fits a Random Forest 
        regressor. Feature lists are dynamically extended based on derived features to ensure 
        consistent preprocessing.

        This method centralises pipeline assembly to guarantee that identical
        transformations are applied across cross-validation folds. 
        """
        logger.debug("Building preprocessing and modeling pipeline")

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

        logger.debug("Pipeline successfully constructed")
        return pipeline
        

    # Training
    def train_with_grid_search(self, df):
        """
        Train a model using group-aware train/test split, grid search, and diagnostics.

        Steps:
        1. Constructs a group-aware train-test split to prevent leakage across soil samples.
        2. Defines and fits grid search with GroupKFold CV for hyperamater tuning (R^2 scoring).
        3. Extracts best estimator and applies fitted preprocessing (imputer + feature derivation)
        to transform training and holdout sets.
        4. Runs diagnostics, including:
            - CV fold metrics (RMSE, MAE, R^2, normalized target variance, high-variance fraction)
            - Local target variance in mixed numeric + categorical feature space
            - Train-test similarity using Gower distance
        5. Evaluates the holdout set and logs metrics (RMSE, MAE, R^2, local variance)

        Args: 
            df: DataFrame containing merged and matched soil samples and corresponding crop yields.

        Returns:
            best_model: Pipeline fitted with the best hyperparameters from grid search.

        Notes:
        - Local target variance is computed per sample using a mixed-distance metric
        that weights numeric L1 differences and categorical mismatches.
        - Group-aware splitting ensures no data leakage between train and test sets for 
        the same soil sample ('pedlabsampnum').
        """
        logger.debug("Splitting data into train/test sets")
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

        logger.debug(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

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

        best_model = grid.best_estimator_

        feature_pipeline = Pipeline([
            ('imputer', best_model.named_steps['imputer']),
            ('derived', best_model.named_steps['derived']) 
        ])

        X_train_feat = feature_pipeline.transform(X_train)
        X_train_feat = pd.DataFrame(
            X_train_feat, 
            columns = self.numeric_features + self.categorical_features,
            index = X_train.index
        )

        X_test_feat = feature_pipeline.transform(X_test)
        X_test_feat = pd.DataFrame(
            X_test_feat, 
            columns = self.numeric_features + self.categorical_features,
            index = X_test.index
        )

        diagnostics = self.run_diagnostics(
            X_train = X_train_feat, 
            y_train = y_train, 
            X_test = X_test_feat,
            y_test = y_test,
            groups = groups.iloc[train_idx],
            cv = cv,
            best_model = best_model
        )

        logger.debug(f"Best params: {grid.best_params_}")
        logger.info(f"Best CV Score (R^2): {grid.best_score_:.4}")

        preds = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        logger.debug(f"Holdout metrics -> RMSE: {rmse:.3f}, MAE: {mae: .3f}, R^2: {r2: .3f}")
        logger.info(f"Holdout R^2: {r2: .3f}")

        self.model = best_model
        return best_model


    def run_diagnostics(self, X_train, y_train, X_test, y_test, groups, cv, best_model):
        """
        Runs a suite of model diagnostics for cross-validation folds and the holdout set.

        Measures include:
        1. Nearest-neighbor similarity between train and test folds used in CV (Gower-based).
        2. Local target variance in mixed-feature space (numeric + categorical) for training and holdout sets.
        3. Fraction of samples in high local variance regions in folds and holdout.
        4. Fold-level performance metrics (RSME, MAE, R^2, normalized y variance, y range).
        5. Identification of "bad" folds with unusually low R^2.
        6. Correlation between fold R^2 and high local variance fraction.

        Args: 
            X_train: Training features
            y_train: Training targets
            X_test: Validation features
            y_test: Validation features
            groups: Group labels for group-aware CV
            cv: CV splitter
            best_model: Estimator to clone fit on each fold

        Returns:
        - similarity_summary: DataFrame summarizing max NN Gower similarity per CV fold.
        - local_var: Series of local target variance for each training sample.
        """
        similarity_summary = run_cv_similarity_diagnostics(
            cv = cv,
            X = X_train,
            y = y_train, 
            groups = groups,
            numeric_features = self.numeric_features,
            categorical_features = self.categorical_features
        )

        local_var = compute_local_y_variance(
            X = X_train,
            y = y_train,
            numeric_features = self.numeric_features, 
            categorical_features = self.categorical_features, 
            k = 10
        )

        target_var = compute_local_y_variance(
            X = X_test,
            y = y_test, 
            numeric_features = self.numeric_features,
            categorical_features = self.categorical_features
        )

        fold_df = fold_stats(
            cv, X_train, y_train, groups, best_model
        )

        fold_df["is_bad_fold"] = fold_df["r2"] < 0.3

        if fold_df['is_bad_fold'].any():
            bad_folds = fold_df.index[fold_df['is_bad_fold']]
            logger.warning(f"Detected {fold_df['is_bad_fold'].sum()} bad Cv fold(s): {list(bad_folds)}\n"
            "Likely due to high target variance")

        train_threshold = local_var.quantile(0.9)
        fold_df["high_var_frac"] = fold_df["val_idx"].apply(
            lambda idxs: (local_var.iloc[idxs] > train_threshold).mean()
        )

        test_threshold = target_var.quantile(0.9)
        target_high_var_frac = target_var.apply(lambda x: x > test_threshold).mean()
         
        logger.debug(f"\n{fold_df.drop(columns = "val_idx")}")

        y_var_corr = fold_df[["r2", "high_var_frac"]].corr().iloc[0, 1] 

        logger.debug(f"Holdout local var normalized: {target_var.mean() / np.var(y_train, ddof = 1): .3f}")
        logger.debug(f"Holdout high var frac: {target_high_var_frac: .3f}")
        logger.info(f"Correllation(R^2, high local variance fraction): {y_var_corr:.3f}")

        logger.debug(f"Train test similarity summary (max NN Gower similarity, averaged per fold):\n" 
        f"{similarity_summary}")
        return similarity_summary, local_var
        

# Main Pipeline Runner
def main():
    """
    Orchestrates the full weather-based crop yield modeling pipeline.

    Steps:
    1. Fetch weather and soil data from Supabase.
    2. Assign geographic state information to each soil sample.
    3. Match enriched data with historical crop yields.
    4. Save preprocessed training data for diagnostics and reproducibility.
    5. Train a Random Forest regression model with grid search and group-aware cross-validation.

    Logs progress at each step and aborts cleanly if data is missing.
    """ 
    logger.debug("Starting full training pipeline")
    trainer = WeatherBasedModelTrainer()
    
    try:
        logger.info("Starting weather-based model training")
        
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
        
        # Step 3: Match with crop yields
        training_data = trainer.match_with_crop_yields(enhanced_data)
        if training_data.empty:
            logger.error("No training data created, aborting")
            return
        
        # Save for Diagnostics
        PROCESSED_DATA_PATH = DATA_DIR / 'training_data_processed.csv'
        training_data.to_csv(PROCESSED_DATA_PATH, index = False) # not saving index
        logger.debug(f"Saved processed training data to {PROCESSED_DATA_PATH}")

        # step 3: Run pipeline
        model = trainer.train_with_grid_search(df = training_data)
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()