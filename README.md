# Crop Yield Prediction with Stability-Aware Diagnostics

#### Random Forest Regression on Weather, Soil, and Spatial-Temporal Heuristics

## Overview
This project implements an end-to-end machine learning pipeline for predicting crop yields using heterogeneous environmental data, including:
- Weather observations
- Soil chemical and physical laboratory measurements
- Spatial information derived from TIGER state polygons
- Historical crop yield records
The core modeling approach uses a Random Forest Regressor accompanied by a structured diagnostic framework. The pipeline explicitly analyzes local target instability, fold-level performance variation, and train-test similarity in mixed feature spaces. 

The outcome of this project is both a predictive model and a structured investigation into where and why predictive accuracy varies. This work was created to demonstrate predictive accuracy, applied ML reasoning, a deep understanding of how the data behaves, its implications for modeling, and production-aware pipeline design.

## Problem Setting
Some of the challenges faced during the project's construction include:
- Coarse spatial alignment (soil samples vs. state-level yield data)
- Temporal mismatch between sampling and reported yields
- Neighborhoods of high target variance, where predictive accuracy is structurally limited
- Mixed numeric and categorical feature spaces

The project explicitly models and measures local violations of smoothness assumptions instead of assuming  i.i.d or smooth target behavior everywhere, quantifying where smoothness fails and connecting it to model uncertainty.

## Data Sources:
Data is retrieved from Supabase-backed tables and merged via left joins to preserve observational integrity and prioritize weather data:
- Weather and soil samples (weather_soil_samples)
- Soil chemical properties (ssurgo_lab_chemical_properties)
- Soil physical properties (ssurgo_lab_physical_properties)
- Crop yield records (nass_crops)

### Spatial Enrichment
Soil samples are assigned U.S. labels via point-in-polygon joins using TIGER shapefiles, reprojected to WGS84 (EPSG: 4326). State-level matching is treated as a coarse spatial proxy, not a claim of fine-grained spatial alignment.

## Matching Logic: Soil -> Crop Yield
Each soil-weather observation is matched to historical crop yields using appropriate spatial and temporal heuristics:
- Crops: CORN, SOYBEANS, WHEAT, COTTON, BARLEY
- Years: 1948-2025
- Fallback: Closest prior year if no match exists within the window

A single soil sample may generate multiple training rows (one matched per crop-year), which motivates group-aware splitting downstream.

## Feature Engineering

### Raw Features:
- Weather aggregates (temperature, precipitation, humidity, GDD)
- Soil chemistry (pH, carbon, nitrogen, CEC, base saturation)
- Soil texture and physical structure
- Spatial coordinates and sample year

### Derived Features:
Deterministic, domain-informed features added via a custom transformer
- soil_quality_score (composite heuristic)
- temp_optimality (distance from agronomic optimum)
- ca_mg_ratio
- gdd_suitability (crop-specific categorical feature)
No learned parameters are introduced at this stage; all derivations are reproducible and pipeline-safe.

## Preprocessing and Leakage Control

### Group-Aware Imputation
Missing values are imputed within state groups, with global fallbacks where necessary.
- Numeric features -> state-level medians
- Categorical values -> state-level modes
This technique avoids leakage across spatial regions while remaining robust to sparsely populated groups.

### Encoding and Scaling:
- Numeric features: standardized
- categorical features: one-hot encoded with safe handling of unseen categories

All preprocessing occurs inside a single scikit-learn pipeline, ensuring identical transformations across folds.

## Model
Estimator: RandomForestRegressor
Tuning: Grid search over depth, tree count, and split parameters
Scoring: R^2
Cross-Validation: GroupKFold using soil sample IDs (pedlabsampnum)

Group-aware splitting ensures that no soil sample appears in both training and validation sets, even when matched to multiple crop years.

RandomForestRegressor was chosen due to its ability to capture non-linear relationships, handle mixed feature types, computational efficiency, and favorable performance metrics in multi-model crop yield prediction studies.

## Diagnostics

### 1. Local Target Variance:
For each observation, the project computes Var(Y|X~x) using a custom mixed-feature distance designed to preserve numeric smoothness:
- Numeric features: MinMax-scaled L1 distance
- Categorical features: normalized Hamming distance
- Combined via feature-count-weighted averaging

This avoids discontinuities induced by standard Gower distance while remaining applicable to mixed data.

High local variance regions indicate where predictability is structurally limited by the data.

### 2. Fold-level Instability Analysis
For each cv fold, the pipeline reports:
- RMSE, MAE, R^2
- Normalized target variance
- Target range
- Fraction of samples in high-variance neighborhoods

Folds with unusually low R^2 are explicitly identified and interpreted through the lens of target instability rather than model failure.

### 3. Train-Test Similarity (Gower Diagnostics)
A Gower-based nearest-neighbor similarity analysis quantifies how “familiar” validation samples are relative to training data in each fold.

This separates distributional shift effects from intrinsic target noise, suggesting covariate shift is not the primary driver of high holdout R^2 variance between runs.

### 4. Correlation Analysis
The pipeline computes the correlation between:
- Fold-level R^2
- Fraction of high local variance samples
Provides strong evidence that performance degradation is positively correlated with local instability, not random variance.

## Results Interpretation:
Model performance is intentionally contextualized:
- Strong average performance does not imply universal predictability
- Evidence suggests that some regions of high feature similarity exhibit unstable targets
- Diagnostic signals explain where the model can and cannot be trusted
- The model provides strong holdout performance in the majority of runs, but performance degrades when high-variance neighborhoods are unevenly distributed across folds, occasionally resulting in poor validation scores

## Engineering Highlights
- End-to-end reproducible pipeline
- Strict leakage prevention via group-aware splitting
- Deterministic feature derivation
- Modular diagnostics architecture
- Centralized path configuration
- Verbose logging with fold-level transparency
- Explicit diagnosis of where assumptions fail
- Interpretable and robust under authentic, messy data
- Standard and custom distance metrics used to probe data geometry and preserve numeric smoothness in mixed-feature spaces

## Limitations and Future Work
- State-level matching is a coarse spatial proxy
- County or field-level alignment is expected to improve signal quality
- Explicit spatial-temporal kernels could replace heuristics
- Diagnostics can be extended into uncertainty-aware prediction intervals