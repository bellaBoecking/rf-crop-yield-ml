"""
paths.py

Centralized path configuration for treeModel project.

Defines absolute, project-root-relative filesystem paths used throughout the codebase,
ensuring consistent and portable access to data, pipelines, and diagnostic
artifacts across environments.

All paths are resolved dynamically from the project root to avoid hard-coded 
filesystem dependencies.
"""
from pathlib import Path

# treeModel/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DIAGNOSTICS_DIR = PROJECT_ROOT / "diagnostics"

STATE_SHP = DATA_DIR / "tl_2025_us_state.shp"