import os

# Paths configuration
# Adjusted to be relative to where the script is run, or user can override via env vars
BASE_DIR = os.getenv('LST_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assuming the data is in data/processed relative to project root, or absolute path
DATA_PATH = os.getenv('LST_DATA_PATH', os.path.join(BASE_DIR, 'datasets/processed/ML_READY_mesonet_goes_embeddings_2024.csv'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'models')  # Base models dir, will be structured as models/xgb/{model_type}/
TEMP_PRED_DIR = os.path.join(OUTPUT_DIR, 'temp_predictions')  # Temporary predictions for analysis

# Target Definition
TARGET_COL = 'LST'  # Land Surface Temperature

# Feature Definitions
CMI_BANDS = [f'CMI_C{i:02d}' for i in range(1, 17)]
EMBEDDINGS = [f'A{i:02d}' for i in range(64)]

# Split keys (for identification/splitting, NOT for training)
SPLIT_KEYS = ['SITE_ID', 'LOCAL_TIME', 'ACMC_BCM']

# Auxiliary Features (Numeric, used for training)
# Auxiliary Features (Numeric, used for training)
AUXILIARY_FEATURES = [
    'Elevation', 
    'SZA_sin', 'SZA_cos', 
    'SAA_sin', 'SAA_cos', 
    'HOUR_sin', 'HOUR_cos', 
    'DOY_sin', 'DOY_cos'
]

# All metadata columns to load
METADATA = SPLIT_KEYS + AUXILIARY_FEATURES

# Climate Divisions (One-Hot Encoded)
CLIMATE_DIVISIONS = [
    'CD_Hawaiʻi Mauka', 'CD_Hilo', 'CD_Kaʻu', 'CD_Kona', 'CD_Koʻolau',
    'CD_Leeward Kauaʻi', 'CD_Leeward Kohala', 'CD_Leeward Maui Nui',
    'CD_Waianae', 'CD_Windward Kauaʻi', 'CD_Windward Kohala', 'CD_Windward Maui Nui'
]

# Feature Groups Definition (Must only contain numeric features for XGBoost)
FEATURE_SETS = {
    'BLM': AUXILIARY_FEATURES + [f'CMI_C{i:02d}' for i in range(13, 17)],       # Baseline
    'BLAM': AUXILIARY_FEATURES + [f'CMI_C{i:02d}' for i in range(13, 17)] + EMBEDDINGS, # Baseline + AEFE
    'BLAM-C': AUXILIARY_FEATURES + EMBEDDINGS,                                   # Baseline + AEFE - CMI (Embeddings + Auxiliary)
    'CIM': AUXILIARY_FEATURES + CMI_BANDS,                                       # CMI Only
    'CIAM': AUXILIARY_FEATURES + CMI_BANDS + EMBEDDINGS,                         # CMI + AEFE
}

# Scaling Configuration
# CMI bands and Elevation are typically scaled. 
# Embeddings (A00-A63) are usually already normalized or suitable for XGB as is.
SCALING_COLS = CMI_BANDS + ['Elevation']

# XGBoost Params Defaults (can be overridden by Optuna)
DEFAULT_XGB_PARAMS = {
    'n_estimators': 1000,
    # 'device': 'cuda', # Commented out for safety on generic environments, uncomment if GPU confirmed
    'tree_method': 'hist', # Efficient for large data
    'early_stopping_rounds': 50
}

RANDOM_STATE = 808
