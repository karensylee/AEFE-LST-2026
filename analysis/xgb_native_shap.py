# ==============================================================================
# XGBOOST NATIVE SHAP - PER STATION (A100 & POLARS OPTIMIZED)
# ==============================================================================
"""
This script computes SHAP values per station using XGBoost's native TreeSHAP 
implementation. Optimized for A100 GPU with safe batching and CPU fallback.

Usage:
    python analysis/xgb_native_shap.py --model_type BLAM
"""

import os
import gc
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import joblib
import xgboost as xgb

# Optional torch import (only needed for GPU memory management)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

print(f"NumPy version: {np.__version__}")
print(f"Polars version: {pl.__version__}")
print(f"XGBoost version: {xgb.__version__}")
if TORCH_AVAILABLE:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch: Not installed (GPU memory management disabled)")
print("--- Libraries Imported Successfully ---")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_STATE = 808
np.random.seed(RANDOM_STATE)

# Sampling and batching configuration
SAMPLES_PER_STATION = 10000
BATCH_SIZE = 500  # Safe batching for GPUTreeSHAP on A100
CHECKPOINT_EVERY = 2

# Paths configuration
BASE_DIR = '/home/ksylee/projects/lst'
DATA_PATH = os.path.join(BASE_DIR, 'datasets/processed/ML_READY_mesonet_goes_embeddings_2024.csv')

# Target Definition
TARGET_VARIABLE = 'LST'

# Feature Definitions
CMI_FEATURES = [f'CMI_C{i:02d}' for i in range(1, 17)]
AEFE_FEATURES = [f'A{i:02d}' for i in range(64)]
AUXILIARY_FEATURES = [
    'Elevation', 'HOUR_sin', 'HOUR_cos', 'DOY_sin', 'DOY_cos',
    'SZA_sin', 'SZA_cos', 'SAA_sin', 'SAA_cos'
]

# Feature Sets (matching config/settings.py)
FEATURE_SETS = {
    'BLM': AUXILIARY_FEATURES + CMI_FEATURES,
    'BLAM': AUXILIARY_FEATURES + CMI_FEATURES + AEFE_FEATURES,
    'BLAM-C': AUXILIARY_FEATURES + AEFE_FEATURES,
    'CIM': CMI_FEATURES,
    'CIAM': CMI_FEATURES + AEFE_FEATURES,
}

# Scaling columns (CMI bands + Elevation)
SCALING_COLS = CMI_FEATURES + ['Elevation']


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_model_and_scaler(model_dir, model_prefix, scaler_prefix, station_id_str):
    """
    Helper to find and load models/scalers using flexible naming conventions.
    
    Args:
        model_dir: Directory containing model and scaler files
        model_prefix: Prefix for model files (e.g., 'BLAM_model_')
        scaler_prefix: Prefix for scaler files (e.g., 'BLAM_scaler_')
        station_id_str: Station ID as string
    
    Returns:
        Tuple of (model, scaler) or (None, None) if not found
    """
    # Check multiple possible paths
    possible_model_paths = [
        os.path.join(model_dir, f'{model_prefix}{station_id_str}.joblib'),
        os.path.join(model_dir, f'embedded_xgb_model_holdout_{station_id_str}.joblib')
    ]

    model = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"  Loaded model: {os.path.basename(path)}")
            break

    if model is None:
        return None, None

    # Load scaler if defined
    scaler = None
    scaler_path = os.path.join(model_dir, f'{scaler_prefix}{station_id_str}.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  Loaded scaler: {os.path.basename(scaler_path)}")

    return model, scaler


def compute_native_shap(model, X_np, feature_names, batch_size=200):
    """
    Computes SHAP values with a focus on memory safety.
    Uses small batches and provides a CPU fallback.
    
    Args:
        model: Trained XGBoost model
        X_np: Feature array (numpy)
        feature_names: List of feature names
        batch_size: Batch size for GPU processing
    
    Returns:
        SHAP values array (without bias column)
    """
    booster = model.get_booster()
    num_samples = X_np.shape[0]
    all_shaps = []

    try:
        # Attempt GPU Path
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        booster.set_param({'device': 'cuda'})

        for i in range(0, num_samples, batch_size):
            X_batch = X_np[i : i + batch_size]
            dmatrix = xgb.DMatrix(X_batch, feature_names=feature_names)

            # This is where the OOM usually happens
            batch_shap_with_bias = booster.predict(dmatrix, pred_contribs=True)
            all_shaps.append(batch_shap_with_bias[:, :-1])
            del dmatrix

        return np.vstack(all_shaps)

    except Exception as e:
        print(f"  ⚠️ GPU SHAP failed or OOM (Requested too much memory). Falling back to CPU...")
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fallback to CPU Path
        booster.set_param({'device': 'cpu'})
        # CPU can usually handle larger batches, but we keep it simple
        dmatrix = xgb.DMatrix(X_np, feature_names=feature_names)
        shap_values_with_bias = booster.predict(dmatrix, pred_contribs=True)
        return shap_values_with_bias[:, :-1]


def get_scale_col_indices(feature_set, scaling_cols):
    """
    Get indices of columns that need scaling within the feature set.
    """
    indices = []
    for i, feat in enumerate(feature_set):
        if feat in scaling_cols:
            indices.append(i)
    return indices


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="XGBoost Native SHAP Analysis")
    parser.add_argument('--model_type', type=str, default='BLAM', 
                        choices=list(FEATURE_SETS.keys()),
                        help="Model type to analyze (default: BLAM)")
    args = parser.parse_args()
    
    model_type = args.model_type
    FEATURE_SET = FEATURE_SETS[model_type]
    
    # Directories
    MODEL_DIR = os.path.join(BASE_DIR, 'models/xgb', model_type)
    OUTPUT_DIR = os.path.join(MODEL_DIR, 'shap_analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Model/scaler prefixes (matching main.py naming)
    MODEL_PREFIX = f'{model_type}_model_'
    SCALER_PREFIX = f'{model_type}_scaler_'
    
    # Get scaling column indices
    scale_col_indices = get_scale_col_indices(FEATURE_SET, SCALING_COLS)
    
    print("\n" + "="*70)
    print(f"XGBOOST NATIVE SHAP ANALYSIS - PER STATION ({model_type})")
    print("="*70)
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Feature Set: {len(FEATURE_SET)} features")
    print(f"Samples per Station: {SAMPLES_PER_STATION}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # ==============================================================================
    # LOAD DATA
    # ==============================================================================
    print("\n--- Loading Data with Polars ---")
    
    cols_to_load = FEATURE_SET + ['SITE_ID', 'ACMC_BCM', 'LOCAL_TIME', TARGET_VARIABLE]
    
    try:
        # Polars scan for lazy loading and efficient memory management
        data_df = pl.scan_csv(DATA_PATH).select(cols_to_load).with_columns([
            pl.col(FEATURE_SET).cast(pl.Float32),
            pl.col(TARGET_VARIABLE).cast(pl.Float32),
            pl.col('SITE_ID').cast(pl.String),
            pl.col('ACMC_BCM').cast(pl.Float32)
        ]).collect()
    
        print(f"Loaded data: {data_df.height:,} rows × {data_df.width} columns")
        print(f"Memory usage: {data_df.estimated_size('mb'):.1f} MB")
    
        station_ids = sorted(data_df['SITE_ID'].unique().to_list())
        print(f"Unique stations: {len(station_ids)}")
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # ==============================================================================
    # EXECUTION LOOP
    # ==============================================================================
    all_station_shap_importance = []
    checkpoint_path = os.path.join(OUTPUT_DIR, 'shap_checkpoint.csv')
    
    # Resume if checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint_pl = pl.read_csv(checkpoint_path)
        completed_stations = set(checkpoint_pl['station'].cast(pl.String).to_list())
        all_station_shap_importance = checkpoint_pl.to_dicts()
        print(f"Resuming from checkpoint. {len(completed_stations)} stations completed.")
    else:
        completed_stations = set()
    
    for idx, station_id in enumerate(station_ids):
        s_id_raw = str(station_id)
        s_id_padded = s_id_raw.zfill(4)
    
        # Check both raw and padded IDs in completed set
        if s_id_raw in completed_stations or s_id_padded in completed_stations:
            continue
    
        station_start = time.time()
        print(f"\n[{idx+1}/{len(station_ids)}] Processing station: {s_id_raw}")
    
        # 1. Load model/scaler using the padded ID (0115)
        model, scaler = load_model_and_scaler(MODEL_DIR, MODEL_PREFIX, SCALER_PREFIX, s_id_padded)
    
        # Fallback to raw ID (115) if padded fails
        if model is None:
            model, scaler = load_model_and_scaler(MODEL_DIR, MODEL_PREFIX, SCALER_PREFIX, s_id_raw)
    
        if model is None:
            print(f"  ⚠️ Skipping: Model files not found for {s_id_raw} or {s_id_padded}")
            continue
    
        # 2. Filter data using raw ID (Polars expects exact match to SITE_ID column)
        station_data = data_df.filter(pl.col("SITE_ID").cast(pl.String) == s_id_raw)
    
        if station_data.height == 0:
            print(f"  ⚠️ No data found in dataframe for SITE_ID: {s_id_raw}")
            continue
    
        # 3. Sampling and Conversion
        if station_data.height > SAMPLES_PER_STATION:
            X_shap_raw = station_data.select(FEATURE_SET).sample(n=SAMPLES_PER_STATION, seed=RANDOM_STATE).to_numpy()
        else:
            X_shap_raw = station_data.select(FEATURE_SET).to_numpy()
    
        # 4. Scaling
        if scaler is not None and len(scale_col_indices) > 0:
            X_shap_raw[:, scale_col_indices] = scaler.transform(X_shap_raw[:, scale_col_indices])
    
        # 5. SHAP Computation
        try:
            shap_values = compute_native_shap(model, X_shap_raw, FEATURE_SET, batch_size=BATCH_SIZE)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
            # Log Top 3 to monitor progress
            top_idx = np.argsort(mean_abs_shap)[-3:][::-1]
            print(f"  ✅ SHAP Complete. Top: {FEATURE_SET[top_idx[0]]} ({mean_abs_shap[top_idx[0]]:.4f})")
    
            # Store using the raw ID for consistency in the final CSV
            result = {'station': s_id_raw}
            for i, feat in enumerate(FEATURE_SET):
                result[feat] = mean_abs_shap[i]
            all_station_shap_importance.append(result)
    
        except Exception as e:
            print(f"  ❌ Computation Error: {e}")
            continue
    
        # 6. Checkpoint and Cleanup
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pl.from_dicts(all_station_shap_importance).write_csv(checkpoint_path)
            print(f"  💾 Checkpoint saved to CSV.")
    
        del model, X_shap_raw, station_data
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        print(f"  Station Time: {time.time() - station_start:.1f}s")
    
    # Final Save
    if all_station_shap_importance:
        final_shap_df = pl.from_dicts(all_station_shap_importance)
        final_path = os.path.join(OUTPUT_DIR, 'station_shap_importance_final.csv')
        final_shap_df.write_csv(final_path)
        print(f"\n✅ FULL ANALYSIS COMPLETE. Saved to {final_path}")
        print(f"   Total stations processed: {len(all_station_shap_importance)}")
    else:
        print("\n⚠️ No SHAP results computed. Check model files and data.")


if __name__ == "__main__":
    main()
