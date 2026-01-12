#!/usr/bin/env python3
"""
B-ALL TreeSHAP Analysis Script
==================================
Computes SHAP values using XGBoost's native TreeSHAP implementation
and generates beeswarm plots for B-ALL model variants.

Uses 10,000 samples per model variant.
"""

import os
import argparse
import sys
import gc
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import rankdata

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_DIR = os.path.join(settings.OUTPUT_DIR, 'xgb')
OUTPUT_DIR = os.path.join(settings.OUTPUT_DIR, 'shap_analysis_b_all')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Variants to analyze
VARIANTS = {
    'B-ALL': {'name': 'All Sky Conditions', 'bcm_filter': None},
    'B-ALL-CLEAR': {'name': 'Clear Sky Only', 'bcm_filter': 0},
    'B-ALL-CLOUDY': {'name': 'Cloudy Sky Only', 'bcm_filter': 1}
}

SAMPLES = 10000
RANDOM_STATE = settings.RANDOM_STATE

# ==============================================================================
# DATA LOADING
# ==============================================================================
def get_lazy_frame():
    """Initialize Polars LazyFrame for memory-efficient data access."""
    print("\n--- Initializing Polars LazyFrame ---")
    features = settings.FEATURE_SETS['B-ALL']
    cols_to_load = features + ['ACMC_BCM']
    
    print(f"Scanning data from: {settings.DATA_PATH}")
    q = pl.scan_csv(settings.DATA_PATH).select(cols_to_load)
    q = q.with_columns([pl.col(c).cast(pl.Float32) for c in cols_to_load])
    
    return q, features

# ==============================================================================
# SHAP COMPUTATION (NATIVE XGBOOST TREESHAP)
# ==============================================================================
def compute_native_shap(model, X_np, features):
    """
    Computes SHAP values using XGBoost's native TreeSHAP implementation.
    
    This is the built-in method: booster.predict(..., pred_contribs=True)
    It returns exact Shapley values in polynomial time for tree models.
    
    Returns:
        np.ndarray: SHAP values of shape (n_samples, n_features)
    """
    booster = model.get_booster()
    
    # Try to use GPU if model was trained with it
    try:
        booster_config = booster.save_config()
        if 'cuda' in booster_config.lower() or 'gpu' in booster_config.lower():
            booster.set_param({'device': 'cuda'})
            print("    Using GPU for SHAP computation")
    except Exception:
        pass  # Fall back to CPU
        
    dmatrix = xgb.DMatrix(X_np, feature_names=features)
    
    # pred_contribs=True enables native TreeSHAP
    shap_values_with_bias = booster.predict(dmatrix, pred_contribs=True)
    
    # XGBoost returns (n_samples, n_features + 1) where last column is bias term
    shap_values = shap_values_with_bias[:, :-1]
    
    return shap_values

# ==============================================================================
# BEESWARM PLOT (CUSTOM MATPLOTLIB IMPLEMENTATION)
# ==============================================================================
def plot_beeswarm(shap_values, X_df, title, filename, max_display=20):
    """
    Generates a SHAP beeswarm-style plot using Matplotlib.
    
    Points are colored by feature value (Red=High, Blue=Low).
    This mimics the shap.summary_plot() visualization without requiring
    the external SHAP library.
    
    Args:
        shap_values: np.ndarray of shape (n_samples, n_features)
        X_df: pandas DataFrame with feature values (for coloring)
        title: Plot title
        filename: Output file path
        max_display: Number of top features to show
    """
    print(f"  Generating Beeswarm plot...")
    
    # 1. Calculate feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 2. Sort features by importance (ascending for bottom-to-top display)
    indices = np.argsort(mean_abs_shap)
    top_indices = indices[-max_display:]  # Top N features
    
    top_features = X_df.columns[top_indices]
    top_shap = shap_values[:, top_indices]
    top_X = X_df.iloc[:, top_indices].values
    
    # 3. Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Y-positions for each feature
    y_pos = np.arange(len(top_features))
    
    # Colormap: Red = High feature value, Blue = Low feature value
    cmap = cm.get_cmap('RdBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # 4. Plot each feature
    np.random.seed(RANDOM_STATE)  # For reproducible jitter
    
    for i in range(len(top_features)):
        s_vals = top_shap[:, i]  # SHAP values (x-axis)
        x_vals = top_X[:, i]     # Feature values (for coloring)
        
        # Normalize feature values for coloring using rank (more robust than min-max)
        if len(np.unique(x_vals)) > 1:
            color_vals = rankdata(x_vals) / len(x_vals)
        else:
            color_vals = np.full_like(x_vals, 0.5)
            
        # Add jitter to y-values for beeswarm effect
        jitter = np.random.normal(0, 0.12, size=len(s_vals))
        
        # Plot scatter
        ax.scatter(s_vals, y_pos[i] + jitter, 
                   c=color_vals, cmap=cmap, s=6, alpha=0.5, 
                   edgecolors='none', norm=norm, rasterized=True)

    # 5. Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_ylim(-0.5, len(top_features) - 0.5)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label('Feature Value (Low → High)', rotation=270, labelpad=20, fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute TreeSHAP values and generate beeswarm plots for B-ALL variants."
    )
    parser.add_argument(
        '--models', 
        nargs='+',
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help=f"Model variants to analyze. Choices: {list(VARIANTS.keys())}. Default: all three."
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("B-ALL TreeSHAP Analysis")
    print("=" * 60)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Samples per model: {SAMPLES:,}")
    print(f"Models to analyze: {args.models}")
    
    # Get LazyFrame (memory efficient - doesn't load all data)
    q_all, features = get_lazy_frame()
    print(f"Features: {len(features)}")
    
    # Filter variants based on command-line args
    selected_variants = {k: v for k, v in VARIANTS.items() if k in args.models}
    
    for variant_key, config in selected_variants.items():
        print(f"\n{'=' * 50}")
        print(f"Processing: {variant_key} ({config['name']})")
        print("=" * 50)
        
        # 1. Load Model (1 model per variant, NOT 53)
        model_path = os.path.join(MODEL_DIR, variant_key, f"{variant_key}_model.joblib")
        if not os.path.exists(model_path):
            print(f"  ❌ Model not found: {model_path}")
            print(f"     Run: python main.py --model_type {variant_key}")
            continue
            
        print(f"  Loading single model: {model_path}")
        model = joblib.load(model_path)
        
        # 2. Filter Data (Lazy)
        if config['bcm_filter'] is not None:
            q_variant = q_all.filter(pl.col('ACMC_BCM') == config['bcm_filter'])
        else:
            q_variant = q_all
            
        # Count available rows
        print("  Counting available rows...")
        try:
            count = q_variant.select(pl.len()).collect().item()
        except Exception as e:
            print(f"  Error counting rows: {e}")
            continue

        if count == 0:
            print("  ⚠️ No data found for this variant.")
            continue
            
        # 3. Sample data
        n = min(SAMPLES, count)
        print(f"  Sampling {n:,} from {count:,} available observations...")
        
        try:
            # Collect full data then sample (LazyFrame doesn't support sample directly)
            df_full = q_variant.collect()
            df_sample = df_full.sample(n, seed=RANDOM_STATE)
            del df_full  # Free memory
        except Exception as e:
            print(f"  Sampling failed: {e}")
            continue

        X_sample_np = df_sample.select(features).to_numpy()
        X_sample_pd = df_sample.select(features).to_pandas()
        
        # 4. Compute SHAP values using native XGBoost TreeSHAP
        print("  Computing TreeSHAP values (XGBoost native)...")
        shap_values = compute_native_shap(model, X_sample_np, features)
        print(f"    SHAP values shape: {shap_values.shape}")
        
        # 5. Generate Beeswarm Plot
        plot_title = f"SHAP Beeswarm: {variant_key}\n({config['name']}, n={n:,})"
        plot_file = os.path.join(OUTPUT_DIR, f"shap_beeswarm_{variant_key}.jpg")
        plot_beeswarm(shap_values, X_sample_pd, plot_title, plot_file)
        
        # 6. Save Feature Importance (Mean |SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': features,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        csv_file = os.path.join(OUTPUT_DIR, f"shap_importance_{variant_key}.csv")
        importance_df.to_csv(csv_file, index=False)
        print(f"  ✓ Saved importance: {csv_file}")
        
        # Print top 10 features
        print(f"\n  Top 10 Features by Mean |SHAP|:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {row['Feature']:12s}: {row['Mean_Abs_SHAP']:.4f}")
        
        # Cleanup
        del model, shap_values, X_sample_np, X_sample_pd
        gc.collect()
        
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
