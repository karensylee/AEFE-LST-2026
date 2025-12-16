"""
Density Plot Analysis for LST Predictions
Generates density plots comparing predicted vs actual LST for BLM and BLHIM models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# Start timing
t0 = time.time()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def calculate_metrics_detailed(df, true_col='true', pred_col='pred'):
    """Calculates expanded metrics for a dataframe."""
    if df.empty or len(df) < 2:
        return (np.nan,) * 7

    error = df[pred_col] - df[true_col]

    rmse = np.sqrt(mean_squared_error(df[true_col], df[pred_col]))
    median_err = np.median(error)
    mean_err = np.mean(error)  # This is Bias
    std_err = np.std(error)
    r2 = r2_score(df[true_col], df[pred_col])
    n = len(df)

    # Return metrics in order: RMSE, Median, Mean, STD, R², n, Bias
    return rmse, median_err, mean_err, std_err, r2, n, mean_err


def plot_density_fig(df, title, vmin, vmax, save_path, station_id=None, 
                     true_col='true', pred_col='pred'):
    """
    Creates, saves, and closes a density plot.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Calculate metrics
    rmse, med_err, mean_err, std_err, r2, n, bias = calculate_metrics_detailed(
        df, true_col=true_col, pred_col=pred_col
    )

    # Format text strings
    metrics_text = (
        f"n = {n:,}\n"
        f"R²: {r2:.2f}\n"
        f"RMSE: {rmse:.2f} W/m²\n"
        f"Bias: {bias:.2f} W/m²\n"
        f"Median: {med_err:.2f} W/m²\n"
        f"STD: {std_err:.2f} W/m²"
    )

    hb = ax.hexbin(
        df[true_col],
        df[pred_col],
        gridsize=100,  # Reduced from 1000 for typical dataset sizes
        cmap='plasma',
        mincnt=1,
        extent=[vmin, vmax, vmin, vmax]
    )

    ax.plot([vmin, vmax], [vmin, vmax], 'k:', linewidth=1.5)

    # Title
    if station_id:
        full_title = f"Station: {station_id} | {title}"
    else:
        full_title = title
    ax.set_title(full_title)

    ax.set_xlabel('Ground Station LST (K)')
    ax.set_ylabel('Predicted LST (K)')

    # Metrics Box
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6)
    )

    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(hb, cax=cax, label='Point Count (Density)')

    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_predictions(model_type):
    """
    Load all prediction files for a given model type and combine them.
    """
    pred_dir = settings.TEMP_PRED_DIR
    all_preds = []
    
    for f in os.listdir(pred_dir):
        if f.startswith('pred_') and f.endswith('.csv'):
            station_id = f.replace('pred_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(pred_dir, f))
            df['station_id'] = station_id
            all_preds.append(df)
    
    if not all_preds:
        raise ValueError(f"No prediction files found in {pred_dir}")
    
    combined = pd.concat(all_preds, ignore_index=True)
    combined['model_type'] = model_type
    return combined


def main():
    # --- Step 1: Define Paths and Create Base Directory ---
    BASE_FIG_DIR = os.path.join(settings.BASE_DIR, 'figures')
    BASE_OUTPUT_DIR = os.path.join(BASE_FIG_DIR, 'densityplots')
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"[Step 1] Base output directory created at: {BASE_OUTPUT_DIR}")

    # --- Step 2: Load ALL prediction data ---
    t1 = time.time()
    
    # Load BLM predictions
    print("Loading BLM predictions...")
    blm_df = load_predictions('BLM')
    print(f"  Loaded {len(blm_df)} BLM prediction rows")
    
    # Load BLHIM predictions (same files since we just trained, 
    # but in practice you'd have separate prediction runs)
    print("Loading BLHIM predictions...")
    blhim_df = load_predictions('BLHIM')
    print(f"  Loaded {len(blhim_df)} BLHIM prediction rows")
    
    print(f"[Step 2] All data loaded in {time.time() - t1:.2f} s")

    # --- Step 3: Get Station List ---
    all_data_df = pd.concat([blm_df, blhim_df])
    station_ids = all_data_df['station_id'].unique()
    station_ids = sorted(station_ids)
    print(f"[Step 3] Found {len(station_ids)} unique station_ids.")

    # --- Step 4: Find GLOBAL color limits ---
    t2 = time.time()
    all_vals = pd.concat([
        all_data_df['true'], all_data_df['pred']
    ])
    vmin, vmax = all_vals.min(), all_vals.max()
    # Add padding
    padding = (vmax - vmin) * 0.05
    vmin -= padding
    vmax += padding
    print(f"[Step 4] Global value ranges computed: {vmin:.2f} to {vmax:.2f} W/m²")
    
    # Free memory
    del all_data_df

    # --- Step 5: Generate GLOBAL plots ---
    t3 = time.time()
    print(f"[Step 5] Generating GLOBAL plots...")

    # Define tasks for each model
    global_tasks_data = {
        # BLM Model
        'BLM (Clear-Sky)': blm_df[blm_df['sky_condition'] == 1],
        'BLM (Cloudy-Sky)': blm_df[blm_df['sky_condition'] == 0],
        'BLM (All-Sky)': blm_df,

        # BLHIM Model
        'BLHIM (Clear-Sky)': blhim_df[blhim_df['sky_condition'] == 1],
        'BLHIM (Cloudy-Sky)': blhim_df[blhim_df['sky_condition'] == 0],
        'BLHIM (All-Sky)': blhim_df,
    }

    for title, df in global_tasks_data.items():
        filename = f"global_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(BASE_OUTPUT_DIR, filename)

        if df.empty:
            print(f"  ... skipping GLOBAL {title} (no data)")
            continue

        print(f"  ... plotting and saving GLOBAL {title}")
        plot_density_fig(df, title, vmin, vmax, save_path, station_id=None)

    print(f"[Step 5] Global plots saved in {time.time() - t3:.2f} s")

    # --- Step 6: Generate plots PER-STATION ---
    t4 = time.time()
    print(f"[Step 6] Starting per-station plot generation...")

    for i, station_id in enumerate(station_ids):
        station_t0 = time.time()
        print(f"\n--- Processing Station {station_id} ({i+1}/{len(station_ids)}) ---")

        station_dir = os.path.join(BASE_OUTPUT_DIR, f"station_{station_id}")
        os.makedirs(station_dir, exist_ok=True)

        blm_stat_df = blm_df[blm_df['station_id'] == station_id]
        blhim_stat_df = blhim_df[blhim_df['station_id'] == station_id]

        station_tasks_data = {
            # BLM
            'BLM (Clear-Sky)': blm_stat_df[blm_stat_df['sky_condition'] == 1],
            'BLM (Cloudy-Sky)': blm_stat_df[blm_stat_df['sky_condition'] == 0],
            'BLM (All-Sky)': blm_stat_df,

            # BLHIM
            'BLHIM (Clear-Sky)': blhim_stat_df[blhim_stat_df['sky_condition'] == 1],
            'BLHIM (Cloudy-Sky)': blhim_stat_df[blhim_stat_df['sky_condition'] == 0],
            'BLHIM (All-Sky)': blhim_stat_df,
        }

        for title, df in station_tasks_data.items():
            filename = f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            save_path = os.path.join(station_dir, filename)

            if df.empty:
                print(f"  ... skipping {title} (no data for this station)")
                continue

            plot_density_fig(df, title, vmin, vmax, save_path, station_id=station_id)

        print(f"--- Station {station_id} finished in {time.time() - station_t0:.2f} s ---")

    print(f"\n[Step 6] All per-station plots saved in {time.time() - t4:.2f} s")
    print(f"\n✅ Total runtime: {time.time() - t0:.2f} s")
    print(f"All plots saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
