"""
Density Plot Analysis for LST Predictions
Generates density plots comparing predicted vs actual LST for all available models.
Reflects logic and style from 02a_densityplots.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# Style settings from 02a_densityplots.py
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def calculate_metrics(df, true_col='true', pred_col='pred'):
    """
    Calculates metrics matching 02a_densityplots.py logic.
    Returns: RMSE, R2, Bias, Mean, STD, n
    """
    if df.empty or len(df) < 2:
        return {'rmse': np.nan, 'r2': np.nan, 'bias': np.nan, 'mean': np.nan, 'std': np.nan, 'n': 0}

    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    error = y_pred - y_true

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'bias': np.mean(error),
        'mean': np.mean(y_true), 
        'error_std': np.std(error),
        'n': len(df)
    }
    return metrics

def plot_density_fig(df, title, vmin, vmax, save_path, true_col='LST_true', pred_col='LST_pred'):
    """
    Creates, saves, and closes a density plot.
    Matches visual style of 02a_densityplots.py.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    metrics = calculate_metrics(df, true_col=true_col, pred_col=pred_col)

    # Metrics Box matches 02a style
    metrics_text = (
        f"n = {metrics['n']:,}\n"
        f"R² = {metrics['r2']:.2f}\n"
        f"RMSE = {metrics['rmse']:.2f} K\n"
        f"Bias = {metrics['bias']:.2f} K\n"
        f"STD = {metrics['error_std']:.2f} K"
    )

    hb = ax.hexbin(
        df[true_col],
        df[pred_col],
        gridsize=100,
        cmap='plasma',
        mincnt=1,
        extent=[vmin, vmax, vmin, vmax]
    )

    # 1:1 Line with label
    ax.plot([vmin, vmax], [vmin, vmax], 'k:', linewidth=1.5, label='1:1 Line')

    # Styling
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Ground Station LST (K)', fontsize=11)
    ax.set_ylabel('Predicted LST (K)', fontsize=11)

    # Metrics Box
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
    )

    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(hb, cax=cax, label='Point Count')

    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_model_predictions():
    """
    Dynamically loads ALL_predictions.csv for each model in specific directories.
    """
    models_data = {}
    
    # Iterate through potential models defined in settings
    for model_name in settings.FEATURE_SETS.keys():
        # Path structure: models/xgb/{MODEL_NAME}/{MODEL_NAME}_ALL_predictions.csv
        pred_path = os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv')
        
        if os.path.exists(pred_path):
            print(f"Loading {model_name} from {pred_path}...")
            df = pd.read_csv(pred_path)
            models_data[model_name] = df
        else:
            print(f"Skipping {model_name} (File not found: {pred_path})")
            
    return models_data

def main():
    parser = argparse.ArgumentParser(description="Generate Density Plots for LST Models")
    parser.add_argument('--per_station', action='store_true', help="Generate density plots for each station individually")
    args = parser.parse_args()

    t0 = time.time()
    
    # --- Step 1: Setup Paths ---
    BASE_FIG_DIR = os.path.join(settings.BASE_DIR, 'figures')
    BASE_OUTPUT_DIR = os.path.join(BASE_FIG_DIR, 'density_plots')
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {BASE_OUTPUT_DIR}")

    # --- Step 2: Load Data ---
    print("Checking for model predictions...")
    models_data = load_model_predictions()
    
    if not models_data:
        print("No prediction files found. Ensure you have run main.py to train models first.")
        return

    # --- Step 3: Compute Global Limits ---
    print("Computing global range...")
    all_mins = []
    all_maxs = []
    for df in models_data.values():
        all_mins.append(min(df['LST_true'].min(), df['LST_pred'].min()))
        all_maxs.append(max(df['LST_true'].max(), df['LST_pred'].max()))
    
    if not all_mins:
        print("No data available to plot.")
        return

    v_min_val = min(all_mins)
    v_max_val = max(all_maxs)
    
    padding = (v_max_val - v_min_val) * 0.05
    vmin = v_min_val - padding
    vmax = v_max_val + padding
    print(f"Global LST range: {vmin:.1f} K to {vmax:.1f} K")

    # --- Step 4: Generate Global Plots ---
    # BCM convention: 0 = clear-sky, 1 = cloudy-sky
    sky_conditions = {'Clear-Sky': 0, 'Cloudy-Sky': 1, 'All-Sky': None}
    
    # Store summary metrics
    summary_results = []

    print("\nGenerating GLOBAL density plots...")
    
    for model_name, df in models_data.items():
        print(f"Processing Model: {model_name}")
        
        # Create model subdirectory
        model_out_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        for sky_name, sky_val in sky_conditions.items():
            # Filter Data
            if sky_val is not None:
                subset = df[df['ACMC_BCM'] == sky_val]
            else:
                subset = df
                
            if subset.empty:
                print(f"  Skipping {sky_name} (no data)")
                continue
                
            # Define Filename and Path
            filename = f"{model_name}_{sky_name.lower().replace('-', '_')}.png"
            save_path = os.path.join(model_out_dir, filename)
            
            # Plot
            plot_density_fig(
                subset, 
                f"{model_name} ({sky_name})", 
                vmin, vmax, 
                save_path,
                true_col='LST_true', 
                pred_col='LST_pred'
            )
            
            # Calculate metrics for summary CSV
            m = calculate_metrics(subset, true_col='LST_true', pred_col='LST_pred')
            summary_results.append({
                'Model': model_name,
                'Sky Condition': sky_name,
                'n': m['n'],
                'R2': m['r2'],
                'RMSE': m['rmse'],
                'Bias': m['bias'],
                'STD': m['error_std']
            })
            
    # --- Step 5: Save Summary CSV ---
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(BASE_OUTPUT_DIR, 'metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nGlobal metrics summary saved to: {summary_path}")

    # --- Step 6: Per-Station Plots (Optional) ---
    if args.per_station:
        print("\n" + "="*50)
        print("GENERATING PER-STATION PLOTS")
        print("="*50)
        
        for model_name, df in models_data.items():
            print(f"\nProcessing Per-Station: {model_name}")
            stations = sorted(df['station_id'].unique())
            
            # Create per-station root dir for this model
            station_root_dir = os.path.join(BASE_OUTPUT_DIR, model_name, 'stations')
            os.makedirs(station_root_dir, exist_ok=True)
            
            count = 0
            for station in stations:
                station_df = df[df['station_id'] == station]
                if station_df.empty:
                    continue
                    
                # Create directory for this specific station
                station_dir = os.path.join(station_root_dir, f"station_{station}")
                os.makedirs(station_dir, exist_ok=True)
                
                # Plot for each sky condition
                for sky_name, sky_val in sky_conditions.items():
                    if sky_val is not None:
                        subset = station_df[station_df['ACMC_BCM'] == sky_val]
                    else:
                        subset = station_df
                        
                    if subset.empty:
                        continue
                        
                    filename = f"{model_name}_{station}_{sky_name.lower().replace('-', '_')}.png"
                    save_path = os.path.join(station_dir, filename)
                    
                    plot_density_fig(
                        subset, 
                        f"{model_name} - Station {station} ({sky_name})", 
                        vmin, vmax, 
                        save_path,
                        true_col='LST_true', 
                        pred_col='LST_pred'
                    )
                count += 1
                if count % 10 == 0:
                    print(f"  ... processed {count} stations")
                    
            print(f"  ✓ {model_name} per-station plots saved to {station_root_dir}")

    print(f"\n✅ Total runtime: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
