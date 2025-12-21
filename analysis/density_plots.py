"""
Density Plot Analysis for LST Predictions - Mobile-First Refactor
Generates a consolidated 2x4 density plot grid (Models x Conditions) for publication.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, r2_score
import time
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# --- configuration ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 14  # Larger base font for scaling
plt.rcParams['axes.linewidth'] = 1.0

def calculate_metrics(df, true_col='LST_true', pred_col='LST_pred'):
    """
    Calculates metrics.
    Returns: RMSE, R2, Bias, Mean, STD, n
    """
    if df.height < 2:
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
        'n': df.height
    }
    return metrics

def plot_density_grid(models_data, output_dir):
    """
    Generates a 2 x N grid of density plots.
    Row 0: Clear-Sky
    Row 1: Cloudy-Sky
    Cols: Models (sorted)
    """
    models = sorted(models_data.keys())
    n_models = len(models)
    
    if n_models == 0:
        print("No models to plot.")
        return

    # Conditions
    # User Spec: Row 0 = Clear-Sky (0), Row 1 = Cloudy-Sky (1)
    conditions = [
        ('Clear-Sky', 0),
        ('Cloudy-Sky', 1)
    ]
    
    # 1. Calculate Global Limits per Model or Global for all?
    # Usually "Shared-Estate" implies shared axes for comparison. 
    # Let's compute global min/max across ALL data for a unified scale.
    all_vals_true = []
    all_vals_pred = []
    for df in models_data.values():
        all_vals_true.append(df['LST_true'])
        all_vals_pred.append(df['LST_pred'])
    
    if not all_vals_true:
        return

    # Concat all series to find global min/max
    # Polars concat requires DataFrames or Series
    all_concat = pl.concat(all_vals_true + all_vals_pred)
    g_min = all_concat.min()
    g_max = all_concat.max()
    
    # Add padding
    pad = (g_max - g_min) * 0.05
    vmin = g_min - pad
    vmax = g_max + pad
    
    # Square grid size
    figsize = (16, 8) # High res source, will rely on DPI scaling. 
    
    fig, axes = plt.subplots(2, n_models, figsize=figsize, sharex=True, sharey=True)
    # Adjust spacing to 0
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Ensure axes is 2D array even if n_models=1
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_models > 1 and len(axes.shape) == 1: # Should be (2, N)
        axes = axes.reshape(2, n_models)
        
    # Colormap
    cmap = 'plasma'
    
    # Iterate
    combined_hb = None # To store one hexbin for colorbar (ideally from all data, but per-plot is standard for density)
    
    for col_idx, model in enumerate(models):
        df = models_data[model]
        
        for row_idx, (cond_name, cond_val) in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            
            # Filter Data
            if cond_val is not None:
                subset = df.filter(pl.col('ACMC_BCM') == cond_val)
            else:
                subset = df
            
            metrics = calculate_metrics(subset)
            
            # Plot Hexbin
            if subset.height > 0:
                hb = ax.hexbin(
                    subset['LST_true'].to_numpy(), 
                    subset['LST_pred'].to_numpy(), 
                    gridsize=50, 
                    cmap=cmap, 
                    mincnt=1,
                    extent=[vmin, vmax, vmin, vmax],
                    linewidths=0
                )
                combined_hb = hb # Keep reference
                
                # 1:1 Line
                ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=0.5, alpha=0.7)
                
                # Stats Annotation (Bottom Right)
                stats_text = (
                    f"$R^2$ = {metrics['r2']:.2f}\n"
                    f"RMSE = {metrics['rmse']:.2f}\n"
                    f"Bias = {metrics['bias']:.2f}\n"
                    f"STD = {metrics['error_std']:.2f}"
                )
                ax.text(
                    0.95, 0.05, stats_text, 
                    transform=ax.transAxes, 
                    fontsize=20, 
                    ha='right', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2)
                )
            
            # Label overlay (Top Left or Right)
            label_text = f"{model}\n{cond_name}"
            ax.text(
                0.05, 0.95, label_text, 
                transform=ax.transAxes, 
                fontsize=16, fontweight='bold',
                ha='left', va='top'
            )
            
            # Formatting
            ax.set_aspect('equal')
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            
            # Ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            
            # Remove tick labels for internal plots
            if row_idx < len(conditions) - 1:
                ax.set_xticklabels([])
            if col_idx > 0:
                ax.set_yticklabels([])
                
            # Axis Labels
            if row_idx == len(conditions) - 1: # Bottom row
                pass 
                
    # Global Axis Labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Ground Station LST (K)', fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel('Predicted LST (K)', fontsize=16)

    # Colorbar
    if combined_hb:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        cbar = fig.colorbar(combined_hb, cax=cbar_ax)
        cbar.set_label('Points ($x10^3$)', fontsize=14)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}'))
    
    # Save
    save_path = os.path.join(output_dir, 'density_plot_grid_mobile_first.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison Grid saved to: {save_path}")

def load_data():
    """
    Loads data for models.
    """
    models_data = {}
    for model_name in settings.FEATURE_SETS.keys():
        pred_path = os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv')
        if os.path.exists(pred_path):
            print(f"Loading {model_name}...")
            # Use polars to read csv
            df = pl.read_csv(pred_path)
            models_data[model_name] = df
        else:
            print(f"Warning: {pred_path} not found.")
    return models_data

def main():
    print("Starting Density Plot Refactor...")
    
    # Setup Output
    output_dir = os.path.join(settings.BASE_DIR, 'figures', 'density_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    models_data = load_data()
    
    if not models_data:
        print("No model data found.")
        return
        
    # Generate Grid
    plot_density_grid(models_data, output_dir)
    
    # Create Summary CSV
    summary = []
    conditions = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
    for model, df in models_data.items():
        for cond_name, cond_val in conditions:
            subset = df.filter(pl.col('ACMC_BCM') == cond_val)
            m = calculate_metrics(subset)
            summary.append({
                'Model': model,
                'Condition': cond_name,
                'R2': m['r2'],
                'RMSE': m['rmse'],
                'Bias': m['bias'],
                'n': m['n']
            })
    
    pl.DataFrame(summary).write_csv(os.path.join(output_dir, 'metrics_summary.csv'))
    print("Summary metrics saved.")

if __name__ == "__main__":
    main()
