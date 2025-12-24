"""
Density Plot Analysis for LST Predictions
Generates:
1. Individual plots (Clear, Cloudy, All) per model.
2. Standard Grid (Rows: Clear, Cloudy; Cols: Models)
3. Compact Grid (Rows: Clear, Cloudy; Cols: Models)
4. BLAM-C Compact Grid (1 Row x 2 Cols: Clear, Cloudy)
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import string

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# --- configuration ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 25
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Define specific models and order for the grids
GRID_MODELS = ['BLM', 'BLAM', 'CIM', 'CIAM']

# Model Title Mappings
MODEL_TITLES = {
    'BLM': 'BLM',
    'BLAM': r'BLAM' + '\n' + r'$\mathit{(BLM + AEFE)}$',
    'CIM': 'CIM',
    'CIAM': r'CIAM' + '\n' + r'$\mathit{(CIM + AEFE)}$',
    'BLAM-C': r'BLAM-C' + '\n' + r'$\mathit{(BLM + AEFE - CIM)}$'
}

def calculate_metrics(df, true_col='LST_true', pred_col='LST_pred'):
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

def get_global_limits(models_data):
    all_vals = []
    for df in models_data.values():
        all_vals.append(df['LST_true'])
        all_vals.append(df['LST_pred'])
    
    if not all_vals:
        return 0, 1
        
    all_concat = pl.concat(all_vals)
    g_min = all_concat.min()
    g_max = all_concat.max()
    
    pad = (g_max - g_min) * 0.05
    return g_min - pad, g_max + pad

def plot_density_hexbin(ax, subset, vmin, vmax, cmap='plasma', use_log=False, norm=None):
    """
    Helper to plot the hexbin and 1:1 line on a given axis.
    Returns the hexbin object.
    """
    if subset.height > 0:
        hb = ax.hexbin(
            subset['LST_true'].to_numpy(), 
            subset['LST_pred'].to_numpy(), 
            gridsize=500, 
            cmap=cmap, 
            mincnt=1,
            extent=[vmin, vmax, vmin, vmax],
            linewidths=0,
            norm=norm
        )
        # 1:1 Line
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=0.5, alpha=0.7)
        return hb
    return None

def add_stats_text(ax, metrics, fontsize=20, loc='lower right'):
    stats_text = (
        f"$R^2$ = {metrics['r2']:.2f}\n"
        f"RMSE = {metrics['rmse']:.2f}\n"
        f"Bias = {metrics['bias']:.2f}\n"
        f"STD = {metrics['error_std']:.2f}"
    )
    
    if loc == 'lower right':
        x_pos, y_pos = 0.95, 0.05
        ha, va = 'right', 'bottom'
    
    ax.text(
        x_pos, y_pos, stats_text, 
        transform=ax.transAxes, 
        fontsize=fontsize, 
        ha=ha, va=va,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2)
    )

def add_subpanel_label(ax, label):
    ax.text(
        0.05, 0.05, label, 
        transform=ax.transAxes, 
        fontsize=24, fontweight='bold',
        ha='left', va='bottom',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='square,pad=0.2')
    )

# --- Function 1: Three separate density plots per model ---
def plot_individual_model_conditions(models_data, output_dir):
    print("Generating individual plots...")
    save_dir = os.path.join(output_dir, 'individual')
    os.makedirs(save_dir, exist_ok=True)
    
    vmin, vmax = get_global_limits(models_data)
    conditions = [('Clear-Sky', 0), ('Cloudy-Sky', 1), ('All-Sky', None)]
    
    for model_name, df in models_data.items():
        display_name = MODEL_TITLES.get(model_name, model_name)
        
        for cond_name, cond_val in conditions:
            if cond_val is not None:
                subset = df.filter(pl.col('ACMC_BCM') == cond_val)
            else:
                subset = df
            
            if subset.height == 0:
                continue
                
            metrics = calculate_metrics(subset)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            hb = plot_density_hexbin(ax, subset, vmin, vmax)
            
            title_text = f"{display_name}\n{cond_name}"
            ax.text(0.05, 0.95, title_text, transform=ax.transAxes, fontsize=20, fontweight='bold', ha='left', va='top')
            
            if model_name == 'BLAM-C':
                if cond_name == 'Clear-Sky':
                    add_subpanel_label(ax, 'a')
                elif cond_name == 'Cloudy-Sky':
                    add_subpanel_label(ax, 'b')
            
            add_stats_text(ax, metrics, fontsize=22)
            
            ax.set_aspect('equal')
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel('Ground Station LST (K)')
            ax.set_ylabel('Predicted LST (K)')
            
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Point Count')
            
            filename = f"{model_name}_{cond_name.replace('-', '_').lower()}.jpg"
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)

# --- Function 2: Standard Grid (2x4) ---
def plot_standard_grid(models_data, output_dir):
    print("Generating standard grid (2x4)...")
    
    models = sorted([m for m in GRID_MODELS if m in models_data], key=lambda x: GRID_MODELS.index(x))
    
    if len(models) == 0:
        print("No matching models for grid.")
        return

    conditions = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
    vmin, vmax = get_global_limits(models_data)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=True, sharey=True)
    
    labels = list(string.ascii_lowercase)
    label_idx = 0
    
    for row_idx, (cond_name, cond_val) in enumerate(conditions):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            df = models_data[model]
            display_name = MODEL_TITLES.get(model, model)
            
            subset = df.filter(pl.col('ACMC_BCM') == cond_val)
            
            metrics = calculate_metrics(subset)
            hb = plot_density_hexbin(ax, subset, vmin, vmax)
            
            label_text = f"{display_name}\n{cond_name}"
            ax.text(0.05, 0.95, label_text, transform=ax.transAxes, fontsize=20, fontweight='bold', ha='left', va='top')
            
            add_subpanel_label(ax, f"{labels[label_idx]}")
            label_idx += 1
            
            add_stats_text(ax, metrics, fontsize=18)
            
            ax.set_aspect('equal')
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)

    for ax in axes[-1, :]:
        ax.set_xlabel('Ground Station LST (K)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Predicted LST (K)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_grid_standard.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- Function 3: Compact Grid (Publication Ready) 2x4 ---
def plot_2x4_compact_publication(models_data, output_dir):
    """
    2 rows x 4 cols with shared colorbar. 
    X ticks visible on bottom row, Y ticks on left column.
    Centered axis labels.
    """
    print("Generating 2x4 compact grid...")
    
    models = sorted([m for m in GRID_MODELS if m in models_data], key=lambda x: GRID_MODELS.index(x))
    
    if len(models) == 0:
        print("No matching models for grid.")
        return

    conditions = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
    vmin, vmax = get_global_limits(models_data)
    
    # First pass: collect all hexbin data to find global color limits
    all_counts = []
    for model in models:
        df = models_data[model]
        for cond_name, cond_val in conditions:
            subset = df.filter(pl.col('ACMC_BCM') == cond_val)
            if subset.height > 0:
                # Create a temporary hexbin to get counts
                fig_temp, ax_temp = plt.subplots()
                hb_temp = ax_temp.hexbin(
                    subset['LST_true'].to_numpy(), 
                    subset['LST_pred'].to_numpy(), 
                    gridsize=500, mincnt=1,
                    extent=[vmin, vmax, vmin, vmax]
                )
                counts = hb_temp.get_array()
                all_counts.extend(counts)
                plt.close(fig_temp)
    
    # Global color limits
    c_min = 1
    c_max = max(all_counts) if all_counts else 1
    
    # Use log norm for better visualization
    norm = LogNorm(vmin=c_min, vmax=c_max)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    plt.subplots_adjust(wspace=0.05, hspace=0.08, bottom=0.12, left=0.08, right=0.88)
    
    labels = list(string.ascii_lowercase)
    label_idx = 0
    stats_fontsize = int(16 * 1.2)
    
    combined_hb = None
    
    for row_idx, (cond_name, cond_val) in enumerate(conditions):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            df = models_data[model]
            display_name = MODEL_TITLES.get(model, model)
            
            subset = df.filter(pl.col('ACMC_BCM') == cond_val)
            metrics = calculate_metrics(subset)
            
            hb = plot_density_hexbin(ax, subset, vmin, vmax, norm=norm)
            if hb: combined_hb = hb
            
            ax.text(0.05, 0.95, f"{display_name}\n{cond_name}", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
            
            add_subpanel_label(ax, f"{labels[label_idx]}")
            label_idx += 1
            
            add_stats_text(ax, metrics, fontsize=stats_fontsize)
            
            ax.set_aspect('equal')
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            
            # X tick labels only on bottom row
            if row_idx == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(labelbottom=True)
            
            # Y tick labels only on left column
            if col_idx > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.tick_params(labelleft=True)

    # Centered Axis Labels
    fig.text(0.48, 0.08, 'Ground Station LST (K)', ha='center', fontsize=25)
    fig.text(0.03, 0.5, 'Predicted LST (K)', va='center', rotation='vertical', fontsize=25)

    # Shared Colorbar
    if combined_hb:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(combined_hb, cax=cbar_ax)
        cbar.set_label('Point Count')
        
    plt.savefig(os.path.join(output_dir, 'density_grid_2x4_compact.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- Function 4: BLAM-C Compact Grid (1x2) ---
def plot_blam_c_2x1_compact(models_data, output_dir):
    """
    BLAM-C 1x2 grid with log scale density and shared colorbar.
    """
    print("Generating BLAM-C 1x2 compact grid...")
    
    if 'BLAM-C' not in models_data:
        print("BLAM-C data not found.")
        return
        
    df = models_data['BLAM-C']
    display_name = MODEL_TITLES.get('BLAM-C', 'BLAM-C')
    
    conditions = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
    vmin, vmax = get_global_limits(models_data)
    
    # First pass: collect all hexbin data to find global color limits
    all_counts = []
    for cond_name, cond_val in conditions:
        subset = df.filter(pl.col('ACMC_BCM') == cond_val)
        if subset.height > 0:
            fig_temp, ax_temp = plt.subplots()
            hb_temp = ax_temp.hexbin(
                subset['LST_true'].to_numpy(), 
                subset['LST_pred'].to_numpy(), 
                gridsize=500, mincnt=1,
                extent=[vmin, vmax, vmin, vmax]
            )
            counts = hb_temp.get_array()
            all_counts.extend(counts)
            plt.close(fig_temp)
    
    # Global color limits with log norm
    c_min = 1
    c_max = max(all_counts) if all_counts else 1
    norm = LogNorm(vmin=c_min, vmax=c_max)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.05, bottom=0.15, left=0.10, right=0.88)
    
    combined_hb = None
    labels = ['a', 'b']
    stats_fontsize = int(16 * 1.2)
    
    for col_idx, (cond_name, cond_val) in enumerate(conditions):
        ax = axes[col_idx]
        
        subset = df.filter(pl.col('ACMC_BCM') == cond_val)
        metrics = calculate_metrics(subset)
        
        hb = plot_density_hexbin(ax, subset, vmin, vmax, norm=norm)
        if hb: combined_hb = hb
        
        ax.text(0.05, 0.95, f"{display_name}\n{cond_name}", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
        
        add_subpanel_label(ax, labels[col_idx])
        
        add_stats_text(ax, metrics, fontsize=stats_fontsize)
        
        ax.set_aspect('equal')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # Y tick labels only on left plot
        if col_idx > 0:
            ax.tick_params(labelleft=False)
        
        # X tick labels visible on both
        ax.tick_params(labelbottom=True)

    # Centered Axis Labels - adjusted positions
    fig.text(0.48, 0.05, 'Ground Station LST (K)', ha='center', fontsize=22)
    fig.text(0.02, 0.5, 'Predicted LST (K)', va='center', rotation='vertical', fontsize=22)

    # Shared Colorbar
    if combined_hb:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
        cbar = fig.colorbar(combined_hb, cax=cbar_ax)
        cbar.set_label('Point Count')
        
    plt.savefig(os.path.join(output_dir, 'density_grid_blam_c_1x2.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_data():
    models_data = {}
    for model_name in settings.FEATURE_SETS.keys():
        pred_path = os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv')
        if os.path.exists(pred_path):
            print(f"Loading {model_name}...")
            df = pl.read_csv(pred_path)
            models_data[model_name] = df
        else:
            print(f"Warning: {pred_path} not found.")
    return models_data

def main():
    print("Starting Density Plot Generation...")
    
    output_dir = os.path.join(settings.BASE_DIR, 'figures', 'density_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    models_data = load_data()
    
    if not models_data:
        print("No model data found.")
        return
        
    plot_individual_model_conditions(models_data, output_dir)
    plot_standard_grid(models_data, output_dir)
    plot_2x4_compact_publication(models_data, output_dir)
    plot_blam_c_2x1_compact(models_data, output_dir)
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
