"""
Paired Dot Plot Visualizations for LST Model Comparisons

Generates 2x4 compact panel of paired dot plots matching density_plots.py style.
Rows: Clear-Sky, Cloudy-Sky
Cols: BLM vs BLAM (RMSE), BLM vs BLAM (STD), CIM vs CIAM (RMSE), CIM vs CIAM (STD)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import string
from scipy import stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings
from analysis.statistical_analysis import load_data, calculate_station_metrics, METRICS, CONDITIONS, COMPARISONS

# --- Configuration (matching density_plots.py template) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 25
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def paired_dot_plot(ax, data1, data2, labels, title, ylabel, p_value, metric_type='RMSE', show_legend=False, label_char=None):
    m1, m2 = np.mean(data1), np.mean(data2)
    n, d1, d2 = len(data1), np.array(data1), np.array(data2)
    colors = np.where(d2 < d1, '#2ecc71', '#e74c3c')
    
    for i in range(n):
        ax.plot([0, 1], [d1[i], d2[i]], color=colors[i], alpha=0.6, linewidth=1.5)
    
    ax.scatter([0]*n, d1, color='#3498db', s=40, zorder=3)
    ax.scatter([1]*n, d2, color='#9b59b6', s=40, zorder=3)
    ax.scatter([0, 1], [m1, m2], color=['#3498db', '#9b59b6'], s=180, marker='D', edgecolor='black', linewidth=2, zorder=5)
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=22)
    
    p_str = (p_value < 0.001 and "p < 0.001") or f"p = {p_value:.4f}"
    ax.set_title(f"{metric_type}: {labels[0]} vs {labels[1]}\nWilcoxon {p_str}", fontsize=16, fontweight='bold', pad=25)
    
    n_imp = np.sum(d2 < d1)
    ax.text(0.22, 1.05, f"Improvement: {n_imp}/{n}", transform=ax.transAxes, color='#1a7f37', fontsize=15, fontweight='bold', ha='center', va='top')
    ax.text(0.73, 1.05, f"No Improvement: {n-n_imp}/{n}", transform=ax.transAxes, color='#b91c1c', fontsize=15, fontweight='bold', ha='center', va='top')
    
    label_char and ax.text(0.05, 0.05, label_char, transform=ax.transAxes, fontsize=24, fontweight='bold', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='square,pad=0.2'))
    
    show_legend and ax.legend(handles=[
        mpatches.Patch(color='#2ecc71', label='Improvement'),
        mpatches.Patch(color='#e74c3c', label='No Improvement'),
        mlines.Line2D([], [], color='white', marker='D', markeredgecolor='black', markerfacecolor='gray', markersize=14, label='Mean')
    ], loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=11)
    
    ax.grid(True, alpha=0.3, axis='y')

def plot_2x4_compact_panel(models_data, output_dir):
    """
    Generate 2x4 compact panel of paired dot plots.
    Rows: Clear-Sky, Cloudy-Sky
    Cols: BLM vs BLAM (RMSE), BLM vs BLAM (STD), CIM vs CIAM (RMSE), CIM vs CIAM (STD)
    """
    print("Generating 2x4 compact paired dot panel...")
    
    fig_dir = os.path.join(output_dir, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Layout: 2 rows (conditions) x 4 cols (comparison+metric combos)
    # Cols: (BLM,BLAM,rmse), (BLM,BLAM,std), (CIM,CIAM,rmse), (CIM,CIAM,std)
    column_configs = [
        (COMPARISONS[0], 'rmse'),  # BLM vs BLAM RMSE
        (COMPARISONS[0], 'std'),   # BLM vs BLAM STD
        (COMPARISONS[1], 'rmse'),  # CIM vs CIAM RMSE
        (COMPARISONS[1], 'std'),   # CIM vs CIAM STD
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(26, 14))
    plt.subplots_adjust(wspace=0.22, hspace=0.25, bottom=0.06, left=0.05, right=0.98, top=0.88)
    
    labels = list(string.ascii_lowercase)
    label_idx = 0
    
    for row_idx, (cond_name, cond_val) in enumerate(CONDITIONS):
        for col_idx, ((m1_name, m2_name), metric) in enumerate(column_configs):
            ax = axes[row_idx, col_idx]
            
            # Calculate station metrics for each model under this condition
            s1 = calculate_station_metrics(models_data[m1_name].filter(
                __import__('polars').col('ACMC_BCM') == cond_val))
            s2 = calculate_station_metrics(models_data[m2_name].filter(
                __import__('polars').col('ACMC_BCM') == cond_val))
            
            # Join on station_id
            joined = s1.join(s2, on='station_id', how='inner', suffix='_m2')
            
            v1 = joined[metric].to_numpy()
            v2 = joined[f'{metric}_m2'].to_numpy()
            _, p_val = stats.wilcoxon(v1, v2)
            
            # Show legend only on top-right (row 0, col 3)
            show_legend = (row_idx == 0 and col_idx == 3)
            
            title = f'{m1_name} vs {m2_name} ({cond_name})'
            ylabel = f'{metric.upper()} (K)'
            metric_type = f'{metric.upper()} ({cond_name})'
            
            paired_dot_plot(ax, v1, v2, [m1_name, m2_name], title, ylabel, p_val, 
                           metric_type=metric_type, show_legend=show_legend, label_char=labels[label_idx])
            label_idx += 1
    
    out_path = os.path.join(fig_dir, 'paired_dot_2x4_compact.jpg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_individual_metrics(models_data, output_dir):
    """Generate individual 2x2 plots for each metric (original format)."""
    print("Generating individual metric plots...")
    
    fig_dir = os.path.join(output_dir, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    import polars as pl
    
    for metric in METRICS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Station-Level {metric.upper()}: Paired Comparisons', fontsize=14, fontweight='bold')
        
        for col, (m1_name, m2_name) in enumerate(COMPARISONS):
            for row, (cond_name, cond_val) in enumerate(CONDITIONS):
                s1 = calculate_station_metrics(models_data[m1_name].filter(pl.col('ACMC_BCM') == cond_val))
                s2 = calculate_station_metrics(models_data[m2_name].filter(pl.col('ACMC_BCM') == cond_val))
                joined = s1.join(s2, on='station_id', how='inner', suffix='_m2')
                
                v1, v2 = joined[metric].to_numpy(), joined[f'{metric}_m2'].to_numpy()
                _, p_val = stats.wilcoxon(v1, v2)
                
                # Legend only for top-right (row 0, col 1)
                show_legend = (row == 0 and col == 1)
                
                paired_dot_plot(axes[row, col], v1, v2, [m1_name, m2_name], 
                               f'{m1_name} vs {m2_name} ({cond_name})', f'{metric.upper()} (K)', 
                               p_val, show_legend=show_legend)
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'paired_dot_{metric}.jpg'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Saved: paired_dot_{metric}.jpg")


def main():
    """Generate all paired dot plot visualizations."""
    print("Loading model data...")
    models_data = load_data()
    
    if not models_data:
        print("No model data found.")
        return
    
    print(f"Loaded models: {list(models_data.keys())}")
    
    output_dir = settings.FIGURES_DIR
    
    # Generate the new 2x4 compact panel
    plot_2x4_compact_panel(models_data, output_dir)
    
    print("\nAll paired dot plots generated.")


if __name__ == "__main__":
    main()
