"""
Monthly Analysis Figures for LST Model Performance.

This module generates publication-ready figures showing monthly metrics:
1. RMSE by Month (K)
2. R² Score by Month  
3. Bias by Month (K)
4. Sample Count (N)
5. Active Stations
6. Cloudiness Proportion

Supports comparison of multiple models (e.g., BLM vs BLAM) with
clear-sky and cloudy-sky stratification.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Optional, Tuple


# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# Color and style definitions
MODEL_STYLES = {
    'BLAM Clear-Sky': {'color': 'darkblue', 'marker': '^', 'linestyle': '-'},
    'BLAM Cloudy': {'color': 'darkgray', 'marker': '^', 'linestyle': '-'},
    'BLM Clear-Sky': {'color': 'cornflowerblue', 'marker': 'o', 'linestyle': ':'},
    'BLM Cloudy': {'color': 'lightgray', 'marker': 'o', 'linestyle': ':'},
    'BLAM-C Clear-Sky': {'color': 'darkgreen', 'marker': 's', 'linestyle': '-'},
    'BLAM-C Cloudy': {'color': 'lightgreen', 'marker': 's', 'linestyle': '-'},
    'CIM Clear-Sky': {'color': 'darkred', 'marker': 'D', 'linestyle': '--'},
    'CIM Cloudy': {'color': 'lightcoral', 'marker': 'D', 'linestyle': '--'},
    'CIAM Clear-Sky': {'color': 'purple', 'marker': 'v', 'linestyle': '--'},
    'CIAM Cloudy': {'color': 'plum', 'marker': 'v', 'linestyle': '--'},
}

CONDITION_STYLES = {
    'All-Sky': {'color': 'black', 'marker': 's'},
    'Clear-Sky': {'color': 'darkblue', 'marker': 's'},
    'Cloudy': {'color': 'darkgray', 'marker': 's'},
}


def load_predictions(path: str) -> pd.DataFrame:
    """Load prediction CSV and prepare for analysis."""
    df = pd.read_csv(path)
    
    # Ensure required columns exist
    required = ['LOCAL_TIME', 'LST_true', 'LST_pred']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse time and extract month info
    df['LOCAL_TIME'] = pd.to_datetime(df['LOCAL_TIME'])
    df['Month'] = df['LOCAL_TIME'].dt.month_name()
    df['Month_Num'] = df['LOCAL_TIME'].dt.month
    
    return df


def calculate_metrics(group: pd.DataFrame) -> pd.Series:
    """Calculate RMSE, R², and Bias for a group of predictions."""
    g = group.dropna(subset=['LST_true', 'LST_pred'])
    
    if len(g) < 2:
        return pd.Series({
            'RMSE': np.nan, 'R2': np.nan, 'Bias': np.nan,
            'Count': 0, 'Active_Stations': 0
        })
    
    y_true = g['LST_true']
    y_pred = g['LST_pred']
    
    return pd.Series({
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Bias': (y_pred - y_true).mean(),
        'Count': len(g),
        'Active_Stations': g['station_id'].nunique() if 'station_id' in g.columns else np.nan
    })


def compute_monthly_metrics(
    df: pd.DataFrame,
    model_name: str,
    cloud_col: str = 'ACMC_BCM'
) -> pd.DataFrame:
    """
    Compute monthly performance metrics stratified by cloud condition.
    
    Args:
        df: DataFrame with predictions
        model_name: Label for this model
        cloud_col: Column containing cloud mask (0=clear, 1=cloudy)
        
    Returns:
        DataFrame with monthly metrics by condition
    """
    results = []
    
    # Define sky conditions
    conditions = {
        'Clear-Sky': df[df[cloud_col] == 0],
        'Cloudy': df[df[cloud_col] != 0]
    }
    
    for condition, data in conditions.items():
        if len(data) == 0:
            continue
            
        stats = data.groupby(['Month_Num', 'Month']).apply(
            calculate_metrics, include_groups=False
        ).reset_index()
        stats['Condition'] = condition
        stats['Model'] = model_name
        stats['Style_Key'] = f"{model_name} {condition}"
        results.append(stats)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_data_statistics(
    df: pd.DataFrame,
    cloud_col: str = 'ACMC_BCM'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute sample counts and cloudiness proportion by month.
    
    Returns:
        Tuple of (stats_df, proportion_df)
    """
    # All-sky statistics
    all_sky = df.groupby(['Month_Num', 'Month']).apply(
        lambda x: pd.Series({
            'Count': len(x),
            'Active_Stations': x['station_id'].nunique() if 'station_id' in x.columns else np.nan
        }), include_groups=False
    ).reset_index()
    all_sky['Condition'] = 'All-Sky'
    
    # Clear/Cloudy statistics
    conditions = {
        'Clear-Sky': df[df[cloud_col] == 0],
        'Cloudy': df[df[cloud_col] != 0]
    }
    
    split_stats = []
    for cond, data in conditions.items():
        if len(data) == 0:
            continue
        stats = data.groupby(['Month_Num', 'Month']).apply(
            lambda x: pd.Series({
                'Count': len(x),
                'Active_Stations': x['station_id'].nunique() if 'station_id' in x.columns else np.nan
            }), include_groups=False
        ).reset_index()
        stats['Condition'] = cond
        split_stats.append(stats)
    
    df_stats = pd.concat([all_sky] + split_stats, ignore_index=True).sort_values('Month_Num')
    
    # Calculate cloudiness proportion
    pivot = df_stats.pivot(index=['Month_Num', 'Month'], columns='Condition', values='Count').reset_index()
    if 'All-Sky' in pivot.columns and 'Cloudy' in pivot.columns:
        pivot['Cloudy_Proportion'] = pivot['Cloudy'] / pivot['All-Sky']
    else:
        pivot['Cloudy_Proportion'] = np.nan
    
    return df_stats, pivot


def plot_performance_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    show_legend: bool = False
) -> None:
    """Plot a single performance metric line chart."""
    
    # Get unique style keys
    style_keys = df['Style_Key'].unique()
    
    for key in style_keys:
        style = MODEL_STYLES.get(key, {'color': 'black', 'marker': 'o', 'linestyle': '-'})
        data = df[df['Style_Key'] == key].sort_values('Month_Num')
        
        ax.plot(
            data['Month'], data[metric],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=10,
            linewidth=2.5,
            label=key
        )
    
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def generate_all_figures(
    model_predictions: Dict[str, str],
    output_dir: str,
    cloud_col: str = 'ACMC_BCM'
) -> List[str]:
    """
    Generate all monthly analysis figures.
    
    Args:
        model_predictions: Dict mapping model name to prediction CSV path
        output_dir: Directory to save figures
        cloud_col: Column containing cloud mask
        
    Returns:
        List of saved figure paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # Load and process all models
    all_metrics = []
    reference_df = None  # For data statistics (model-independent)
    
    for model_name, path in model_predictions.items():
        print(f"Processing {model_name} from {path}...")
        df = load_predictions(path)
        
        if reference_df is None:
            reference_df = df
        
        metrics = compute_monthly_metrics(df, model_name, cloud_col)
        all_metrics.append(metrics)
    
    if not all_metrics:
        print("No metrics computed. Check input files.")
        return []
    
    df_perf = pd.concat(all_metrics, ignore_index=True).sort_values('Month_Num')
    df_stats, df_proportion = compute_data_statistics(reference_df, cloud_col)
    
    # Create custom legend handles
    legend_handles = []
    for key in df_perf['Style_Key'].unique():
        style = MODEL_STYLES.get(key, {'color': 'black', 'marker': 'o', 'linestyle': '-'})
        handle = plt.Line2D(
            [0], [0],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=10,
            linewidth=2.5,
            label=key
        )
        legend_handles.append(handle)
    
    # --- Plot 1: RMSE ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_performance_metric(ax, df_perf, 'RMSE', 'RMSE (K) by Month')
    ax.legend(handles=legend_handles, title='Model & Condition', loc='best')
    path = os.path.join(output_dir, 'monthly_rmse.png')
    fig.savefig(path)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    # --- Plot 2: R² Score ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_performance_metric(ax, df_perf, 'R2', 'R² Score by Month')
    # Dynamic y-axis
    ymin, ymax = df_perf['R2'].min(), df_perf['R2'].max()
    pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
    ax.set_ylim(max(0, ymin - pad), min(1.01, ymax + pad))
    ax.legend(handles=legend_handles, title='Model & Condition', loc='best')
    path = os.path.join(output_dir, 'monthly_r2.png')
    fig.savefig(path)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    # --- Plot 3: Bias ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_performance_metric(ax, df_perf, 'Bias', 'Bias (K) by Month')
    ax.axhline(0, color='black', linewidth=1)
    ax.legend(handles=legend_handles, title='Model & Condition', loc='best')
    path = os.path.join(output_dir, 'monthly_bias.png')
    fig.savefig(path)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    # --- Plot 4: Sample Count ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for cond in ['All-Sky', 'Clear-Sky', 'Cloudy']:
        data = df_stats[df_stats['Condition'] == cond].sort_values('Month_Num')
        if len(data) == 0:
            continue
        style = CONDITION_STYLES.get(cond, {'color': 'black', 'marker': 's'})
        ax.plot(
            data['Month'], data['Count'],
            color=style['color'], marker=style['marker'],
            markersize=10, linewidth=2.5, label=cond
        )
    ax.set_title('Sample Count (N)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Condition', loc='best')
    path = os.path.join(output_dir, 'monthly_count.png')
    fig.savefig(path)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    # --- Plot 5: Active Stations ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for cond in ['All-Sky', 'Clear-Sky', 'Cloudy']:
        data = df_stats[df_stats['Condition'] == cond].sort_values('Month_Num')
        if len(data) == 0 or data['Active_Stations'].isna().all():
            continue
        style = CONDITION_STYLES.get(cond, {'color': 'black', 'marker': 's'})
        ax.plot(
            data['Month'], data['Active_Stations'],
            color=style['color'], marker=style['marker'],
            markersize=10, linewidth=2.5, label=cond
        )
    ax.set_title('Active Stations')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Condition', loc='best')
    path = os.path.join(output_dir, 'monthly_stations.png')
    fig.savefig(path)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    # --- Plot 6: Cloudiness Proportion ---
    fig, ax = plt.subplots(figsize=(10, 8))
    df_prop_sorted = df_proportion.sort_values('Month_Num')
    ax.plot(
        df_prop_sorted['Month'], df_prop_sorted['Cloudy_Proportion'],
        color='black', marker='s', markersize=10, linewidth=2.5
    )
    ax.set_title('Cloudiness Proportion')
    ax.set_ylabel('Proportion Cloudy')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    path = os.path.join(output_dir, 'monthly_cloudiness.png')
    fig.savefig(path)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    print(f"\n✅ Generated {len(saved_files)} figures in {output_dir}/")
    return saved_files


if __name__ == '__main__':
    import argparse
    import sys
    
    # Add project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.settings import OUTPUT_DIR, BASE_DIR
    
    parser = argparse.ArgumentParser(description='Generate monthly analysis figures')
    parser.add_argument('--models', '-m', nargs='+', 
                        help='Model names to analyze (e.g., BLM BLAM)')
    parser.add_argument('--output', '-o', 
                        default=os.path.join(BASE_DIR, 'figures', 'monthly_analysis'),
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Build model paths dictionary
    # Expected structure: models/xgb/{model_name}/loso_ALL_predictions_for_plotting.csv
    model_predictions = {}
    
    if args.models:
        for model in args.models:
            pred_path = os.path.join(OUTPUT_DIR, 'xgb', model, 'loso_ALL_predictions_for_plotting.csv')
            if os.path.exists(pred_path):
                model_predictions[model] = pred_path
            else:
                print(f"Warning: Predictions not found for {model} at {pred_path}")
    else:
        # Auto-discover models
        xgb_dir = os.path.join(OUTPUT_DIR, 'xgb')
        if os.path.exists(xgb_dir):
            for model_name in os.listdir(xgb_dir):
                pred_path = os.path.join(xgb_dir, model_name, 'loso_ALL_predictions_for_plotting.csv')
                if os.path.exists(pred_path):
                    model_predictions[model_name] = pred_path
    
    if not model_predictions:
        print("No model predictions found. Please specify paths or ensure predictions exist.")
        sys.exit(1)
    
    print(f"Found {len(model_predictions)} models: {list(model_predictions.keys())}")
    
    generate_all_figures(model_predictions, args.output)
