"""
Hourly and Daily RMSE Analysis - BLAM vs BLM Comparison.

This module generates publication-ready comparison figures:
1. Hourly Box Plot: Paired box plots comparing BLAM and BLM by local hour (HST)
2. Daily Line Plot: RMSE comparison over each day of year 2024

Author: Generated for LST Analysis
"""

import os
import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

# Matplotlib configuration for publication-quality figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Model colors
MODEL_COLORS = {
    'BLAM': 'steelblue',
    'BLM': 'coral'
}


def load_predictions(model_name: str) -> pl.DataFrame:
    """
    Load model predictions and convert LOCAL_TIME to HST.
    
    The LOCAL_TIME column is in UTC (+00:00). 
    Hawaii Standard Time (HST) is UTC-10.
    """
    prediction_path = os.path.join(
        settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv'
    )
    
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"{model_name} predictions not found at: {prediction_path}")
    
    print(f"Loading {model_name} predictions from: {prediction_path}")
    
    # Load and parse datetime
    df = pl.read_csv(prediction_path)
    
    # Parse LOCAL_TIME as datetime with UTC timezone
    df = df.with_columns(
        pl.col('LOCAL_TIME').str.to_datetime('%Y-%m-%d %H:%M:%S%z').alias('datetime_utc')
    )
    
    # Convert to HST (UTC-10) by subtracting 10 hours
    df = df.with_columns(
        (pl.col('datetime_utc') - pl.duration(hours=10)).alias('datetime_hst')
    )
    
    # Extract hour and date components from HST datetime
    df = df.with_columns([
        pl.col('datetime_hst').dt.hour().alias('hour_hst'),
        pl.col('datetime_hst').dt.date().alias('date_hst'),
        pl.col('datetime_hst').dt.ordinal_day().alias('doy_hst'),
    ])
    
    # Calculate error and squared error for RMSE
    df = df.with_columns([
        (pl.col('LST_pred') - pl.col('LST_true')).alias('error'),
        ((pl.col('LST_pred') - pl.col('LST_true')) ** 2).alias('squared_error'),
    ])
    
    print(f"Loaded {len(df):,} observations")
    print(f"Date range (HST): {df['date_hst'].min()} to {df['date_hst'].max()}")
    
    return df


def plot_hourly_comparison_boxplot(df_blam: pl.DataFrame, df_blm: pl.DataFrame, output_dir: str) -> str:
    """
    Create paired box plots comparing BLAM and BLM RMSE by local hour (HST).
    
    For each hour, shows two side-by-side box plots (BLAM and BLM).
    Secondary axis shows percentage of cloudy observations.
    """
    print("\nGenerating hourly RMSE comparison box plot...")
    
    hours = list(range(24))
    
    # Calculate hourly RMSE for each model
    blam_rmse = []
    blm_rmse = []
    
    for hour in hours:
        blam_errors = df_blam.filter(pl.col('hour_hst') == hour)['squared_error'].to_numpy()
        blm_errors = df_blm.filter(pl.col('hour_hst') == hour)['squared_error'].to_numpy()
        blam_rmse.append(np.sqrt(blam_errors))
        blm_rmse.append(np.sqrt(blm_errors))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Box plot positions
    width = 0.35
    positions_blam = [h - width/2 for h in hours]
    positions_blm = [h + width/2 for h in hours]
    
    # Create paired box plots
    bp_blam = ax.boxplot(
        blam_rmse,
        positions=positions_blam,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'facecolor': MODEL_COLORS['BLAM'], 'alpha': 0.7},
        whiskerprops={'color': 'gray'},
        capprops={'color': 'gray'}
    )
    
    bp_blm = ax.boxplot(
        blm_rmse,
        positions=positions_blm,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'facecolor': MODEL_COLORS['BLM'], 'alpha': 0.7},
        whiskerprops={'color': 'gray'},
        capprops={'color': 'gray'}
    )
    
    # Calculate hourly cloud percentage (use BLAM data, should be same)
    hourly_cloud = df_blam.group_by('hour_hst').agg([
        pl.col('ACMC_BCM').mean().alias('cloud_fraction')
    ]).sort('hour_hst')
    
    cloud_pct = hourly_cloud['cloud_fraction'].to_numpy() * 100
    
    # Create secondary y-axis for cloud percentage
    ax2 = ax.twinx()
    ax2.plot(hours, cloud_pct, 'g--s', linewidth=2, markersize=5, label='Cloud %', alpha=0.8, zorder=10)
    ax2.set_ylabel('Cloud Percentage (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 100)
    
    # Labels and title
    ax.set_xlabel('Local Hour (HST)')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('BLAM vs BLM: Hourly RMSE Distribution (2024)', fontweight='bold')
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}' for h in hours])
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Create legend handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MODEL_COLORS['BLAM'], alpha=0.7, label='BLAM'),
        Patch(facecolor=MODEL_COLORS['BLM'], alpha=0.7, label='BLM'),
        plt.Line2D([0], [0], color='green', linestyle='--', marker='s', label='Cloud %')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    output_path = os.path.join(output_dir, 'BLAM_vs_BLM_hourly_rmse_boxplot.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def plot_daily_comparison_lineplot(df_blam: pl.DataFrame, df_blm: pl.DataFrame, output_dir: str) -> str:
    """
    Create a line plot comparing BLAM and BLM daily RMSE throughout 2024.
    Secondary axis shows percentage of cloudy observations.
    """
    print("\nGenerating daily RMSE comparison line plot...")
    
    # Calculate daily RMSE and cloud percentage for BLAM
    daily_blam = df_blam.group_by('date_hst').agg([
        pl.col('squared_error').mean().sqrt().alias('rmse'),
        pl.col('ACMC_BCM').mean().alias('cloud_fraction'),
    ]).sort('date_hst')
    
    # Calculate daily RMSE for BLM
    daily_blm = df_blm.group_by('date_hst').agg([
        pl.col('squared_error').mean().sqrt().alias('rmse'),
    ]).sort('date_hst')
    
    dates_blam = daily_blam['date_hst'].to_list()
    rmse_blam = daily_blam['rmse'].to_numpy()
    cloud_pct = daily_blam['cloud_fraction'].to_numpy() * 100
    
    dates_blm = daily_blm['date_hst'].to_list()
    rmse_blm = daily_blm['rmse'].to_numpy()
    
    # Convert dates to matplotlib format
    import datetime
    dates_blam_dt = [datetime.date(d.year, d.month, d.day) for d in dates_blam]
    dates_blm_dt = [datetime.date(d.year, d.month, d.day) for d in dates_blm]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot RMSE lines for both models
    ax.plot(dates_blam_dt, rmse_blam, color=MODEL_COLORS['BLAM'], linewidth=1.5, alpha=0.8, label='BLAM')
    ax.plot(dates_blm_dt, rmse_blm, color=MODEL_COLORS['BLM'], linewidth=1.5, alpha=0.8, label='BLM')
    
    # Add rolling mean for each model
    window_size = 7
    if len(rmse_blam) >= window_size:
        rolling_blam = np.convolve(rmse_blam, np.ones(window_size)/window_size, mode='valid')
        rolling_blm = np.convolve(rmse_blm, np.ones(window_size)/window_size, mode='valid')
        rolling_dates = dates_blam_dt[window_size-1:]
        ax.plot(rolling_dates, rolling_blam, color='darkblue', linewidth=2.5, 
                label=f'BLAM {window_size}-Day Avg', linestyle='-')
        ax.plot(rolling_dates, rolling_blm, color='darkred', linewidth=2.5, 
                label=f'BLM {window_size}-Day Avg', linestyle='-')
    
    # Create secondary y-axis for cloud percentage
    ax2 = ax.twinx()
    ax2.fill_between(dates_blam_dt, 0, cloud_pct, color='gray', alpha=0.2, label='Cloud %')
    ax2.set_ylabel('Cloud Percentage (%)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 100)
    
    # Format x-axis with month labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    
    # Labels and title
    ax.set_xlabel('Date (2024)')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('BLAM vs BLM: Daily RMSE Throughout 2024 (HST)', fontweight='bold')
    ax.grid(axis='both', alpha=0.3)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2)
    
    # Add statistics annotation
    mean_blam = np.mean(rmse_blam)
    mean_blm = np.mean(rmse_blm)
    mean_cloud = np.mean(cloud_pct)
    ax.text(
        0.02, 0.98, 
        f'BLAM Mean: {mean_blam:.2f} K\nBLM Mean: {mean_blm:.2f} K\nCloud: {mean_cloud:.1f}%',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    # Save figure
    output_path = os.path.join(output_dir, 'BLAM_vs_BLM_daily_rmse_lineplot.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate comparison figures for BLAM and BLM."""
    print("=" * 60)
    print("BLAM vs BLM: Hourly and Daily RMSE Comparison")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(settings.FIGURES_DIR, 'rmse_temporal')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data for both models
    df_blam = load_predictions('BLAM')
    df_blm = load_predictions('BLM')
    
    # Generate comparison figures
    hourly_path = plot_hourly_comparison_boxplot(df_blam, df_blm, output_dir)
    daily_path = plot_daily_comparison_lineplot(df_blam, df_blm, output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Generated figures:")
    print(f"  1. {hourly_path}")
    print(f"  2. {daily_path}")
    print("=" * 60)
    
    return [hourly_path, daily_path]


if __name__ == "__main__":
    main()
