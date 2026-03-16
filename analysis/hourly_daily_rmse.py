"""
Hourly and Daily RMSE Analysis - SXE vs SX Comparison.

This module generates publication-ready comparison figures:
1. Hourly Box Plot: Paired box plots comparing SXE and SX by local hour (HST)
2. Daily Line Plot: RMSE comparison over each day of year 2024

Author: Generated for LST Analysis. #Left as is. It is true.
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

# Matplotlib configuration (matching density_plots.py template)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 25,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
# Model colors
MODEL_COLORS = {
    'SXE': 'steelblue',
    'SX': 'coral'
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


def plot_hourly_comparison_boxplot(df_sxe: pl.DataFrame, df_sx: pl.DataFrame, output_dir: str) -> str:
    """
    Create paired box plots comparing SXE and SX RMSE by local hour (HST).
    
    For each hour, shows two side-by-side box plots (SXE and SX).
    Secondary axis shows percentage of cloudy observations.
    """
    print("\nGenerating hourly RMSE comparison box plot...")
    
    hours = list(range(24))
    
    # Calculate hourly RMSE for each model
    sxe_rmse = []
    sx_rmse = []
    
    for hour in hours:
        sxe_errors = df_sxe.filter(pl.col('hour_hst') == hour)['squared_error'].to_numpy()
        sx_errors = df_sx.filter(pl.col('hour_hst') == hour)['squared_error'].to_numpy()
        sxe_rmse.append(np.sqrt(sxe_errors))
        sx_rmse.append(np.sqrt(sx_errors))
    
    # Calculate hourly observation counts
    hourly_counts = df_sxe.group_by('hour_hst').agg([
        pl.len().alias('count')
    ]).sort('hour_hst')
    obs_per_hour = hourly_counts['count'].to_list()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Box plot positions
    width = 0.35
    positions_sxe = [h - width/2 for h in hours]
    positions_sx = [h + width/2 for h in hours]
    
    # Create paired box plots
    bp_sxe = ax.boxplot(
        sxe_rmse,
        positions=positions_sxe,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'facecolor': MODEL_COLORS['SXE'], 'alpha': 0.7},
        whiskerprops={'color': 'gray'},
        capprops={'color': 'gray'}
    )
    
    bp_sx = ax.boxplot(
        sx_rmse,
        positions=positions_sx,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'facecolor': MODEL_COLORS['SX'], 'alpha': 0.7},
        whiskerprops={'color': 'gray'},
        capprops={'color': 'gray'}
    )
    
    # Calculate hourly cloud percentage (use SXE data, should be same)
    hourly_cloud = df_sxe.group_by('hour_hst').agg([
        pl.col('ACMC_BCM').mean().alias('cloud_fraction')
    ]).sort('hour_hst')
    
    cloud_pct = hourly_cloud['cloud_fraction'].to_numpy() * 100
    
    # Create secondary y-axis for cloud percentage
    ax2 = ax.twinx()
    ax2.plot(hours, cloud_pct, color='#505050', linestyle='--', marker='s', linewidth=2, markersize=5, label='Cloud %', alpha=0.8, zorder=10)
    ax2.set_ylabel('Cloud Percentage (%)', color='#404040')
    ax2.tick_params(axis='y', labelcolor='#404040')
    ax2.set_ylim(0, 100)
    
    # Labels
    ax.set_xlabel('Local Hour (HST)')
    ax.set_ylabel('RMSE (K)')
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}' for h in hours])
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis to start at 0 and tighten x-axis
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, 23.5)
    
    # Create legend handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MODEL_COLORS['SXE'], alpha=0.7, label='SXE'),
        Patch(facecolor=MODEL_COLORS['SX'], alpha=0.7, label='SX'),
        plt.Line2D([0], [0], color='#505050', linestyle='--', marker='s', label='Cloud %')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=1.0, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'SXE_vs_SX_hourly_rmse_boxplot.jpg')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def plot_daily_comparison_lineplot(df_sxe: pl.DataFrame, df_sx: pl.DataFrame, output_dir: str) -> str:
    """
    Create a line plot comparing SXE and SX daily RMSE throughout 2024.
    Secondary axis shows percentage of cloudy observations.
    """
    print("\nGenerating daily RMSE comparison line plot...")
    
    # Calculate daily RMSE and cloud percentage for SXE
    daily_sxe = df_sxe.group_by('date_hst').agg([
        pl.col('squared_error').mean().sqrt().alias('rmse'),
        pl.col('ACMC_BCM').mean().alias('cloud_fraction'),
    ]).sort('date_hst')
    
    # Calculate daily RMSE for SX
    daily_sx = df_sx.group_by('date_hst').agg([
        pl.col('squared_error').mean().sqrt().alias('rmse'),
    ]).sort('date_hst')
    
    dates_sxe = daily_sxe['date_hst'].to_list()
    rmse_sxe = daily_sxe['rmse'].to_numpy()
    cloud_pct = daily_sxe['cloud_fraction'].to_numpy() * 100
    
    dates_sx = daily_sx['date_hst'].to_list()
    rmse_sx = daily_sx['rmse'].to_numpy()
    
    # Convert dates to matplotlib format
    import datetime
    dates_sxe_dt = [datetime.date(d.year, d.month, d.day) for d in dates_sxe]
    dates_sx_dt = [datetime.date(d.year, d.month, d.day) for d in dates_sx]
    
    # Calculate daily observation counts
    daily_counts = df_sxe.group_by('date_hst').agg([
        pl.len().alias('count')
    ]).sort('date_hst')
    obs_per_day = daily_counts['count'].to_list()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot RMSE lines for both models
    ax.plot(dates_sxe_dt, rmse_sxe, color=MODEL_COLORS['SXE'], linewidth=1.5, alpha=0.8, label='SXE')
    ax.plot(dates_sx_dt, rmse_sx, color=MODEL_COLORS['SX'], linewidth=1.5, alpha=0.8, label='SX')
    
    # Add rolling mean for each model
    window_size = 7
    if len(rmse_sxe) >= window_size:
        rolling_sxe = np.convolve(rmse_sxe, np.ones(window_size)/window_size, mode='valid')
        rolling_sx = np.convolve(rmse_sx, np.ones(window_size)/window_size, mode='valid')
        rolling_dates = dates_sxe_dt[window_size-1:]
        ax.plot(rolling_dates, rolling_sxe, color='darkblue', linewidth=2.5, 
                label=f'SXE {window_size}-Day Avg', linestyle='-')
        ax.plot(rolling_dates, rolling_sx, color='darkred', linewidth=2.5, 
                label=f'SX {window_size}-Day Avg', linestyle='-')
    
    # Create secondary y-axis for cloud percentage (render behind main plot elements)
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)  # Put ax2 behind ax
    ax.patch.set_visible(False)  # Make ax background transparent so ax2 shows through
    ax2.fill_between(dates_sxe_dt, 0, cloud_pct, color='gray', alpha=0.2, label='Cloud %', zorder=0)
    ax2.set_ylabel('Cloud Percentage (%)', color='#404040')
    ax2.tick_params(axis='y', labelcolor='#404040')
    ax2.set_ylim(0, 100)
    
    # Format x-axis with month labels (just month names)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    ax.tick_params(axis='x', labelsize=16)  # Reduce x-axis label size to prevent overlap
    
    # Labels
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean RMSE (K)')
    ax.grid(axis='both', alpha=0.3)
    
    # Set y-axis to start at 0 and tighten x-axis, remove "0" label
    ax.set_ylim(bottom=0)
    ax.set_xlim(dates_sxe_dt[0], dates_sxe_dt[-1])
    # Remove the 0 from y-axis ticks
    yticks = ax.get_yticks()
    ax.set_yticks([t for t in yticks if t > 0])
    
    # Combined legend with pure white opaque background (top right, aligned with stats box)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, 
              framealpha=1.0, facecolor='#ffffff', edgecolor='gray', fontsize=15,
              bbox_to_anchor=(0.975, 0.975), borderaxespad=0)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)  # Ensure legend is on top of everything
    
    # Add statistics annotation (top left, aligned with legend top edge)
    mean_sxe = np.mean(rmse_sxe)
    mean_sx = np.mean(rmse_sx)
    mean_cloud = np.mean(cloud_pct)
    text_box = ax.text(
        0.025, 0.975, 
        f'SXE Mean: {mean_sxe:.2f} K\nSX Mean: {mean_sx:.2f} K\nCloud %: {mean_cloud:.1f}',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=15,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', alpha=1.0, edgecolor='gray'),
        zorder=100  # Ensure text box is on top of everything
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'SXE_vs_SX_daily_rmse_lineplot.jpg')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def plot_rmse_difference(df_sxe: pl.DataFrame, df_sx: pl.DataFrame, output_dir: str) -> str:
    """
    Create a line plot showing (SX) - (SXE) RMSE difference.
    Shows daily difference in yellow and 7-day moving average in purple.
    Negative values indicate SX outperforms SXE (SX has lower RMSE).
    Positive values indicate SXE outperforms SX (SXE has lower RMSE).
    Secondary axis shows cloud cover percentage.
    """
    print("\nGenerating RMSE difference plot (SX minus SXE)...")
    
    # Calculate daily RMSE and cloud percentage for SXE
    daily_sxe = df_sxe.group_by('date_hst').agg([
        pl.col('squared_error').mean().sqrt().alias('rmse'),
        pl.col('ACMC_BCM').mean().alias('cloud_fraction'),
    ]).sort('date_hst')
    
    # Calculate daily RMSE for SX
    daily_sx = df_sx.group_by('date_hst').agg([
        pl.col('squared_error').mean().sqrt().alias('rmse'),
    ]).sort('date_hst')
    
    # Merge on date to ensure alignment
    daily_merged = daily_sxe.join(daily_sx, on='date_hst', suffix='_sx')
    
    dates = daily_merged['date_hst'].to_list()
    rmse_sxe = daily_merged['rmse'].to_numpy()  # SXE
    rmse_sx = daily_merged['rmse_sx'].to_numpy()  # SX
    cloud_pct = daily_merged['cloud_fraction'].to_numpy() * 100
    
    # Calculate difference: (SX) - (SXE)
    # Negative = SX is better (SX has lower RMSE, since lower is better)
    # Positive = SXE is better (SXE has lower RMSE)
    rmse_diff = rmse_sx - rmse_sxe
    
    # Convert dates to matplotlib format
    import datetime
    dates_dt = [datetime.date(d.year, d.month, d.day) for d in dates]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot daily difference in yellow
    ax.plot(dates_dt, rmse_diff, color='#FFD700', linewidth=1.5, alpha=0.8, label='Daily Δ RMSE')
    
    # Add 7-day rolling mean in purple
    window_size = 7
    if len(rmse_diff) >= window_size:
        rolling_diff = np.convolve(rmse_diff, np.ones(window_size)/window_size, mode='valid')
        rolling_dates = dates_dt[window_size-1:]
        ax.plot(rolling_dates, rolling_diff, color='purple', linewidth=2.5, 
                label=f'Δ RMSE {window_size}-Day Avg')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Fill regions to indicate which model is better (corrected interpretation)
    ax.fill_between(dates_dt, rmse_diff, 0, where=(rmse_diff < 0), 
                    color='coral', alpha=0.2, label='SX better')  # Negative = SX has lower RMSE
    ax.fill_between(dates_dt, rmse_diff, 0, where=(rmse_diff > 0), 
                    color='steelblue', alpha=0.2, label='SXE better')  # Positive = SXE has lower RMSE
    
    # Format x-axis with month labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    ax.tick_params(axis='x', labelsize=16)
    
    # Labels
    ax.set_xlabel('Month')
    ax.set_ylabel('RMSE Difference (K): (SX) - (SXE)')
    ax.grid(axis='both', alpha=0.3)
    
    # Set y-axis limits - extend to -1.5
    y_max = max(1.0, np.max(rmse_diff) * 1.1)
    ax.set_ylim(-1.5, y_max)
    ax.set_xlim(dates_dt[0], dates_dt[-1])
    
    # Create secondary y-axis for cloud percentage
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    
    # Plot daily cloud percentage (light gray, thin)
    ax2.plot(dates_dt, cloud_pct, color='gray', linewidth=1, alpha=0.5, label='Daily Cloud %')
    
    # Add 7-day rolling mean for cloud percentage (dark gray, thicker)
    if len(cloud_pct) >= window_size:
        rolling_cloud = np.convolve(cloud_pct, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(rolling_dates, rolling_cloud, color='#404040', linewidth=2, 
                 linestyle='--', label=f'Cloud % {window_size}-Day Avg')
    
    ax2.set_ylabel('Cloud Percentage (%)', color='#404040')
    ax2.tick_params(axis='y', labelcolor='#404040')
    ax2.set_ylim(0, 100)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, 
              framealpha=1.0, facecolor='#ffffff', edgecolor='gray', fontsize=12,
              bbox_to_anchor=(0.975, 0.975), borderaxespad=0)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)
    
    # Add statistics annotation (corrected)
    mean_diff = np.mean(rmse_diff)
    pct_be_better = (rmse_diff < 0).sum() / len(rmse_diff) * 100
    text_box = ax.text(
        0.025, 0.975, 
        f'Mean Δ: {mean_diff:.3f} K\nSX better: {pct_be_better:.1f}% of days',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=15,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', alpha=1.0, edgecolor='gray'),
        zorder=100
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'SXE_vs_SX_rmse_difference.jpg')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate comparison figures for SXE and SX."""
    print("=" * 60)
    print("SXE vs SX: Hourly and Daily RMSE Comparison")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(settings.FIGURES_DIR, 'rmse_temporal')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data for both models
    df_sxe = load_predictions('SXE')
    df_sx = load_predictions('SX')
    
    # Generate comparison figures
    hourly_path = plot_hourly_comparison_boxplot(df_sxe, df_sx, output_dir)
    daily_path = plot_daily_comparison_lineplot(df_sxe, df_sx, output_dir)
    diff_path = plot_rmse_difference(df_sxe, df_sx, output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Generated figures:")
    print(f"  1. {hourly_path}")
    print(f"  2. {daily_path}")
    print(f"  3. {diff_path}")
    print("=" * 60)
    
    return [hourly_path, daily_path, diff_path]


if __name__ == "__main__":
    main()
