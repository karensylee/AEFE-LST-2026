"""
Cloud Percentage by Station Analysis with B vs B-E Metrics.

This script visualizes:
1. Cloud percentage per station (horizontal bar chart)
2. Total observations per station
3. RMSE and STD differences (B - B-E) per station
4. Summary statistics
"""

import os
import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

# Matplotlib configuration (matching density_plots.py template)
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def load_predictions(model_name: str) -> pl.DataFrame:
    """
    Load model predictions.
    """
    prediction_path = os.path.join(
        settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv'
    )
    
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Predictions not found at: {prediction_path}")
    
    print(f"Loading {model_name} predictions from: {prediction_path}")
    
    df = pl.read_csv(prediction_path)
    print(f"Loaded {len(df):,} observations")
    
    return df


def compute_station_metrics(df: pl.DataFrame, model_name: str) -> pl.DataFrame:
    """
    Compute per-station metrics: cloud stats, RMSE, and STD.
    
    Returns DataFrame with:
    - station_id
    - total_obs: total number of observations
    - cloudy_obs: number of cloudy observations (ACMC_BCM = 1)
    - clear_obs: number of clear observations (ACMC_BCM = 0)
    - cloud_pct: percentage of cloudy observations
    - rmse_{model}: RMSE for this model
    - std_{model}: STD of residuals for this model
    """
    # Calculate residuals
    df = df.with_columns([
        (pl.col('LST_pred') - pl.col('LST_true')).alias('residual')
    ])
    
    stats = df.group_by('station_id').agg([
        pl.len().alias('total_obs'),
        pl.col('ACMC_BCM').sum().alias('cloudy_obs'),
        (1 - pl.col('ACMC_BCM')).sum().alias('clear_obs'),
        (pl.col('ACMC_BCM').mean() * 100).alias('cloud_pct'),
        # RMSE = sqrt(mean(residual^2))
        (pl.col('residual') ** 2).mean().sqrt().alias(f'rmse_{model_name}'),
        # STD = standard deviation of residuals
        pl.col('residual').std().alias(f'std_{model_name}')
    ])
    
    return stats


def compute_combined_stats(df_blam: pl.DataFrame, df_blm: pl.DataFrame) -> pl.DataFrame:
    """
    Compute combined statistics from both models.
    
    Returns DataFrame with cloud stats and RMSE/STD differences (B - B-E).
    """
    # Compute metrics for each model
    stats_blam = compute_station_metrics(df_blam, 'B')
    stats_blm = compute_station_metrics(df_blm, 'B-E')
    
    # Keep cloud stats from B (should be same as B-E)
    cloud_cols = ['station_id', 'total_obs', 'cloudy_obs', 'clear_obs', 'cloud_pct']
    
    # Select only RMSE/STD from B-E stats
    blm_metrics = stats_blm.select(['station_id', 'rmse_B-E', 'std_B-E'])
    
    # Join and compute differences
    combined = stats_blam.join(blm_metrics, on='station_id', how='inner')
    
    # Calculate differences (B - B-E): negative = B is better
    combined = combined.with_columns([
        (pl.col('rmse_B') - pl.col('rmse_B-E')).alias('rmse_diff'),
        (pl.col('std_B') - pl.col('std_B-E')).alias('std_diff')
    ])
    
    # Sort by cloud percentage (most cloudy first)
    combined = combined.sort('cloud_pct', descending=True)
    
    return combined


def plot_cloud_percentage_bar(stats: pl.DataFrame, output_dir: str) -> str:
    """
    Create a horizontal bar chart showing cloud percentage by station.
    
    Bars are colored by cloud percentage (gradient from blue to gray).
    Shows observation count on each bar.
    """
    print("\nGenerating cloud percentage by station bar chart...")
    
    # Convert to pandas for easier plotting
    df_plot = stats.to_pandas()
    n_stations = len(df_plot)
    
    # Create figure with appropriate height
    fig_height = max(10, n_stations * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Create colormap based on cloud percentage
    cmap = plt.cm.RdYlBu_r  # Red (cloudy) to Blue (clear)
    colors = cmap(df_plot['cloud_pct'] / 100)
    
    # Create horizontal bar chart
    y_positions = np.arange(n_stations)
    bars = ax.barh(y_positions, df_plot['cloud_pct'], color=colors, edgecolor='gray', alpha=0.8)
    
    # Add observation count as text annotation on bars
    for i, (pct, total) in enumerate(zip(df_plot['cloud_pct'], df_plot['total_obs'])):
        # Place text inside or outside bar depending on bar length
        if pct > 15:
            ax.text(pct - 1, i, f'n={total:,}', va='center', ha='right', 
                   fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(pct + 1, i, f'n={total:,}', va='center', ha='left', 
                   fontsize=8, color='black')
    
    # Format station labels (show last 3 digits for readability)
    station_labels = [f"...{str(sid)[-3:]}" if len(str(sid)) > 3 else str(sid) 
                      for sid in df_plot['station_id']]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(station_labels)
    
    # Labels and title
    ax.set_xlabel('Cloud Percentage (%)')
    ax.set_ylabel('Station ID')
    ax.set_title('Cloud Percentage by Station\n(sorted by cloudiness)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label('Cloud %', fontsize=10)
    
    # Add summary statistics annotation
    mean_cloud = df_plot['cloud_pct'].mean()
    median_cloud = df_plot['cloud_pct'].median()
    total_obs = df_plot['total_obs'].sum()
    
    stats_text = (
        f"Summary Statistics\n"
        f"─────────────────\n"
        f"Stations: {n_stations}\n"
        f"Total Obs: {total_obs:,}\n"
        f"Mean Cloud: {mean_cloud:.1f}%\n"
        f"Median Cloud: {median_cloud:.1f}%"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'),
            zorder=100)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'cloud_percentage_by_station.jpg')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def plot_observations_by_station(stats: pl.DataFrame, output_dir: str) -> str:
    """
    Create a horizontal bar chart showing total observations by station.
    
    - Bars colored by cloud percentage
    - RMSE diff and STD diff annotations
    """
    print("\nGenerating observations by station bar chart...")
    
    # Sort by total observations
    df_plot = stats.sort('total_obs', descending=True).to_pandas()
    n_stations = len(df_plot)
    
    # Create figure with appropriate height
    fig_height = max(12, n_stations * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Create colormap based on cloud percentage
    cmap = plt.cm.RdYlBu_r  # Red (cloudy) to Blue (clear)
    colors = cmap(df_plot['cloud_pct'] / 100)
    
    # Create horizontal bar chart
    y_positions = np.arange(n_stations)
    bars = ax.barh(y_positions, df_plot['total_obs'], color=colors, edgecolor='gray', alpha=0.8)
    
    max_obs = int(df_plot['total_obs'].max())
    
    # Add cloud percentage on the LEFT
    for i, pct in enumerate(df_plot['cloud_pct']):
        ax.text(-max_obs * 0.02, i, f'{pct:.0f}%', va='center', ha='right', 
               fontsize=8, color='black')
    
    # Add RMSE diff and STD diff on the RIGHT
    for i, row in df_plot.iterrows():
        rmse_diff = row['rmse_diff']
        std_diff = row['std_diff']
        obs = row['total_obs']
        
        rmse_sign = '+' if rmse_diff > 0 else ''
        std_sign = '+' if std_diff > 0 else ''
        
        text = f"ΔRMSE: {rmse_sign}{rmse_diff:.3f}K  ΔSTD: {std_sign}{std_diff:.3f}K"
        ax.text(max_obs * 1.02, i, text, va='center', ha='left', 
               fontsize=7, color='black', fontfamily='monospace')
    
    # Format station labels
    station_labels = [f"...{str(sid)[-3:]}" if len(str(sid)) > 3 else str(sid) 
                      for sid in df_plot['station_id']]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(station_labels)
    
    # Labels and title
    ax.set_xlabel('Total Observations')
    ax.set_ylabel('Station ID')
    ax.set_title('Total Observations by Station with B vs B-E Performance\n(colored by cloud %, sorted by obs count)', 
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Extend plot area
    plt.subplots_adjust(right=0.72, left=0.12)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9, aspect=20, pad=0.02)
    cbar.set_label('Cloud %', fontsize=10)
    
    # Save figure
    output_path = os.path.join(output_dir, 'observations_by_station.jpg')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def plot_cloud_vs_observations_scatter(stats: pl.DataFrame, output_dir: str) -> str:
    """
    Create a scatter plot of cloud percentage vs total observations.
    
    Points are colored by RMSE difference.
    """
    print("\nGenerating cloud vs observations scatter plot...")
    
    df_plot = stats.to_pandas()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Normalize RMSE diff for coloring (diverging colormap centered at 0)
    vmax = max(abs(df_plot['rmse_diff'].min()), abs(df_plot['rmse_diff'].max()))
    
    # Create scatter plot
    scatter = ax.scatter(
        df_plot['total_obs'],
        df_plot['cloud_pct'],
        c=df_plot['rmse_diff'],
        cmap='RdBu',  # Red = worse, Blue = better
        vmin=-vmax,
        vmax=vmax,
        s=100,
        alpha=0.8,
        edgecolors='gray',
        linewidths=0.5
    )
    
    # Add station labels for extreme cases
    for _, row in df_plot.iterrows():
        # Label stations with very high/low cloud % or extreme RMSE diff
        if row['cloud_pct'] > 75 or row['cloud_pct'] < 30 or \
           abs(row['rmse_diff']) > df_plot['rmse_diff'].std() * 1.5:
            label = f"...{str(row['station_id'])[-3:]}"
            ax.annotate(label, (row['total_obs'], row['cloud_pct']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Labels and title
    ax.set_xlabel('Total Observations')
    ax.set_ylabel('Cloud Percentage (%)')
    ax.set_title('Cloud % vs. Observations (colored by ΔRMSE)\nBlue = B better, Red = B-E better', 
                 fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('ΔRMSE (B - B-E) [K]', fontsize=10)
    
    # Add reference lines for mean values
    mean_cloud = df_plot['cloud_pct'].mean()
    mean_obs = df_plot['total_obs'].mean()
    ax.axhline(y=mean_cloud, color='gray', linestyle='--', alpha=0.5, label=f'Mean Cloud: {mean_cloud:.1f}%')
    ax.axvline(x=mean_obs, color='gray', linestyle=':', alpha=0.5, label=f'Mean Obs: {mean_obs:,.0f}')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'cloud_vs_observations_scatter.jpg')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def print_summary_table(stats: pl.DataFrame):
    """Print a summary table of cloud and performance statistics by station."""
    print("\n" + "=" * 100)
    print("CLOUD & PERFORMANCE STATISTICS BY STATION (B vs B-E)")
    print("=" * 100)
    print(f"{'Station':<10} {'Total Obs':>12} {'Cloudy':>10} {'Clear':>10} {'Cloud %':>8} {'ΔRMSE':>10} {'ΔSTD':>10}")
    print("-" * 100)
    
    for row in stats.iter_rows(named=True):
        rmse_diff = row['rmse_diff']
        std_diff = row['std_diff']
        print(f"{row['station_id']:<10} {row['total_obs']:>12,} {row['cloudy_obs']:>10,.0f} "
              f"{row['clear_obs']:>10,.0f} {row['cloud_pct']:>7.1f}% {rmse_diff:>+9.4f}K {std_diff:>+9.4f}K")
    
    print("-" * 100)
    
    # Summary row
    df = stats.to_pandas()
    total_obs = df['total_obs'].sum()
    total_cloudy = df['cloudy_obs'].sum()
    total_clear = df['clear_obs'].sum()
    overall_cloud_pct = (total_cloudy / total_obs) * 100
    mean_rmse_diff = df['rmse_diff'].mean()
    mean_std_diff = df['std_diff'].mean()
    
    print(f"{'MEAN/TOTAL':<10} {total_obs:>12,} {total_cloudy:>10,.0f} {total_clear:>10,.0f} "
          f"{overall_cloud_pct:>7.1f}% {mean_rmse_diff:>+9.4f}K {mean_std_diff:>+9.4f}K")
    print("=" * 100)
    print("Note: Negative ΔRMSE/ΔSTD means B outperforms B-E")


def save_stats_csv(stats: pl.DataFrame, output_dir: str) -> str:
    """Save station cloud and performance statistics to CSV."""
    output_path = os.path.join(output_dir, 'cloud_statistics_by_station.csv')
    stats.write_csv(output_path)
    print(f"\nSaved statistics to: {output_path}")
    return output_path


def main():
    """Generate cloud percentage by station visualizations with B vs B-E metrics."""
    print("=" * 60)
    print("Cloud Percentage by Station Analysis (B vs B-E)")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(settings.FIGURES_DIR, 'cloud_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load predictions for both models
    df_blam = load_predictions('B')
    df_blm = load_predictions('B-E')
    
    # Compute combined station-level statistics
    print("\nComputing per-station metrics...")
    stats = compute_combined_stats(df_blam, df_blm)
    
    # Print summary table
    print_summary_table(stats)
    
    # Generate visualizations
    figures = []
    figures.append(plot_cloud_percentage_bar(stats, output_dir))
    figures.append(plot_cloud_vs_observations_scatter(stats, output_dir))
    
    # Save statistics to CSV
    csv_path = save_stats_csv(stats, output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Generated outputs:")
    for i, fig in enumerate(figures, 1):
        print(f"  {i}. {fig}")
    print(f"  {len(figures)+1}. {csv_path}")
    print("=" * 60)
    
    return figures, csv_path


if __name__ == "__main__":
    main()
