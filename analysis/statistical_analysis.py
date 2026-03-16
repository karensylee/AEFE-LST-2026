"""
Statistical Analysis Module for LST Predictions

Core statistical functions and metrics calculation.
Separated from visualization for reusability.
"""

import os
import sys
import numpy as np
import polars as pl
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

METRICS = ['rmse', 'std']
CONDITIONS = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
COMPARISONS = [('SX', 'SXE'), ('S', 'SE')]


def load_data(models=None):
    """Load prediction data for specified models."""
    if models is None:
        models = [m for pair in COMPARISONS for m in pair]
    paths = {m: os.path.join(settings.OUTPUT_DIR, 'xgb', m, f'{m}_ALL_predictions.csv') for m in models}
    return {m: pl.read_csv(p) for m, p in paths.items() if os.path.exists(p)}


def calculate_station_metrics(df):
    """Calculate per-station RMSE and STD metrics."""
    df = df.with_columns((pl.col('LST_pred') - pl.col('LST_true')).alias('error'))
    return df.group_by('station_id').agg([
        (pl.col('error') ** 2).mean().sqrt().alias('rmse'),
        pl.col('error').std().alias('std'),
        pl.len().alias('count')
    ]).sort('station_id')


def cohen_q(r1, r2):
    """
    Calculate Cohen's q for effect size between two correlations.
    q = 0.5 * (ln((1+r1)/(1-r1)) - ln((1+r2)/(1-r2)))
    """
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    return z1 - z2


def perform_tests(data1, data2, is_correlation=False):
    """
    Perform statistical tests.
    If is_correlation is True, data1 and data2 are treated as correlation coefficients.
    """
    d1, d2 = np.array(data1), np.array(data2)
    
    # Wilcoxon Signed-Rank Test
    w_stat, w_p = stats.wilcoxon(d1, d2)
    
    results = {'w_p': w_p, 'n': len(d1)}
    
    # Cohen's q (Effect Size for Correlations)
    if is_correlation:
        r1 = np.mean(d1)
        r2 = np.mean(d2)
        q = cohen_q(r1, r2)
        results['cohen_q'] = q
    
    return results


def generate_metrics_csv(models_data):
    """
    Generate a CSV with STD and Median Absolute Residual Bias for all models and conditions.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    all_models = list(models_data.keys())
    results = []
    
    # Extended conditions: include All-Sky (None means no filter)
    all_conditions = [('Clear-Sky', 0), ('Cloudy-Sky', 1), ('All-Sky', None)]
    
    for model_name in all_models:
        df = models_data[model_name]
        
        for cond_name, cond_val in all_conditions:
            # Filter by condition, or use all data for All-Sky
            if cond_val is not None:
                df_cond = df.filter(pl.col('ACMC_BCM') == cond_val)
            else:
                df_cond = df
            
            y_true = df_cond['LST_true'].to_numpy()
            y_pred = df_cond['LST_pred'].to_numpy()
            errors = y_pred - y_true
            
            n = len(errors)
            
            # RMSE
            rmse = np.sqrt(np.mean(errors ** 2))
            
            # R²
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # STD of residuals
            std = np.std(errors, ddof=1)
            
            # Median Bias (median of errors, not absolute)
            median_bias = np.median(errors)
            
            results.append({
                'Model': model_name,
                'Condition': cond_name,
                'n': n,
                'RMSE': round(rmse, 2),
                'R2': round(r2, 2),
                'STD': round(std, 2),
                'Median_Bias': round(median_bias, 2)
            })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    out_path = os.path.join(fig_dir, 'model_metrics_summary.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nMetrics CSV saved to: {out_path}")
    print(results_df.to_string(index=False))
    
    return results_df


def plot_residual_distributions(models_data):
    """
    Plot histograms of All-Sky residual errors for all models.
    Overlays a fitted normal distribution for visual comparison.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    model_order = ['SXE', 'SX', 'SE', 'S']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for idx, model_name in enumerate(model_order):
        if model_name not in models_data:
            continue
            
        df = models_data[model_name]
        ax = axes[idx]
        
        # Calculate residual errors
        errors = (df['LST_pred'] - df['LST_true']).to_numpy()
        
        # Sample for faster plotting if needed (use all for stats)
        n_plot = min(500000, len(errors))
        errors_sample = np.random.choice(errors, n_plot, replace=False) if len(errors) > n_plot else errors
        
        # Calculate statistics
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        std_err = np.std(errors, ddof=1)
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        
        # Plot histogram
        n, bins, patches = ax.hist(errors_sample, bins=100, density=True, 
                                    alpha=0.7, color=colors[idx], edgecolor='white', linewidth=0.5)
        
        # Overlay fitted normal distribution
        x_range = np.linspace(errors.min(), errors.max(), 200)
        normal_pdf = stats.norm.pdf(x_range, mean_err, std_err)
        ax.plot(x_range, normal_pdf, 'k--', linewidth=2, label='Normal fit')
        
        # Add vertical lines for mean and median
        ax.axvline(mean_err, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_err:.2f}')
        ax.axvline(median_err, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}')
        
        # Labels and title
        ax.set_xlabel('Residual Error (°C)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{model_name} (n={len(errors):,})', fontsize=13, fontweight='bold')
        
        # Stats annotation
        stats_text = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}\nStd: {std_err:.2f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-20, 20)
    
    plt.suptitle('All-Sky Residual Error Distributions', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(fig_dir, 'residual_distributions_allsky.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nResidual distribution plot saved to: {out_path}")


def plot_residual_distributions_by_station(models_data):
    """
    Plot histograms of All-Sky residual errors for each station.
    Creates one 2x2 plot per station showing all 4 models.
    Saves to a dedicated subfolder.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests', 'residual_distributions_by_station')
    os.makedirs(fig_dir, exist_ok=True)
    
    model_order = ['SXE', 'SX', 'SE', 'S']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    # Get all unique stations from the first model
    first_model = list(models_data.keys())[0]
    stations = sorted(models_data[first_model]['station_id'].unique().to_list())
    
    print(f"\nGenerating residual distribution plots for {len(stations)} stations...")
    
    for station_idx, station in enumerate(stations):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(model_order):
            if model_name not in models_data:
                continue
                
            df = models_data[model_name]
            ax = axes[idx]
            
            # Filter for this station
            df_station = df.filter(pl.col('station_id') == station)
            
            if len(df_station) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
                continue
            
            # Calculate residual errors
            errors = (df_station['LST_pred'] - df_station['LST_true']).to_numpy()
            n = len(errors)
            
            # Calculate statistics
            mean_err = np.mean(errors)
            median_err = np.median(errors)
            std_err = np.std(errors, ddof=1) if n > 1 else 0
            skewness = stats.skew(errors) if n > 2 else 0
            kurtosis = stats.kurtosis(errors) if n > 3 else 0
            
            # Determine bins based on data range
            n_bins = min(50, max(10, n // 100))
            
            # Plot histogram
            ax.hist(errors, bins=n_bins, density=True, 
                    alpha=0.7, color=colors[idx], edgecolor='white', linewidth=0.5)
            
            # Overlay fitted normal distribution if we have enough data
            if std_err > 0:
                x_range = np.linspace(errors.min(), errors.max(), 200)
                normal_pdf = stats.norm.pdf(x_range, mean_err, std_err)
                ax.plot(x_range, normal_pdf, 'k--', linewidth=2, label='Normal fit')
            
            # Add vertical lines for mean and median
            ax.axvline(mean_err, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_err:.2f}')
            ax.axvline(median_err, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}')
            
            # Labels and title
            ax.set_xlabel('Residual Error (°C)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{model_name} (n={n:,})', fontsize=13, fontweight='bold')
            
            # Stats annotation
            stats_text = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}\nStd: {std_err:.2f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(loc='upper left', fontsize=9)
            ax.set_xlim(-20, 20)
        
        plt.suptitle(f'Station {station} - All-Sky Residual Error Distributions', 
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        out_path = os.path.join(fig_dir, f'residual_dist_station_{station}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Progress indicator
        if (station_idx + 1) % 10 == 0 or station_idx == len(stations) - 1:
            print(f"  Processed {station_idx + 1}/{len(stations)} stations...")
    
    print(f"\nAll station plots saved to: {fig_dir}")


def ks_normality_station_aggregates(models_data):
    """
    Test if the distribution of per-station median and mean residuals is normal.
    For each model:
    - Calculate median residual per station (53 values)
    - Calculate mean residual per station (53 values)
    - Perform KS test on these 53-value distributions
    - Visualize with histograms
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    model_order = ['SXE', 'SX', 'SE', 'S']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    results = []
    station_stats = {}  # Store for visualization
    
    print("\nCalculating station-level median and mean residuals...")
    
    for model_name in model_order:
        if model_name not in models_data:
            continue
            
        df = models_data[model_name]
        
        # Calculate per-station statistics
        station_agg = df.with_columns(
            (pl.col('LST_pred') - pl.col('LST_true')).alias('error')
        ).group_by('station_id').agg([
            pl.col('error').median().alias('median_residual'),
            pl.col('error').mean().alias('mean_residual'),
            pl.len().alias('n')
        ]).sort('station_id')
        
        median_residuals = station_agg['median_residual'].to_numpy()
        mean_residuals = station_agg['mean_residual'].to_numpy()
        n_stations = len(median_residuals)
        
        # Store for visualization
        station_stats[model_name] = {
            'median': median_residuals,
            'mean': mean_residuals
        }
        
        # KS test for median residuals
        median_standardized = (median_residuals - np.mean(median_residuals)) / np.std(median_residuals, ddof=1)
        ks_stat_median, ks_p_median = stats.kstest(median_standardized, 'norm')
        
        # KS test for mean residuals
        mean_standardized = (mean_residuals - np.mean(mean_residuals)) / np.std(mean_residuals, ddof=1)
        ks_stat_mean, ks_p_mean = stats.kstest(mean_standardized, 'norm')
        
        results.append({
            'Model': model_name,
            'Metric': 'Median Residual',
            'n_stations': n_stations,
            'Mean_of_Metric': round(np.mean(median_residuals), 4),
            'Std_of_Metric': round(np.std(median_residuals, ddof=1), 4),
            'KS_Statistic': round(ks_stat_median, 4),
            'KS_p_value': f"{ks_p_median:.4f}",
            'Normal_at_0.05': "Yes" if ks_p_median > 0.05 else "No"
        })
        
        results.append({
            'Model': model_name,
            'Metric': 'Mean Residual',
            'n_stations': n_stations,
            'Mean_of_Metric': round(np.mean(mean_residuals), 4),
            'Std_of_Metric': round(np.std(mean_residuals, ddof=1), 4),
            'KS_Statistic': round(ks_stat_mean, 4),
            'KS_p_value': f"{ks_p_mean:.4f}",
            'Normal_at_0.05': "Yes" if ks_p_mean > 0.05 else "No"
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    out_path = os.path.join(fig_dir, 'ks_normality_station_aggregates.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nKS Normality (Station Aggregates) CSV saved to: {out_path}")
    print(results_df.to_string(index=False))
    
    # Create visualization - 2 rows (median, mean) x 4 columns (models)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for col_idx, model_name in enumerate(model_order):
        if model_name not in station_stats:
            continue
            
        color = colors[col_idx]
        median_vals = station_stats[model_name]['median']
        mean_vals = station_stats[model_name]['mean']
        
        # Row 0: Median residuals
        ax_median = axes[0, col_idx]
        ax_median.hist(median_vals, bins=15, density=True, alpha=0.7, color=color, edgecolor='white')
        
        # Overlay normal fit
        x_range = np.linspace(median_vals.min(), median_vals.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, np.mean(median_vals), np.std(median_vals, ddof=1))
        ax_median.plot(x_range, normal_pdf, 'k--', linewidth=2)
        ax_median.axvline(np.mean(median_vals), color='red', linestyle='-', linewidth=1.5, label='Mean')
        ax_median.axvline(np.median(median_vals), color='orange', linestyle='--', linewidth=1.5, label='Median')
        
        # Get KS result for annotation
        median_result = [r for r in results if r['Model'] == model_name and r['Metric'] == 'Median Residual'][0]
        ax_median.set_title(f'{model_name}\np={median_result["KS_p_value"]}', fontsize=11)
        if col_idx == 0:
            ax_median.set_ylabel('Density\n(Median Residual)', fontsize=10)
        ax_median.tick_params(labelsize=9)
        
        # Row 1: Mean residuals
        ax_mean = axes[1, col_idx]
        ax_mean.hist(mean_vals, bins=15, density=True, alpha=0.7, color=color, edgecolor='white')
        
        # Overlay normal fit
        x_range = np.linspace(mean_vals.min(), mean_vals.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, np.mean(mean_vals), np.std(mean_vals, ddof=1))
        ax_mean.plot(x_range, normal_pdf, 'k--', linewidth=2)
        ax_mean.axvline(np.mean(mean_vals), color='red', linestyle='-', linewidth=1.5)
        ax_mean.axvline(np.median(mean_vals), color='orange', linestyle='--', linewidth=1.5)
        
        mean_result = [r for r in results if r['Model'] == model_name and r['Metric'] == 'Mean Residual'][0]
        ax_mean.set_xlabel(f'Residual (K)\np={mean_result["KS_p_value"]}', fontsize=10)
        if col_idx == 0:
            ax_mean.set_ylabel('Density\n(Mean Residual)', fontsize=10)
        ax_mean.tick_params(labelsize=9)
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Station-Level Residual Distributions (n=53 stations)\nBlack dashed = Normal fit', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(fig_dir, 'station_aggregate_normality.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
    return results_df


def ks_normality_test_allsky_pooled(models_data):
    """
    Perform Kolmogorov-Smirnov normality test for All-Sky residual errors.
    Tests the entire dataset (pooled) for each model.
    Returns a DataFrame with KS statistic and p-value for each model.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    results = []
    
    for model_name, df in models_data.items():
        # Calculate residual errors (predicted - true)
        errors = (df['LST_pred'] - df['LST_true']).to_numpy()
        n = len(errors)
        
        # Standardize errors for KS test (compare to standard normal)
        errors_standardized = (errors - np.mean(errors)) / np.std(errors, ddof=1)
        
        # Perform KS test against normal distribution
        ks_stat, ks_p = stats.kstest(errors_standardized, 'norm')
        
        # Interpret result
        is_normal = "Yes" if ks_p > 0.05 else "No"
        
        results.append({
            'Model': model_name,
            'Condition': 'All-Sky',
            'n': n,
            'KS_Statistic': round(ks_stat, 6),
            'KS_p_value': f"{ks_p:.2e}",
            'Normal_at_0.05': is_normal
        })
    
    results_df = pd.DataFrame(results)
    out_path = os.path.join(fig_dir, 'ks_normality_allsky_pooled.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nKS Normality (All-Sky Pooled) CSV saved to: {out_path}")
    print(results_df.to_string(index=False))
    
    return results_df


def ks_laplace_test_allsky_pooled(models_data):
    """
    Perform Kolmogorov-Smirnov test comparing All-Sky residual errors to a Laplace distribution.
    Tests the entire dataset (pooled) for each model.
    
    The Laplace distribution is characterized by:
    - Location parameter (mu): estimated by the median of residuals
    - Scale parameter (b): estimated by the mean absolute deviation from median
    
    Returns a DataFrame with KS statistic and p-value for each model.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    results = []
    
    for model_name, df in models_data.items():
        # Calculate residual errors (predicted - true)
        errors = (df['LST_pred'] - df['LST_true']).to_numpy()
        n = len(errors)
        
        # Fit Laplace distribution parameters (MLE estimates)
        # Location = median, Scale = mean absolute deviation from median
        loc = np.median(errors)
        scale = np.mean(np.abs(errors - loc))
        
        # Perform KS test against Laplace distribution with fitted parameters
        ks_stat, ks_p = stats.kstest(errors, 'laplace', args=(loc, scale))
        
        # Interpret result
        is_laplace = "Yes" if ks_p > 0.05 else "No"
        
        results.append({
            'Model': model_name,
            'Condition': 'All-Sky',
            'n': n,
            'Laplace_loc': round(loc, 4),
            'Laplace_scale': round(scale, 4),
            'KS_Statistic': round(ks_stat, 6),
            'KS_p_value': f"{ks_p:.2e}",
            'Laplace_at_0.05': is_laplace
        })
    
    results_df = pd.DataFrame(results)
    out_path = os.path.join(fig_dir, 'ks_laplace_allsky_pooled.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nKS Laplace Test (All-Sky Pooled) CSV saved to: {out_path}")
    print(results_df.to_string(index=False))
    
    return results_df


def plot_laplace_vs_normal_fit(models_data):
    """
    Plot histograms of All-Sky residual errors with both Normal and Laplace fits overlaid.
    Allows visual comparison of which distribution better fits the residuals.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    model_order = ['SXE', 'SX', 'SE', 'S']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for idx, model_name in enumerate(model_order):
        if model_name not in models_data:
            continue
            
        df = models_data[model_name]
        ax = axes[idx]
        
        # Calculate residual errors
        errors = (df['LST_pred'] - df['LST_true']).to_numpy()
        
        # Sample for faster plotting if needed
        n_plot = min(500000, len(errors))
        errors_sample = np.random.choice(errors, n_plot, replace=False) if len(errors) > n_plot else errors
        
        # Calculate statistics
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        std_err = np.std(errors, ddof=1)
        
        # Laplace parameters (MLE estimates)
        laplace_loc = median_err
        laplace_scale = np.mean(np.abs(errors - laplace_loc))
        
        # KS tests
        errors_standardized = (errors - mean_err) / std_err
        ks_stat_norm, ks_p_norm = stats.kstest(errors_standardized, 'norm')
        ks_stat_lap, ks_p_lap = stats.kstest(errors, 'laplace', args=(laplace_loc, laplace_scale))
        
        # Plot histogram
        n, bins, patches = ax.hist(errors_sample, bins=100, density=True, 
                                    alpha=0.6, color=colors[idx], edgecolor='white', linewidth=0.5)
        
        # X range for PDFs
        x_range = np.linspace(errors.min(), errors.max(), 300)
        
        # Normal fit (dashed line)
        normal_pdf = stats.norm.pdf(x_range, mean_err, std_err)
        ax.plot(x_range, normal_pdf, 'k--', linewidth=2, label=f'Normal (p={ks_p_norm:.2e})')
        
        # Laplace fit (solid line)
        laplace_pdf = stats.laplace.pdf(x_range, laplace_loc, laplace_scale)
        ax.plot(x_range, laplace_pdf, 'darkred', linewidth=2.5, label=f'Laplace (p={ks_p_lap:.2e})')
        
        # Add vertical lines for mean and median
        ax.axvline(mean_err, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_err:.2f}')
        ax.axvline(median_err, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Median: {median_err:.2f}')
        
        # Labels and title
        ax.set_xlabel('Residual Error (°C)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{model_name} (n={len(errors):,})', fontsize=13, fontweight='bold')
        
        # Stats annotation
        stats_text = f'Std: {std_err:.2f}\nLaplace scale: {laplace_scale:.2f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(-15, 15)
    
    plt.suptitle('All-Sky Residual Distributions: Normal vs Laplace Fit', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(fig_dir, 'residual_laplace_vs_normal.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nLaplace vs Normal comparison plot saved to: {out_path}")


def ks_normality_test_by_station(models_data):
    """
    Perform Kolmogorov-Smirnov normality test for All-Sky residual errors.
    Tests each station separately for each model.
    Returns a DataFrame with KS statistic and p-value per station per model.
    """
    fig_dir = os.path.join(settings.FIGURES_DIR, 'statistical_tests')
    os.makedirs(fig_dir, exist_ok=True)
    
    results = []
    
    for model_name, df in models_data.items():
        # Get unique stations
        stations = df['station_id'].unique().to_list()
        
        for station in sorted(stations):
            df_station = df.filter(pl.col('station_id') == station)
            
            # Calculate residual errors (predicted - true)
            errors = (df_station['LST_pred'] - df_station['LST_true']).to_numpy()
            n = len(errors)
            
            # Skip if too few samples for meaningful test
            if n < 10:
                results.append({
                    'Model': model_name,
                    'Station': station,
                    'Condition': 'All-Sky',
                    'n': n,
                    'KS_Statistic': None,
                    'KS_p_value': None,
                    'Normal_at_0.05': 'Insufficient data'
                })
                continue
            
            # Standardize errors for KS test (compare to standard normal)
            std_dev = np.std(errors, ddof=1)
            if std_dev == 0:
                results.append({
                    'Model': model_name,
                    'Station': station,
                    'Condition': 'All-Sky',
                    'n': n,
                    'KS_Statistic': None,
                    'KS_p_value': None,
                    'Normal_at_0.05': 'Zero variance'
                })
                continue
                
            errors_standardized = (errors - np.mean(errors)) / std_dev
            
            # Perform KS test against normal distribution
            ks_stat, ks_p = stats.kstest(errors_standardized, 'norm')
            
            # Interpret result
            is_normal = "Yes" if ks_p > 0.05 else "No"
            
            results.append({
                'Model': model_name,
                'Station': station,
                'Condition': 'All-Sky',
                'n': n,
                'KS_Statistic': round(ks_stat, 6),
                'KS_p_value': f"{ks_p:.2e}",
                'Normal_at_0.05': is_normal
            })
    
    results_df = pd.DataFrame(results)
    out_path = os.path.join(fig_dir, 'ks_normality_allsky_by_station.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nKS Normality (All-Sky By Station) CSV saved to: {out_path}")
    print(f"Total rows: {len(results_df)}")
    print(results_df.head(20).to_string(index=False))
    
    return results_df


def main():
    """Generate metrics CSV and KS normality tests."""
    print("Loading model data...")
    models_data = load_data()
    
    if not models_data:
        print("No model data found.")
        return
    
    print(f"Loaded models: {list(models_data.keys())}")
    
    # Generate standard metrics
    generate_metrics_csv(models_data)
    
    # Perform KS normality tests for all-sky conditions
    print("\n" + "="*60)
    print("KOLMOGOROV-SMIRNOV NORMALITY TESTS (ALL-SKY)")
    print("="*60)
    
    # 1. Pooled (entire dataset) per model
    ks_normality_test_allsky_pooled(models_data)
    
    # 2. Per station per model
    ks_normality_test_by_station(models_data)

    # 3. Laplace distribution test (Pooled)
    print("\n" + "-"*60)
    print("LAPLACE DISTRIBUTION TEST (ALL-SKY)")
    print("-"*60)
    ks_laplace_test_allsky_pooled(models_data)
    plot_laplace_vs_normal_fit(models_data)
    
    print("\nStatistical analysis complete.")


if __name__ == "__main__":
    main()
