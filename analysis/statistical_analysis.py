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

# Add project root to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

METRICS = ['rmse', 'std']
CONDITIONS = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
COMPARISONS = [('B-E', 'B'), ('B-E-X', 'B-X')]


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


def main():
    """Generate metrics CSV only."""
    print("Loading model data...")
    models_data = load_data()
    
    if not models_data:
        print("No model data found.")
        return
    
    print(f"Loaded models: {list(models_data.keys())}")
    generate_metrics_csv(models_data)
    print("\nStatistical analysis complete.")


if __name__ == "__main__":
    main()
