"""
Statistical Analysis for LST Predictions
Performs Paired T-test and Wilcoxon Signed-Rank Test on RMSE and STD
for:
1. BLM vs BLAM (Clear & Cloudy)
2. CIM vs CIAM (Clear & Cloudy)

Two analysis modes:
- Station-Level: Aggregates metrics by station, then performs paired tests
- Aggregate Residuals: Tests all individual residuals (paired by observation)
"""

import os
import sys
import numpy as np
import polars as pl
from scipy import stats
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

METRICS = ['rmse', 'std']
CONDITIONS = [('Clear-Sky', 0), ('Cloudy-Sky', 1)]
COMPARISONS = [
    ('BLM', 'BLAM'),
    ('CIM', 'CIAM')
]

def load_data(models):
    models_data = {}
    for model_name in models:
        pred_path = os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv')
        if os.path.exists(pred_path):
            print(f"Loading {model_name}...")
            df = pl.read_csv(pred_path)
            models_data[model_name] = df
        else:
            print(f"Warning: {pred_path} not found.")
    return models_data

def calculate_station_metrics(df):
    """
    Groups by station_id and calculates RMSE and Error STD for each station.
    """
    # Calculate error
    df = df.with_columns(
        (pl.col('LST_pred') - pl.col('LST_true')).alias('error')
    )
    
    # Group by station
    station_stats = df.group_by('station_id').agg([
        (pl.col('error') ** 2).mean().sqrt().alias('rmse'),
        pl.col('error').std().alias('std'),
        pl.len().alias('count')
    ]).sort('station_id')
    
    return station_stats

def perform_tests(data1, data2, metric=None):
    """
    Performs Paired T-test, Wilcoxon Signed-Rank Test, and Cohen's d.
    data1, data2: lists or numpy arrays of metric values (aligned by station or observation).
    """
    d1 = np.array(data1)
    d2 = np.array(data2)
    differences = d1 - d2
    
    # Paired T-test
    t_stat, t_p = stats.ttest_rel(d1, d2)
    
    # Wilcoxon Signed-Rank Test
    try:
        w_stat, w_p = stats.wilcoxon(d1, d2)
    except ValueError:
        # Happens if all differences are zero
        w_stat, w_p = np.nan, np.nan
        
    # Cohen's d (Paired)
    # Calculation: Mean of differences / Standard Deviation of differences
    # Interpreting d: 0.2 (small), 0.5 (medium), 0.8 (large)
    std_diff = np.std(differences, ddof=1)
    cohen_d = np.mean(differences) / std_diff if std_diff != 0 else 0
        
    return {
        't_stat': t_stat, 't_p': t_p,
        'w_stat': w_stat, 'w_p': w_p,
        'cohen_d': cohen_d,
        'mean_diff': np.mean(differences),
        'n': len(d1)
    }


def perform_aggregate_residual_tests(df1, df2, m1_name, m2_name):
    """
    Performs statistical tests on aggregate residuals.
    Pairs observations by timestamp and station_id.
    
    Returns dict with test results for absolute errors.
    """
    # Calculate errors for each model
    df1 = df1.with_columns([
        (pl.col('LST_pred') - pl.col('LST_true')).alias('error_m1'),
        (pl.col('LST_pred') - pl.col('LST_true')).abs().alias('abs_error_m1')
    ])
    
    df2 = df2.with_columns([
        (pl.col('LST_pred') - pl.col('LST_true')).alias('error_m2'),
        (pl.col('LST_pred') - pl.col('LST_true')).abs().alias('abs_error_m2')
    ])
    
    # Select only necessary columns for joining
    df1_select = df1.select(['timestamp', 'station_id', 'error_m1', 'abs_error_m1'])
    df2_select = df2.select(['timestamp', 'station_id', 'error_m2', 'abs_error_m2'])
    
    # Join on timestamp and station_id to pair observations
    joined = df1_select.join(
        df2_select,
        on=['timestamp', 'station_id'],
        how='inner'
    )
    
    if joined.height < 2:
        return None
    
    # Extract paired values
    abs_err1 = joined['abs_error_m1'].to_numpy()
    abs_err2 = joined['abs_error_m2'].to_numpy()
    
    # Perform tests on absolute errors
    test_res = perform_tests(abs_err1, abs_err2)
    
    return test_res


def station_level_analysis(models_data):
    """
    Original station-level analysis: aggregates by station, then tests.
    """
    results = []
    
    print("\n=== Station-Level Analysis ===")
    print("(Paired tests on station-aggregated RMSE and STD)")
    
    for m1_name, m2_name in COMPARISONS:
        if m1_name not in models_data or m2_name not in models_data:
            print(f"Skipping {m1_name} vs {m2_name}: Data missing.")
            continue
            
        df1_all = models_data[m1_name]
        df2_all = models_data[m2_name]
        
        for cond_name, cond_val in CONDITIONS:
            print(f"  {m1_name} vs {m2_name} ({cond_name})")
            
            # Filter by condition
            df1_cond = df1_all.filter(pl.col('ACMC_BCM') == cond_val)
            df2_cond = df2_all.filter(pl.col('ACMC_BCM') == cond_val)
            
            # Calculate station-wise metrics
            stats1 = calculate_station_metrics(df1_cond)
            stats2 = calculate_station_metrics(df2_cond)
            
            # Join on station_id to ensure we compare same stations
            joined = stats1.join(stats2, on='station_id', how='inner', suffix=f'_{m2_name}')
            
            if joined.height < 2:
                print(f"    Insufficient matching stations (n={joined.height}). Skipping.")
                continue
                
            for metric in METRICS:
                col1 = metric
                col2 = f"{metric}_{m2_name}"

                val1 = joined[col1].to_numpy()
                val2 = joined[col2].to_numpy()
                
                test_res = perform_tests(val1, val2, metric)
                
                res_row = {
                    'Analysis': 'Station-Level',
                    'Comparison': f"{m1_name} vs {m2_name}",
                    'Condition': cond_name,
                    'Metric': metric.upper(),
                    'N': test_res['n'],
                    'Mean_Diff': test_res['mean_diff'],
                    'Cohen_d': test_res['cohen_d'],
                    'Paired_T_p': test_res['t_p'],
                    'Wilcoxon_p': test_res['w_p']
                }
                results.append(res_row)
    
    return results


def aggregate_residual_analysis(models_data):
    """
    Aggregate residual analysis: tests all individual residuals paired by observation.
    """
    results = []
    
    print("\n=== Aggregate Residual Analysis ===")
    print("(Paired tests on all individual absolute errors)")
    
    for m1_name, m2_name in COMPARISONS:
        if m1_name not in models_data or m2_name not in models_data:
            print(f"Skipping {m1_name} vs {m2_name}: Data missing.")
            continue
            
        df1_all = models_data[m1_name]
        df2_all = models_data[m2_name]
        
        for cond_name, cond_val in CONDITIONS:
            print(f"  {m1_name} vs {m2_name} ({cond_name})")
            
            # Filter by condition
            df1_cond = df1_all.filter(pl.col('ACMC_BCM') == cond_val)
            df2_cond = df2_all.filter(pl.col('ACMC_BCM') == cond_val)
            
            # Perform aggregate residual tests
            test_res = perform_aggregate_residual_tests(df1_cond, df2_cond, m1_name, m2_name)
            
            if test_res is None:
                print(f"    Insufficient paired observations. Skipping.")
                continue
            
            res_row = {
                'Analysis': 'Aggregate Residuals',
                'Comparison': f"{m1_name} vs {m2_name}",
                'Condition': cond_name,
                'Metric': 'Abs_Error',
                'N': test_res['n'],
                'Mean_Diff': test_res['mean_diff'],
                'Cohen_d': test_res['cohen_d'],
                'Paired_T_p': test_res['t_p'],
                'Wilcoxon_p': test_res['w_p']
            }
            results.append(res_row)
    
    return results


def main():
    print("Starting Statistical Analysis...")
    
    # Identify all needed models
    required_models = set()
    for m1, m2 in COMPARISONS:
        required_models.add(m1)
        required_models.add(m2)
    
    models_data = load_data(list(required_models))
    
    # Run both analyses
    station_results = station_level_analysis(models_data)
    aggregate_results = aggregate_residual_analysis(models_data)
    
    # Combine results
    all_results = station_results + aggregate_results
    res_df = pd.DataFrame(all_results)
    
    # Save combined results
    out_path = os.path.join(settings.BASE_DIR, 'analysis', 'statistical_results.csv')
    res_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    
    # Display formatted tables
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    # Format p-values for display
    disp_df = res_df.copy()
    disp_df['Paired_T_p'] = disp_df['Paired_T_p'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    disp_df['Wilcoxon_p'] = disp_df['Wilcoxon_p'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    disp_df['Mean_Diff'] = disp_df['Mean_Diff'].apply(lambda x: f"{x:.6f}")
    disp_df['Cohen_d'] = disp_df['Cohen_d'].apply(lambda x: f"{x:.4f}")
    
    # Display station-level results
    station_df = disp_df[disp_df['Analysis'] == 'Station-Level']
    if not station_df.empty:
        print("\nStation-Level Analysis (RMSE and STD aggregated by station):")
        print(station_df.drop(columns=['Analysis']).to_markdown(index=False))
    
    # Display aggregate residual results
    agg_df = disp_df[disp_df['Analysis'] == 'Aggregate Residuals']
    if not agg_df.empty:
        print("\nAggregate Residual Analysis (All individual absolute errors):")
        print(agg_df.drop(columns=['Analysis']).to_markdown(index=False))


if __name__ == "__main__":
    main()
