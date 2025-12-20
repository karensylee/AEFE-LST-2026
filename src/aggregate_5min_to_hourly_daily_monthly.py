"""
Main Aggregation Pipeline Script.

This script runs the full aggregation pipeline:
1. Load 5-minute data (with embeddings and CMI/SZA/SAA features)
2. Aggregate to Hourly (QC: ≥10/12 obs)
3. Aggregate to Daily (QC: ≥22/24 hours)
4. Aggregate to Monthly (QC: 100% daily completeness)
5. Save all aggregated datasets
"""

import os
import sys
import pandas as pd
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregation import run_aggregation_pipeline
from config.settings import AGGREGATION_CONFIG, DATA_PATH, BASE_DIR


def main():
    parser = argparse.ArgumentParser(description='Run 5-minute to Hourly/Daily/Monthly Aggregation Pipeline')
    parser.add_argument('--input', '-i', default=DATA_PATH, help=f'Input 5-minute CSV path (default: {DATA_PATH})')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory (default: same as input)')
    parser.add_argument('--qc-col', default='LWOUT', help='Column to use for QC counts (default: LWOUT)')
    
    args = parser.parse_args()
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        print("Please specify a valid input file using --input")
        sys.exit(1)
        
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(input_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    try:
        # Load data - optimize types for memory if needed
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
        
    # Run the pipeline using settings from config
    run_aggregation_pipeline(
        df,
        qc_col=args.qc_col,
        min_5min_per_hour=AGGREGATION_CONFIG['min_5min_obs_per_hour'],
        min_hours_per_day=AGGREGATION_CONFIG['min_hourly_obs_per_day'],
        require_complete_months=AGGREGATION_CONFIG['require_complete_months'],
        timezone=AGGREGATION_CONFIG['timezone']
    )
    
    # Note: run_aggregation_pipeline returns dfs, but here we might want to save them
    # The current src.aggregation.run_aggregation_pipeline returns the dataframes but relies on the caller to save
    # Let's modify the imports to call the functions directly or handle saving here.
    
    # Actually, let's call the logic locally to handle saving properly with the filenames we want
    # Re-running the pipeline logic locally to capture the DataFrames
    
    from src.aggregation import aggregate_hourly, aggregate_daily, aggregate_monthly
    
    print("\n--- Running Aggregation ---")
    
    # Hourly
    df_hourly = aggregate_hourly(
        df, 
        qc_col=args.qc_col,
        min_obs=AGGREGATION_CONFIG['min_5min_obs_per_hour'],
        timezone=AGGREGATION_CONFIG['timezone']
    )
    
    # Daily
    df_daily = aggregate_daily(
        df_hourly, 
        qc_col=args.qc_col,
        min_hours=AGGREGATION_CONFIG['min_hourly_obs_per_day']
    )
    
    # Monthly
    df_monthly = aggregate_monthly(
        df_daily, 
        qc_col=args.qc_col,
        require_complete=AGGREGATION_CONFIG['require_complete_months']
    )
    
    # Save files
    hourly_path = os.path.join(output_dir, 'QC_mean_hourly_data_2024.csv')
    daily_path = os.path.join(output_dir, 'QC_mean_daily_data_2024.csv')
    monthly_path = os.path.join(output_dir, 'QC_mean_monthly_data_2024.csv')
    
    print(f"\n--- Saving Results to {output_dir} ---")
    df_hourly.to_csv(hourly_path, index=False)
    print(f"Saved {hourly_path} ({len(df_hourly)} rows)")
    
    df_daily.to_csv(daily_path, index=False)
    print(f"Saved {daily_path} ({len(df_daily)} rows)")
    
    df_monthly.to_csv(monthly_path, index=False)
    print(f"Saved {monthly_path} ({len(df_monthly)} rows)")
    
    print("\n✅ Aggregation Pipeline Complete!")


if __name__ == "__main__":
    main()
