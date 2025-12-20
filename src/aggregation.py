"""
Temporal Aggregation Module for 5-minute GOES-18 and Ground Station Data.

This module provides quality-controlled aggregation functions for converting
high-frequency (5-minute) observations to hourly, daily, and monthly means.
QC thresholds follow the methodology from Lucas et al. (2020).

QC Thresholds:
- Hourly: ≥10 of 12 possible 5-min observations
- Daily: ≥22 of 24 possible hourly observations  
- Monthly: 100% daily completeness (all days in month must be valid)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


# Default QC thresholds
DEFAULT_MIN_5MIN_OBS_PER_HOUR = 10
DEFAULT_MIN_HOURLY_OBS_PER_DAY = 22


def _get_aggregation_rules(
    df: pd.DataFrame,
    time_varying_prefixes: Tuple[str, ...] = ('CMI_', 'LWOUT', 'LST', 'HOUR_', 'DOY_', 'SZA_', 'SAA_'),
    static_prefixes: Tuple[str, ...] = ('Elevation', 'A')
) -> Dict[str, str]:
    """
    Build aggregation rules dictionary based on column prefixes.
    
    Time-varying columns are averaged, static columns take first value.
    
    Args:
        df: DataFrame to analyze
        time_varying_prefixes: Prefixes for columns to average
        static_prefixes: Prefixes for columns to take first value
        
    Returns:
        Dictionary mapping column names to aggregation functions
    """
    aggs = {}
    for col in df.columns:
        if any(col.startswith(p) for p in time_varying_prefixes):
            aggs[col] = 'mean'
        elif any(col.startswith(p) for p in static_prefixes):
            aggs[col] = 'first'
    return aggs


def aggregate_hourly(
    df: pd.DataFrame,
    time_col: str = 'LOCAL_TIME',
    site_col: str = 'SITE_ID',
    qc_col: str = 'LWOUT',
    min_obs: int = DEFAULT_MIN_5MIN_OBS_PER_HOUR,
    timezone: str = 'Pacific/Honolulu'
) -> pd.DataFrame:
    """
    Aggregate 5-minute data to hourly means with quality control.
    
    An hourly mean is only calculated if at least `min_obs` of the 12 
    possible 5-minute observations are present.
    
    Args:
        df: DataFrame with 5-minute observations
        time_col: Name of the datetime column
        site_col: Name of the station ID column
        qc_col: Column to use for counting observations
        min_obs: Minimum observations required per hour (default: 10)
        timezone: Timezone for resampling (default: Pacific/Honolulu)
        
    Returns:
        DataFrame with hourly means, filtered by QC threshold
    """
    df = df.copy()
    
    # Ensure datetime and set timezone
    df[time_col] = pd.to_datetime(df[time_col])
    try:
        df[time_col] = df[time_col].dt.tz_localize(timezone)
    except TypeError:
        # Already timezone-aware
        df[time_col] = df[time_col].dt.tz_convert(timezone)
    
    # Set time index
    df.set_index(time_col, inplace=True)
    df.index.name = 'time_hst'
    
    # Build aggregation rules
    aggs = _get_aggregation_rules(df)
    if qc_col not in aggs:
        aggs[qc_col] = 'mean'
    
    # Perform aggregation
    hourly_counts = df.groupby(site_col)[qc_col].resample('h').count()
    df_hourly = df.groupby(site_col).resample('h').agg(aggs)
    
    # Apply QC filter
    df_hourly_qc = df_hourly.where(hourly_counts >= min_obs).dropna(how='all')
    
    print(f"✅ Hourly aggregation: {len(df_hourly_qc)} records with ≥{min_obs} obs/hr")
    
    return df_hourly_qc.reset_index()


def aggregate_daily(
    df_hourly: pd.DataFrame,
    time_col: str = 'time_hst',
    site_col: str = 'SITE_ID',
    qc_col: str = 'LWOUT',
    min_hours: int = DEFAULT_MIN_HOURLY_OBS_PER_DAY
) -> pd.DataFrame:
    """
    Aggregate hourly data to daily means with quality control.
    
    A daily mean is only calculated if at least `min_hours` of the 24 
    possible hourly observations are present.
    
    Args:
        df_hourly: DataFrame with hourly observations
        time_col: Name of the datetime column
        site_col: Name of the station ID column
        qc_col: Column to use for counting observations
        min_hours: Minimum hours required per day (default: 22)
        
    Returns:
        DataFrame with daily means, filtered by QC threshold
    """
    df = df_hourly.copy()
    
    # Ensure datetime
    if time_col not in df.columns:
        df = df.reset_index()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Build aggregation rules
    aggs = _get_aggregation_rules(df)
    if qc_col not in aggs:
        aggs[qc_col] = 'mean'
    
    # Perform aggregation
    daily_counts = df.groupby(site_col).resample('d', on=time_col)[qc_col].count()
    df_daily = df.groupby(site_col).resample('d', on=time_col).agg(aggs)
    
    # Apply QC filter
    df_daily_qc = df_daily.where(daily_counts >= min_hours).dropna(how='all')
    
    print(f"✅ Daily aggregation: {len(df_daily_qc)} records with ≥{min_hours} hrs/day")
    
    return df_daily_qc.reset_index()


def aggregate_monthly(
    df_daily: pd.DataFrame,
    time_col: str = 'time_hst',
    site_col: str = 'SITE_ID',
    qc_col: str = 'LWOUT',
    require_complete: bool = True
) -> pd.DataFrame:
    """
    Aggregate daily data to monthly means with quality control.
    
    By default, a monthly mean is only calculated if 100% of the days
    in that month have valid daily observations.
    
    Args:
        df_daily: DataFrame with daily observations
        time_col: Name of the datetime column
        site_col: Name of the station ID column
        qc_col: Column to use for counting observations
        require_complete: If True, require all days in month (default: True)
        
    Returns:
        DataFrame with monthly means, filtered by QC threshold
    """
    df = df_daily.copy()
    
    # Ensure datetime
    if time_col not in df.columns:
        df = df.reset_index()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Build aggregation rules
    aggs = _get_aggregation_rules(df)
    if qc_col not in aggs:
        aggs[qc_col] = 'mean'
    
    # Count days per month
    monthly_counts = df.groupby(site_col).resample('ME', on=time_col)[qc_col].count()
    
    # Get expected days in each month
    days_in_month = monthly_counts.index.get_level_values(time_col).days_in_month
    
    # Perform aggregation
    df_monthly = df.groupby(site_col).resample('ME', on=time_col).agg(aggs)
    
    if require_complete:
        # Only keep months with 100% daily completeness
        complete_mask = monthly_counts.values == days_in_month
        df_monthly_qc = df_monthly[complete_mask].dropna(how='all')
        print(f"✅ Monthly aggregation: {len(df_monthly_qc)} complete months (100% daily)")
    else:
        df_monthly_qc = df_monthly.dropna(how='all')
        print(f"✅ Monthly aggregation: {len(df_monthly_qc)} records")
    
    return df_monthly_qc.reset_index()


def run_aggregation_pipeline(
    df: pd.DataFrame,
    time_col: str = 'LOCAL_TIME',
    site_col: str = 'SITE_ID',
    qc_col: str = 'LWOUT',
    min_5min_per_hour: int = DEFAULT_MIN_5MIN_OBS_PER_HOUR,
    min_hours_per_day: int = DEFAULT_MIN_HOURLY_OBS_PER_DAY,
    require_complete_months: bool = True,
    timezone: str = 'Pacific/Honolulu'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full aggregation pipeline from 5-minute to hourly, daily, and monthly.
    
    Args:
        df: DataFrame with 5-minute observations
        time_col: Name of the datetime column
        site_col: Name of the station ID column
        qc_col: Column to use for QC counting
        min_5min_per_hour: Minimum 5-min obs per hour (default: 10)
        min_hours_per_day: Minimum hours per day (default: 22)
        require_complete_months: Require 100% daily completeness (default: True)
        timezone: Timezone for resampling
        
    Returns:
        Tuple of (hourly_df, daily_df, monthly_df)
    """
    print("=" * 60)
    print("Starting Quality-Controlled Aggregation Pipeline")
    print("=" * 60)
    print(f"Input: {len(df)} observations")
    print(f"QC Thresholds: ≥{min_5min_per_hour}/hr, ≥{min_hours_per_day}/day, complete months={require_complete_months}")
    print()
    
    # Hourly
    df_hourly = aggregate_hourly(
        df, time_col=time_col, site_col=site_col, qc_col=qc_col,
        min_obs=min_5min_per_hour, timezone=timezone
    )
    
    # Daily
    df_daily = aggregate_daily(
        df_hourly, time_col='time_hst', site_col=site_col, qc_col=qc_col,
        min_hours=min_hours_per_day
    )
    
    # Monthly
    df_monthly = aggregate_monthly(
        df_daily, time_col='time_hst', site_col=site_col, qc_col=qc_col,
        require_complete=require_complete_months
    )
    
    print()
    print("=" * 60)
    print("Pipeline Complete")
    print(f"  Hourly:  {len(df_hourly)} records")
    print(f"  Daily:   {len(df_daily)} records")
    print(f"  Monthly: {len(df_monthly)} records")
    print("=" * 60)
    
    return df_hourly, df_daily, df_monthly


if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Aggregate 5-minute data to hourly/daily/monthly')
    parser.add_argument('--input', '-i', required=True, help='Input CSV path (5-minute data)')
    parser.add_argument('--output', '-o', required=True, help='Output directory for aggregated CSVs')
    parser.add_argument('--qc-col', default='LWOUT', help='Column for QC counting (default: LWOUT)')
    parser.add_argument('--min-hourly', type=int, default=10, help='Min 5-min obs per hour (default: 10)')
    parser.add_argument('--min-daily', type=int, default=22, help='Min hours per day (default: 22)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Run pipeline
    df_hourly, df_daily, df_monthly = run_aggregation_pipeline(
        df,
        qc_col=args.qc_col,
        min_5min_per_hour=args.min_hourly,
        min_hours_per_day=args.min_daily
    )
    
    # Save outputs
    os.makedirs(args.output, exist_ok=True)
    
    hourly_path = os.path.join(args.output, 'QC_mean_hourly_data.csv')
    daily_path = os.path.join(args.output, 'QC_mean_daily_data.csv')
    monthly_path = os.path.join(args.output, 'QC_mean_monthly_data.csv')
    
    df_hourly.to_csv(hourly_path, index=False)
    df_daily.to_csv(daily_path, index=False)
    df_monthly.to_csv(monthly_path, index=False)
    
    print(f"\n✅ Saved aggregated data to {args.output}/")
