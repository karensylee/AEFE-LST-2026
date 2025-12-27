"""
Gantt Chart Analysis of Data Availability
Generates a layered Gantt chart showing data availability and gaps for all stations in 2024.
Reflects logic from 02a_ganttchart.py but adapted for Polars and project structure.
"""

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# Matplotlib configuration (matching density_plots.py template)
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_data():
    """Load and merge ML-ready data with station metadata."""
    print("Loading ML-ready data...")
    if not os.path.exists(settings.DATA_PATH):
        print(f"Error: Data file not found at {settings.DATA_PATH}")
        sys.exit(1)
        
    df = pl.read_csv(settings.DATA_PATH)
    
    # Metadata path
    meta_path = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef_hiclimatedivision.csv')
    
    if os.path.exists(meta_path):
        print(f"Loading metadata from {meta_path}...")
        meta_df = pl.read_csv(meta_path)
        
        # Ensure ID columns are strings for joining
        df = df.with_columns(pl.col('SITE_ID').cast(pl.Utf8))
        meta_df = meta_df.with_columns(pl.col('station_id').cast(pl.Utf8))
        
        # Join to get full_name
        # Keep left (data)
        merged = df.join(
            meta_df.select(['station_id', 'full_name']), 
            left_on='SITE_ID', 
            right_on='station_id', 
            how='left'
        )
        
        # Create display name: full_name if avail, else SITE_ID
        merged = merged.with_columns(
            pl.coalesce([pl.col('full_name'), pl.col('SITE_ID')]).alias('display_name')
        )
        merged = merged.with_columns(
            pl.col('display_name').str.replace_all(r"[()]", "")
        )
    else:
        print(f"Warning: Metadata file not found at {meta_path}. Using SITE_ID as display name.")
        df = df.with_columns(
            pl.col('SITE_ID').cast(pl.Utf8).alias('display_name')
        )
        merged = df

    return merged

def get_availability_intervals(df, gap_threshold_mins=10):
    """
    Identify continuous blocks of data availability.
    A block breaks if the time difference to the next sample > gap_threshold_mins.
    """
    print("Calculating availability intervals...")
    
    # Ensure datetime
    if 'LOCAL_TIME' in df.columns:
        time_col = 'LOCAL_TIME'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        time_col = 'sample_time'
    
    print(f"Using time column: {time_col}")
    
    # Convert to datetime if not already
    # Note: strptime checks might be needed depending on format, but usually automatic or from clean CSV
    # Convert to datetime if not already
    # Note: strptime checks might be needed depending on format, but usually automatic or from clean CSV
    # Convert to datetime with explicit format to avoid ComputeError
    # Data format: 2024-05-01 12:40:00+00:00
    try:
        df = df.with_columns(
            pl.col(time_col).str.to_datetime(format="%Y-%m-%d %H:%M:%S%z").alias(time_col)
        )
    except Exception:
        # Fallback for other formats or if already mixed
        print("Warning: Standard format parsing failed, trying automatic inference...")
        df = df.with_columns(pl.col(time_col).str.to_datetime(time_zone='UTC', strict=False).alias(time_col))

    
    # Sort by display_name then time
    df = df.sort(['display_name', time_col])
    
    # Calculate time difference in minutes
    # We want to find where (time - prev_time) > threshold
    
    # Window function over each station
    # Calculate diff in minutes
    df = df.with_columns([
        (pl.col(time_col).diff().dt.total_minutes().fill_null(0)).over('display_name').alias('diff_mins')
    ])
    
    # Identify starts of new blocks
    # A new block starts if diff_mins > threshold OR if it's the first row for the station
    # First row per group will have null diff from diff(), but we filled with 0. 
    # Actually, diff() result is null for the first element. fill_null(0) makes it 0.
    # But wait, the first element of a group shouldn't be a gap unless we treat it so.
    # Let's say:
    # is_start = (diff_mins > threshold) | (row_number == 0 in group)
    # We can use group identifiers (run-length encoding style)
    
    # Alternative:
    # 1. Flag gaps: diff > threshold
    # 2. Cumulative sum of flags gives us a "block_id"
    # 3. Group by station and block_id, then get min(time) and max(time)
    
    gap_threshold = gap_threshold_mins
    
    # We need to handle the first row of each group correctly.
    # diff() gives null for first.
    # fill_null(gap_threshold + 1) makes the first row start a new block? No, first row is always start of a block.
    
    df = df.with_columns([
        (pl.col('diff_mins').fill_null(gap_threshold + 1) > gap_threshold).alias('is_new_block')
    ])
    
    # Create block identifier
    df = df.with_columns([
        pl.col('is_new_block').cum_sum().over('display_name').alias('block_id')
    ])
    
    # Group by station and block_id to find start/end
    intervals = df.group_by(['display_name', 'SITE_ID', 'block_id']).agg([
        pl.col(time_col).min().alias('start'),
        pl.col(time_col).max().alias('finish'),
        pl.col('SITE_ID').first().alias('station_id') # Redundant but kept for structure
    ])
    
    # Filter out single points if strictly needed, but interval (t, t) is width 0 and won't show on broken_barh anyway unless we add width.
    # Usually we want valid intervals. Width 0 might be invisible.
    
    return intervals

def plot_layered_gantt(avail_df, year=2024, output_path=None):
    # If no output_path specified, use default location in figures/analysis
    if output_path is None:
        base_fig_dir = os.path.join(settings.BASE_DIR, 'figures/analysis')
        os.makedirs(base_fig_dir, exist_ok=True)
        output_path = os.path.join(base_fig_dir, f'station_availability_{year}.jpg')
    
    print(f"Generating Gantt chart for year {year}...")
    
    # Check if empty
    if avail_df.height == 0:
        print("No availability data found.")
        return

    # Sort stations
    # If station_id is numeric string, cast to int for sorting
    try:
        avail_df = avail_df.with_columns(pl.col('station_id').cast(pl.Int32).alias('station_id_num'))
        unique_stations = avail_df.select(['display_name', 'station_id_num']).unique().sort('station_id_num')
    except:
        unique_stations = avail_df.select(['display_name']).unique().sort('display_name')
        
    stations_display = unique_stations['display_name'].to_list()
    
    # Map name to y-axis index
    y_map = {name: i for i, name in enumerate(stations_display)}
    
    # Define year bounds
    year_start = pd.Timestamp(f'{year}-01-01')
    year_end = pd.Timestamp(f'{year}-12-31 23:59:59')
    full_duration = year_end - year_start
    
    # Setup Plot
    fig_height = max(12, len(stations_display) * 0.4)
    fig, ax = plt.subplots(figsize=(22, fig_height), facecolor='white')
    
    # 1. Background (Red = Missing)
    for i in range(len(stations_display)):
        ax.barh(i, full_duration, left=year_start, height=0.7, color='tab:red', zorder=1)
        
    # 2. Foreground (Blue = Available)
    print("Plotting intervals...")
    
    # Group by display_name to collect intervals
    # We can iterate through the Polars df or convert to pandas for iterating.
    # Iterating polars rows is slower than pandas but for plotting this is not the bottleneck.
    # Let's simple iterate over groups from polars
    
    # Convert to pandas for easier iteration with iterrows or groupby if comfortable, 
    # but let's stick to polars logic -> convert relevant cols to lists
    
    # We want list of (start, width) for each station
    
    # Calculate widths in the dataframe
    # Duration in matplotlib dates is days? No, we pass Timestamps.
    # broken_barh expects xranges as (start_time, duration)
    # If we pass dates as start, duration must be timedelta.
    
    # It might be easier to just loop over the unique stations and filter the dataframe
    
    # Pre-calculate widths
    # We need to make sure 'finish' and 'start' are datetimes
    
    # Using pandas for the plotting loop is convenient for matplotlib compatibility (timestamps)
    # Convert result to pandas for plotting
    avail_pd = avail_df.to_pandas()
    
    for name, group in avail_pd.groupby('display_name'):
        if name not in y_map: continue
        y_idx = y_map[name]
        
        xranges = []
        for _, row in group.iterrows():
            start = row['start']
            width = row['finish'] - row['start']
            xranges.append((start, width))
            
        ax.broken_barh(xranges, (y_idx - 0.35, 0.7), facecolors='tab:blue', zorder=2, edgecolor='none')

    # Formatting
    ax.set_ylim(-0.6, len(stations_display) - 0.4)
    ax.set_yticks(range(len(stations_display)))
    ax.set_yticklabels(stations_display, color='black', fontsize=10, family='serif')
    ax.set_ylabel("Station Name", color='black', fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Date ({year})", color='black', fontsize=14, fontweight='bold')
    
    # Date Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Grid & Legend
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='black', zorder=0)
    
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='tab:blue', label='Available'),
        plt.Rectangle((0, 0), 1, 1, color='tab:red', label='Gap / Missing')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Gantt chart to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Data Availability Gantt Chart")
    parser.add_argument('--year', type=int, default=2024, help="Year to visualize")
    parser.add_argument('--gap_threshold', type=int, default=10, help="Gap threshold in minutes (default 10)")
    args = parser.parse_args()
    
    # Setup output dir
    base_fig_dir = os.path.join(settings.BASE_DIR, 'figures/analysis')
    os.makedirs(base_fig_dir, exist_ok=True)
    output_path = os.path.join(base_fig_dir, f'station_availability_{args.year}.jpg')
    
    # Run pipeline
    df = load_data()
    avail_df = get_availability_intervals(df, gap_threshold_mins=args.gap_threshold)
    plot_layered_gantt(avail_df, year=args.year, output_path=output_path)

if __name__ == "__main__":
    main()
