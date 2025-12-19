"""
Gantt Chart Analysis of Data Availability
Generates a layered Gantt chart showing data availability and gaps for all stations in 2024.
Reflects logic from 02a_ganttchart.py but adapted for Pandas and project structure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

def load_data():
    """Load and merge ML-ready data with station metadata."""
    print("Loading ML-ready data...")
    if not os.path.exists(settings.DATA_PATH):
        print(f"Error: Data file not found at {settings.DATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(settings.DATA_PATH)
    
    # Metadata path (using hclimatedivision file as confirmed)
    meta_path = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef_hiclimatedivision.csv')
    
    if os.path.exists(meta_path):
        print(f"Loading metadata from {meta_path}...")
        meta_df = pd.read_csv(meta_path)
        
        # Ensure ID columns are strings for merging
        df['SITE_ID'] = df['SITE_ID'].astype(str)
        meta_df['station_id'] = meta_df['station_id'].astype(str)
        
        # Merge to get full_name
        # Keep left (data) to ensure we only plot what we have
        merged = pd.merge(df, meta_df[['station_id', 'full_name']], left_on='SITE_ID', right_on='station_id', how='left')
        
        # Create display name: full_name if avail, else SITE_ID
        merged['display_name'] = merged['full_name'].fillna(merged['SITE_ID']).astype(str)
        merged['display_name'] = merged['display_name'].str.replace(r"[()]", "", regex=True)
    else:
        print(f"Warning: Metadata file not found at {meta_path}. Using SITE_ID as display name.")
        df['display_name'] = df['SITE_ID'].astype(str)
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
        # Fallback to sample_time (UTC) if LOCAL_TIME not present, but usually we prefer LOCAL_TIME
        time_col = 'sample_time'
    
    print(f"Using time column: {time_col}")
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort
    df = df.sort_values(by=['display_name', time_col])
    
    intervals = []
    
    # Process each station
    for station, group in df.groupby('display_name'):
        # Calculate time diffs
        # We look for gaps larger than threshold
        times = group[time_col].values
        
        if len(times) == 0:
            continue
            
        # Identify breaks (indices where diff > threshold)
        diffs = np.diff(times).astype('timedelta64[m]').astype(int)
        break_indices = np.where(diffs > gap_threshold_mins)[0]
        
        # Start indices of blocks are 0 and break_indices + 1
        starts = np.concatenate(([0], break_indices + 1))
        # End indices are break_indices and the last index
        ends = np.concatenate((break_indices, [len(times) - 1]))
        
        for s, e in zip(starts, ends):
            intervals.append({
                'display_name': station,
                'station_id': group.iloc[0]['SITE_ID'], # Keep ID
                'start': times[s],
                'finish': times[e]
            })
            
    return pd.DataFrame(intervals)

import numpy as np # Needed for diff above

def plot_layered_gantt(avail_df, year=2024, output_path="station_availability.png"):
    print(f"Generating Gantt chart for year {year}...")
    
    # Sort stations by ID (numeric) if possible, or name
    # ID is better for consistency
    try:
        avail_df['station_id_num'] = avail_df['station_id'].astype(int)
        unique_stations = avail_df[['display_name', 'station_id_num']].drop_duplicates().sort_values('station_id_num')
    except:
        unique_stations = avail_df[['display_name']].drop_duplicates().sort_values('display_name')
        
    stations_display = unique_stations['display_name'].tolist()
    
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
    # Vectorized plotting is hard with broken bars, so we iterate rows (or use broken_barh)
    # Using barh per interval
    # To speed up, we can group by station and use broken_barh
    
    print("Plotting intervals...")
    for name, group in avail_df.groupby('display_name'):
        if name not in y_map: continue
        y_idx = y_map[name]
        
        # Prepare list of (start, width) tuples
        xranges = []
        for _, row in group.iterrows():
            start = row['start']
            width = row['finish'] - row['start']
            xranges.append((start, width))
            
        ax.broken_barh(xranges, (y_idx - 0.35, 0.7), facecolors='tab:blue', zorder=2, edgecolor='none')

    # Formatting
    ax.set_ylim(-0.6, len(stations_display) - 0.4)
    ax.set_yticks(range(len(stations_display)))
    ax.set_yticklabels(stations_display, color='black', fontsize=10, family='monospace')
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
    output_path = os.path.join(base_fig_dir, f'station_availability_{args.year}.png')
    
    # Run pipeline
    df = load_data()
    avail_df = get_availability_intervals(df, gap_threshold_mins=args.gap_threshold)
    plot_layered_gantt(avail_df, year=args.year, output_path=output_path)

if __name__ == "__main__":
    main()
