"""
Geographic Analysis Maps for LST Model Performance.

Generates island-specific maps showing model performance metrics (RMSE, Bias, etc.)
and difference metrics (e.g., BLAM vs BLM) for Hawaiian stations.
"""

import polars as pl
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import sys
import argparse
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# Suppress runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Bounding boxes for islands
ISLAND_BOUNDS = {
    'kauai': {'name': 'Kauaʻi', 'min_lon': -159.8457, 'max_lon': -159.2388, 'min_lat': 21.8321, 'max_lat': 22.2842, 'lon_mult': 0.2, 'lat_mult': 0.2},
    'oahu': {'name': 'Oʻahu', 'min_lon': -158.3051, 'max_lon': -157.6349, 'min_lat': 21.245298, 'max_lat': 21.7142899, 'lon_mult': 0.2, 'lat_mult': 0.2},
    'maui_nui': {'name': 'Maui Nui', 'min_lon': -157.3505, 'max_lon': -155.8454, 'min_lat': 20.4649, 'max_lat': 21.2964, 'lon_mult': 0.5, 'lat_mult': 0.4},
    'hawaii': {'name': 'Island of Hawaiʻi', 'min_lon': -156.2519, 'max_lon': -154.7578, 'min_lat': 18.8874, 'max_lat': 20.3053, 'lon_mult': 0.5, 'lat_mult': 0.5}
}

# --- Helper Functions ---

def calculate_station_metrics(df, model_name):
    """Calculate per-station metrics for a given model DF using Polars."""
    if df.height == 0:
        return pl.DataFrame()
    
    # We need to calculate metrics per station and per condition
    # Conditions: All (no filter), Clear (ACMC_BCM=0), Cloudy (ACMC_BCM=1)
    
    metrics_list = []
    
    conditions = {
        'all': None,
        'clear': pl.col('ACMC_BCM') == 0,
        'cloudy': pl.col('ACMC_BCM') == 1
    }
    
    for cond_name, filter_expr in conditions.items():
        subset = df if filter_expr is None else df.filter(filter_expr)
        
        if subset.height == 0:
            continue
            
        # Calculate residuals for easy aggregation
        subset = subset.with_columns([
            (pl.col('LST_pred') - pl.col('LST_true')).alias('residual')
        ])
        
        # Aggregation
        agg_df = subset.group_by('station_id').agg([
            pl.col('residual').mean().alias(f'mean_residual_{model_name}_{cond_name}'),
            pl.col('residual').median().alias(f'median_residual_{model_name}_{cond_name}'),
            (pl.col('residual') ** 2).mean().sqrt().alias(f'rmse_{model_name}_{cond_name}'),
            (pl.col('LST_pred').std() - pl.col('LST_true').std()).alias(f'std_dev_diff_{model_name}_{cond_name}')
        ])
        
        metrics_list.append(agg_df)
    
    if not metrics_list:
        return pl.DataFrame()
        
    # Join all metrics using full join with coalesce to avoid duplicate station_id columns
    final_df = metrics_list[0]
    for m in metrics_list[1:]:
        final_df = final_df.join(m, on='station_id', how='full')
        # Coalesce station_id with station_id_right and drop the right column
        if 'station_id_right' in final_df.columns:
            final_df = final_df.with_columns(
                pl.coalesce(['station_id', 'station_id_right']).alias('station_id')
            ).drop('station_id_right')
        
    return final_df

def load_predictions(model_name):
    """Load predictions for a specific model."""
    # Try multiple standard paths
    paths = [
        os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, 'loso_ALL_predictions_for_plotting.csv'),
        os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv'),
        os.path.join(settings.TEMP_PRED_DIR, f'pred_{model_name}.csv') # Fallback for temp
    ]
    
    for p in paths:
        if os.path.exists(p):
            print(f"  Loaded {model_name} from {p}")
            df = pl.read_csv(p)
            # Ensure station_id is string and present
            # Rename SITE_ID -> station_id if needed
            if 'station_id' not in df.columns and 'SITE_ID' in df.columns:
                df = df.rename({'SITE_ID': 'station_id'})
            
            df = df.with_columns(
                pl.col('station_id').cast(pl.Utf8).str.zfill(4)
            )
            return df
            
    print(f"  Warning: Predictions for {model_name} not found.")
    return pl.DataFrame()

# --- Plotting Functions ---

def format_lon(x, pos):
    return f'{abs(x):.1f}°W'

def format_lat(y, pos):
    return f'{abs(y):.1f}°N'

def add_scalebar(ax, bbox):
    mid_lat_rad = np.radians((bbox['min_lat'] + bbox['max_lat']) / 2)
    m_per_deg_lon = 111320 * np.cos(mid_lat_rad)
    scalebar = ScaleBar(
        m_per_deg_lon, "m", location="lower left",
        frameon=False, color="black",
        font_properties={'size': 10, 'family': 'serif'}
    )
    ax.add_artist(scalebar)

def add_north_arrow(ax):
    ax.text(
        0.9, 0.9, 'N\n▲', transform=ax.transAxes,
        ha='center', va='center', fontsize=14, fontweight='bold',
        path_effects=[path_effects.withStroke(linewidth=3, foreground='white')]
    )

def format_axes(ax, lon_mult=0.2, lat_mult=0.1):
    ax.xaxis.set_major_locator(mticker.MultipleLocator(lon_mult))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(lat_mult))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_tick_params(rotation=0)  # Horizontal labels to avoid overlap
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')

def plot_island_map(ax, gdf_data, coastline_data, bbox, data_col, norm, cmap, title, lon_mult=0.2, lat_mult=0.1):
    ax.set_xlim(bbox['min_lon'], bbox['max_lon'])
    ax.set_ylim(bbox['min_lat'], bbox['max_lat'])

    # Clip data to bbox
    try:
        filtered_coastline = coastline_data.cx[bbox['min_lon']:bbox['max_lon'], bbox['min_lat']:bbox['max_lat']]
        filtered_gdf = gdf_data.cx[bbox['min_lon']:bbox['max_lon'], bbox['min_lat']:bbox['max_lat']]
    except Exception:
        # Fallback if spatial index fails
        filtered_coastline = coastline_data
        filtered_gdf = gdf_data

    # Plot Coastline
    filtered_coastline.plot(ax=ax, facecolor='white', edgecolor='darkgray', linewidth=0.75, zorder=1)

    # Plot Stations
    if not filtered_gdf.empty and data_col in filtered_gdf.columns:
        filtered_gdf.plot(
            ax=ax,
            column=data_col,
            cmap=cmap,
            norm=norm,
            s=110,
            edgecolor='black',
            linewidth=0.75,
            zorder=5,
            legend=False
        )

    # Attribution
    ax.text(0.02, 0.02, "Sources: State of Hawaii GIS, USGS.", 
            transform=ax.transAxes, fontsize=6, color='gray', ha='left', va='bottom', zorder=10)

    # Map Elements
    add_scalebar(ax, bbox)
    add_north_arrow(ax)
    format_axes(ax, lon_mult, lat_mult)
    ax.set_title(title, fontsize=16, fontweight='bold')


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate Geographic Maps of Model Metrics")
    parser.add_argument('--models', nargs='+', default=['BLM', 'BLAM', 'CIM', 'CIAM'], help="Models to analyze")
    parser.add_argument('--baseline', default='BLM', help="Baseline model for difference calculation")
    parser.add_argument('--compare', default='BLAM', help="Comparison model for difference calculation")
    args = parser.parse_args()

    # Define Paths
    stations_path = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef_hiclimatedivision.csv')
    coastline_path = os.path.join(settings.BASE_DIR, 'datasets/other/Coastline/Coastline.shp')
    output_dir = os.path.join(settings.BASE_DIR, 'figures/maps')
    os.makedirs(output_dir, exist_ok=True)

    print("--- Geographic Map Generation ---")
    
    # 1. Load Station Metadata
    print(f"Loading stations from {stations_path}...")
    if not os.path.exists(stations_path):
        print("Error: Station file not found.")
        sys.exit(1)
        
    stations_df = pl.read_csv(stations_path)
    stations_df = stations_df.with_columns(
        pl.col('station_id').cast(pl.Utf8).str.zfill(4)
    )
    # Ensure standard lon/lat names
    if 'lng' in stations_df.columns:
        stations_df = stations_df.rename({'lng': 'lon'})

    # 2. Load Coastline
    print(f"Loading coastline from {coastline_path}...")
    if not os.path.exists(coastline_path):
        print("Error: Coastline shapefile not found.")
        sys.exit(1)
        
    coastline_gdf = gpd.read_file(coastline_path).to_crs(epsg=4326)

    # 3. Load Predictions & Calculate Metrics
    print("Loading predictions and calculating metrics...")
    
    metrics_frames = []
    
    # Calculate for Baseline
    base_df = load_predictions(args.baseline)
    if base_df.height > 0:
        metrics_frames.append(calculate_station_metrics(base_df, args.baseline))
    
    # Calculate for Compare
    comp_df = load_predictions(args.compare)
    if comp_df.height > 0:
        metrics_frames.append(calculate_station_metrics(comp_df, args.compare))
        
    if not metrics_frames:
        print("No metrics could be calculated. Modeling results missing?")
        sys.exit(1)
        
    # Join metrics with coalesce to handle full join key duplication
    full_metrics_df = metrics_frames[0]
    for m in metrics_frames[1:]:
        full_metrics_df = full_metrics_df.join(m, on='station_id', how='full')
        # Coalesce station_id with station_id_right and drop the right column
        if 'station_id_right' in full_metrics_df.columns:
            full_metrics_df = full_metrics_df.with_columns(
                pl.coalesce(['station_id', 'station_id_right']).alias('station_id')
            ).drop('station_id_right')
    
    # Calculate Differences (Compare - Baseline)
    # diff = exp - ctrl
    
    metrics_to_diff = ['mean_residual', 'median_residual', 'rmse', 'std_dev_diff']
    conditions = ['all', 'clear', 'cloudy']
    
    diff_cols = []
    for m in metrics_to_diff:
        for c in conditions:
            col_exp = f'{m}_{args.compare}_{c}'
            col_ctrl = f'{m}_{args.baseline}_{c}'
            col_diff = f'{m}_diff_{args.compare}_vs_{args.baseline}_{c}'
            
            if col_exp in full_metrics_df.columns and col_ctrl in full_metrics_df.columns:
                full_metrics_df = full_metrics_df.with_columns(
                    (pl.col(col_exp) - pl.col(col_ctrl)).alias(col_diff)
                )
                diff_cols.append(col_diff)
    
    print(f"Calculated differences for {full_metrics_df.height} stations.")

    # 4. Prepare GeoDataFrame
    # Convert Polars DF to Pandas for GeoPandas
    full_metrics_pd = full_metrics_df.to_pandas()
    stations_pd = stations_df.to_pandas()
    
    # Merge metrics with stations for geometry
    # We left join metrics to stations to get lat/lon
    gdf_pd = stations_pd.merge(full_metrics_pd, on='station_id', how='left')
    
    gdf = gpd.GeoDataFrame(
        gdf_pd,
        geometry=gpd.points_from_xy(gdf_pd.lon, gdf_pd.lat),
        crs="EPSG:4326"
    )
    
    # 5. Generate Maps
    print("Generating maps...")
    
    # Configuration matches template
    metric_display_names = {
        'mean_residual': '|Mean Residual|',
        'median_residual': '|Median Residual|',
        'rmse': 'RMSE',
        'std_dev_diff': 'STD Diff'
    }
    
    cmap = plt.cm.RdBu_r  # Red-Blue diverging
    
    # Loop through metrics to plot
    for m_key, m_name in metric_display_names.items():
        for c_key in conditions:
            
            data_col = f'{m_key}_diff_{args.compare}_vs_{args.baseline}_{c_key}'
            if data_col not in gdf.columns:
                continue
            
            if gdf[data_col].isnull().all():
                print(f"Skipping {data_col} (All NaN)")
                continue

            print(f"Plotting {data_col}...")
            
            # Determine Global Vmax for this metric/condition
            vmax = gdf[data_col].abs().max()
            if pd.isna(vmax) or vmax == 0: vmax = 1.0
            vmin = -vmax
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            # Format condition name for title (Clear-Sky, Cloudy-Sky, All-Sky)
            cond_display = f"{c_key.title()}-Sky" if c_key != 'all' else "All-Sky"
            title = f"{m_name} Difference by Station\n({args.compare}) - ({args.baseline}) ({cond_display})"
            filename = f"map_{m_key}_{args.compare}_vs_{args.baseline}_{c_key}.png"
            filepath = os.path.join(output_dir, filename)

            # Setup Figure
            fig = plt.figure(figsize=(22, 14), facecolor='white')
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.5, 0.15], width_ratios=[1, 1, 1.4], wspace=0.15, hspace=0.25)
            
            gs_top_left = gs[0, 0:2].subgridspec(1, 2, wspace=0.05)
            ax_kauai = fig.add_subplot(gs_top_left[0, 0])
            ax_oahu = fig.add_subplot(gs_top_left[0, 1])
            ax_maui = fig.add_subplot(gs[1, 0:2])
            ax_hawaii = fig.add_subplot(gs[0:2, 2])
            cbar_ax = fig.add_subplot(gs[2, :])

            # Plot Islands
            plot_island_map(ax_kauai, gdf, coastline_gdf, ISLAND_BOUNDS['kauai'], data_col, norm, cmap, "Kauaʻi")
            plot_island_map(ax_oahu, gdf, coastline_gdf, ISLAND_BOUNDS['oahu'], data_col, norm, cmap, "Oʻahu")
            plot_island_map(ax_maui, gdf, coastline_gdf, ISLAND_BOUNDS['maui_nui'], data_col, norm, cmap, "Maui Nui")
            plot_island_map(ax_hawaii, gdf, coastline_gdf, ISLAND_BOUNDS['hawaii'], data_col, norm, cmap, "Island of Hawaiʻi")

            # Shared Elements
            fig.suptitle(title, fontsize=24, fontweight='bold', y=0.99)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=0.1)
            cbar.set_label(f"Residual Difference (K)", fontsize=13, fontweight='bold')
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved to {filepath}")

    print("Done.")

if __name__ == "__main__":
    main()
