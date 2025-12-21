"""
Station Metrics Heatmap Analysis
Generates heatmaps comparing model performance across stations.
Supports ordering by: station_id (default), elevation, or climate division.
"""

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

# =============================================================================
# CONSTANTS
# =============================================================================

# Hawaii Climate Divisions color mapping (by Id)
DIVISION_COLORS = {
    1: "#b8d7e9", 2: "#4a92c2", 3: "#5eb254", 4: "#5cb355",
    5: "#fcaeac", 6: "#e84848", 7: "#fdcb8c", 8: "#ff9934",
    9: "#d3c0df", 10: "#8864ae", 11: "#c27a54", 12: "#AAAAAA"
}

# Hawaii Climate Divisions name mapping (by Id)
DIVISION_NAMES = {
    1: "Leeward Kauaʻi", 2: "Windward Kauaʻi", 3: "Waianae", 4: "Koʻolau",
    5: "Leeward Maui Nui", 6: "Windward Maui Nui", 7: "Leeward Kohala", 8: "Windward Kohala",
    9: "Kona", 10: "Hawaiʻi Mauka", 11: "Kaʻu", 12: "Hilo"
}

# Column mappings for heatmap display labels
COLUMN_LABELS = {
    'mean_residual_diff_emb_vs_base_all': 'Mean Resid.\nBLAM vs BLM (All-Sky)',
    'mean_residual_diff_emb_vs_base_clear': 'Mean Resid.\nBLAM vs BLM (Clear)',
    'mean_residual_diff_emb_vs_base_cloudy': 'Mean Resid.\nBLAM vs BLM (Cloudy)',
    'median_residual_diff_emb_vs_base_all': 'Median Resid.\nBLAM vs BLM (All-Sky)',
    'median_residual_diff_emb_vs_base_clear': 'Median Resid.\nBLAM vs BLM (Clear)',
    'median_residual_diff_emb_vs_base_cloudy': 'Median Resid.\nBLAM vs BLM (Cloudy)',
    'rmse_diff_emb_vs_base_all': 'RMSE\nBLAM vs BLM (All-Sky)',
    'rmse_diff_emb_vs_base_clear': 'RMSE\nBLAM vs BLM (Clear)',
    'rmse_diff_emb_vs_base_cloudy': 'RMSE\nBLAM vs BLM (Cloudy)',
    'std_dev_diff_diff_emb_vs_base_all': 'STD\nBLAM vs BLM (All-Sky)',
    'std_dev_diff_diff_emb_vs_base_clear': 'STD\nBLAM vs BLM (Clear)',
    'std_dev_diff_diff_emb_vs_base_cloudy': 'STD\nBLAM vs BLM (Cloudy)',
}


def load_station_metadata():
    """Load station metadata including climate divisions and elevation."""
    stations_path = os.path.join(
        settings.BASE_DIR, 
        'datasets/stations/stations_aef_hiclimatedivision.csv'
    )
    
    # Only load the columns we need for metadata
    cols_to_load = ['station_id', 'elevation', 'Id', 'HICLIMATEDIVISION']
    stations_df = pl.read_csv(stations_path, columns=cols_to_load)
    
    # Ensure strings
    stations_df = stations_df.with_columns(pl.col('station_id').cast(pl.Utf8))
    
    return stations_df


def load_predictions(model_type):
    """
    Load prediction data from aggregated ALL_predictions file.
    
    Args:
        model_type: 'BLM', 'BLAM', etc.
        
    Returns:
        DataFrame with columns: station_id, LST_true, LST_pred, ACMC_BCM
    """
    # Try aggregated file first
    agg_path = os.path.join(settings.OUTPUT_DIR, 'xgb', model_type, f'{model_type}_ALL_predictions.csv')
    
    if os.path.exists(agg_path):
        return pl.read_csv(agg_path)
    
    # Fallback to individual files in loso_temp_predictions
    pred_dir = os.path.join(settings.OUTPUT_DIR, 'xgb', model_type, 'loso_temp_predictions')
    all_preds = []
    
    if not os.path.exists(pred_dir):
        raise ValueError(f"Predictions not found for {model_type}. Looked for:\n  - {agg_path}\n  - {pred_dir}")
        
    for f in os.listdir(pred_dir):
        if f.startswith('preds_') and f.endswith('.csv'):
            df = pl.read_csv(os.path.join(pred_dir, f))
            all_preds.append(df)
    
    if not all_preds:
        raise ValueError(f"No prediction files found in {pred_dir}")
    
    return pl.concat(all_preds)


def calculate_station_metrics(preds_df, stations_df):
    """
    Calculate per-station metrics for all sky conditions.
    
    Args:
        preds_df: DataFrame with predictions
        stations_df: DataFrame with station metadata
        
    Returns:
        DataFrame with per-station metrics
    """
    # Conditions: All, Clear (sky_condition=0), Cloudy (sky_condition=1)
    # Check columns
    cols = preds_df.columns
    if 'LST_true' in cols: preds_df = preds_df.rename({'LST_true': 'true'})
    if 'LST_pred' in cols: preds_df = preds_df.rename({'LST_pred': 'pred'})
    if 'ACMC_BCM' in cols: preds_df = preds_df.rename({'ACMC_BCM': 'sky_condition'})
    
    # Calculate Residuals
    preds_df = preds_df.with_columns([
        (pl.col('pred') - pl.col('true')).alias('residual')
    ])
    
    metrics_list = []
    
    conditions = {
        'all': None,
        'clear': pl.col('sky_condition') == 0,
        'cloudy': pl.col('sky_condition') == 1
    }
    
    for cond_name, filter_expr in conditions.items():
        subset = preds_df if filter_expr is None else preds_df.filter(filter_expr)
        
        # Aggregations
        agg = subset.group_by('station_id').agg([
            pl.col('residual').mean().alias(f'mean_residual_{cond_name}'),
            pl.col('residual').median().alias(f'median_residual_{cond_name}'),
            (pl.col('residual')**2).mean().sqrt().alias(f'rmse_{cond_name}'),
            pl.col('residual').std().alias(f'std_dev_diff_{cond_name}'), # Naming convention from original: std of resids
            pl.len().alias(f'n_{cond_name}')
        ])
        metrics_list.append(agg)
        
    # Join all
    metrics_df = metrics_list[0]
    for m in metrics_list[1:]:
        metrics_df = metrics_df.join(m, on='station_id', how='outer')
        
    # Join metadata
    metrics_df = metrics_df.join(stations_df, on='station_id', how='left')
    
    return metrics_df


def calculate_difference_metrics(blm_metrics, blam_metrics):
    """
    Calculate the difference between BLAM and BLM metrics.
    Negative values = BLAM is better (smaller error).
    
    Args:
        blm_metrics: DataFrame with BLM per-station metrics
        blam_metrics: DataFrame with BLAM per-station metrics
        
    Returns:
        DataFrame with difference metrics
    """
    # Join on station_id
    # We need to preserve metadata from one of them (they should be same)
    
    # Prefix columns to avoid collision if necessary, but we are subtracting specific cols
    
    joined = blm_metrics.join(blam_metrics, on='station_id', suffix='_blam')
    
    # Original naming: blm_metrics has 'mean_residual_all', blam has 'mean_residual_all_blam' (if suffix applied)
    # Actually join adds suffix to right table cols if collision
    # blm cols: 'mean_residual_all', etc.
    # blam cols: 'mean_residual_all_blam'
    
    metrics_to_diff = ['mean_residual', 'median_residual', 'rmse', 'std_dev_diff']
    conditions = ['all', 'clear', 'cloudy']
    
    diff_exprs = []
    
    for metric in metrics_to_diff:
        for condition in conditions:
            col_base = f'{metric}_{condition}'
            col_comp = f'{metric}_{condition}_blam'
            col_diff = f'{metric}_diff_emb_vs_base_{condition}'
            
            diff_exprs.append(
                (pl.col(col_comp) - pl.col(col_base)).alias(col_diff)
            )
            
    # Keep metadata
    meta_cols = ['elevation', 'Id', 'HICLIMATEDIVISION']
    # If they collided, they might be renamed. Elevation is in both. 
    # 'elevation' (left), 'elevation_blam' (right).
    
    diff_df = joined.select(
        [pl.col('station_id')] + 
        [pl.col(c) for c in meta_cols if c in joined.columns] +
        diff_exprs
    )
    
    return diff_df


def sort_dataframe(df, order_by='station_id', ascending=True):
    """
    Sort dataframe by specified column.
    
    Args:
        df: DataFrame to sort
        order_by: 'station_id', 'elevation', or 'division'
        ascending: Sort order
        
    Returns:
        Sorted DataFrame
    """
    if order_by == 'station_id':
        return df.sort('station_id', descending=not ascending)
    elif order_by == 'elevation':
        return df.sort('elevation', descending=not ascending)
    elif order_by == 'division':
        return df.sort('Id', descending=not ascending)
    else:
        raise ValueError(f"Unknown order_by value: {order_by}")


def generate_heatmap(diff_df, order_by='station_id', ascending=True, 
                     output_dir=None, show_plot=True):
    """
    Generate and save the comparative heatmap.
    
    Args:
        diff_df: Polars DataFrame with difference metrics
        order_by: 'station_id', 'elevation', or 'division'
        ascending: Sort order
        output_dir: Directory to save the figure
        show_plot: Whether to display the plot
    """
    print(f"\n--- Generating Heatmap (ordered by {order_by}, {'ascending' if ascending else 'descending'}) ---")
    
    # Sort data
    sorted_df = sort_dataframe(diff_df, order_by=order_by, ascending=ascending)
    
    # Convert to Pandas for plotting
    sorted_pd = sorted_df.to_pandas()
    
    # Set index for heatmap labeling
    sorted_pd.set_index('station_id', inplace=True)
    
    # Create plot labels based on ordering
    if order_by == 'division':
        plot_labels = [
            f"{idx} ({DIVISION_NAMES.get(row.Id, 'Unknown')})" 
            for idx, row in sorted_pd.iterrows()
        ]
    elif order_by == 'elevation':
        plot_labels = [
            f"{idx} ({int(row.elevation)}m)" 
            for idx, row in sorted_pd.iterrows()
        ]
    else:
        plot_labels = list(sorted_pd.index)
    
    # Prepare heatmap data (exclude metadata columns)
    metadata_cols = ['elevation', 'Id', 'HICLIMATEDIVISION']
    heatmap_cols = [c for c in sorted_pd.columns if c not in metadata_cols]
    
    # Order columns by metric type
    ordered_cols = []
    metrics = ['mean_residual', 'median_residual', 'rmse', 'std_dev_diff']
    conditions = ['all', 'clear', 'cloudy']
    
    for metric in metrics:
        for condition in conditions:
            col = f"{metric}_diff_emb_vs_base_{condition}"
            if col in heatmap_cols:
                ordered_cols.append(col)
    
    heatmap_data = sorted_pd[ordered_cols].rename(columns=COLUMN_LABELS)
    
    # Generate plot
    plt.figure(figsize=(13, 14))
    cmap = sns.diverging_palette(130, 10, as_cmap=True, s=80, l=50, sep=10)
    
    vmax = heatmap_data.abs().max().max()
    vmax = 1.0 if pd.isna(vmax) or vmax == 0 else vmax
    vmin = -vmax
    
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={'label': 'Residual Difference (K)', 'shrink': 0.9, 'pad': 0.03},
        annot=True,
        fmt=".2f",
        yticklabels=plot_labels
    )
    
    # Y-axis formatting
    order_label = {
        'station_id': 'Station',
        'elevation': 'Station\n(Elevation)',
        'division': 'Station\n(Division)'
    }
    ax.set_ylabel(order_label.get(order_by, 'Station'), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.xticks(rotation=90, ha='right', fontsize=9)
    
    # Color y-tick labels by division
    tick_colors = [DIVISION_COLORS.get(row.Id, "#000000") for _, row in sorted_pd.iterrows()]
    for tick_label, color in zip(ax.get_yticklabels(), tick_colors):
        tick_label.set_color(color)
        tick_label.set_fontweight('bold')
    
    # Add vertical dividers between metric groups
    metric_group_indices = [3, 6, 9]
    for idx in metric_group_indices:
        if idx < len(heatmap_data.columns):
            ax.axvline(idx, color='black', linestyle=':', linewidth=2)
    
    ax.set_title(
        "Comparative Model Performance by Station\n(BLAM) - (BLM)\nNegative = BLAM Better",
        fontsize=16, pad=20
    )
    ax.set_xlabel('Comparative Residual Metrics (Condition)', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"station_metrics_heatmap_by_{order_by}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative heatmaps for BLM vs BLAM model performance"
    )
    parser.add_argument(
        '--order_by', 
        type=str, 
        default='station_id',
        choices=['station_id', 'elevation', 'division'],
        help="Order stations by: station_id (default), elevation, or division"
    )
    parser.add_argument(
        '--descending', 
        action='store_true',
        help="Sort in descending order (default: ascending)"
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help="Don't display the plot (just save)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("STATION METRICS HEATMAP ANALYSIS")
    print("=" * 80)
    
    # Output directory
    output_dir = os.path.join(settings.BASE_DIR, 'figures', 'heatmaps')
    
    # Load station metadata
    print("\n--- Loading Station Metadata ---")
    stations_df = load_station_metadata()
    print(f"✓ Loaded {stations_df.height} stations")
    
    # Load predictions for both models
    print("\n--- Loading Model Predictions ---")
    
    # For now, we'll load from temp_predictions
    try:
        preds_df = load_predictions('BLM')
        print(f"✓ Loaded predictions for {preds_df['station_id'].n_unique()} stations")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        print("Make sure to run training first: python main.py --model_type BLM")
        return
    
    # Calculate metrics (placeholder - in real scenario you'd have both BLM and BLAM)
    print("\n--- Calculating Per-Station Metrics ---")
    blm_metrics = calculate_station_metrics(preds_df, stations_df)
    
    # For demonstration - using same predictions (replace with actual BLAM predictions)
    blam_metrics = blm_metrics.clone() # Clone in Polars
    
    # Calculate differences
    diff_df = calculate_difference_metrics(blm_metrics, blam_metrics)
    print(f"✓ Calculated difference metrics for {diff_df.height} stations")
    
    # Generate heatmap
    generate_heatmap(
        diff_df,
        order_by=args.order_by,
        ascending=not args.descending,
        output_dir=output_dir,
        show_plot=not args.no_show
    )
    
    print("\n" + "=" * 80)
    print("✅ Heatmap generation complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
