"""
Moran's I Spatial Autocorrelation Analysis for LST Predictions

Computes Global and Local Moran's I statistics to assess spatial patterns
in model prediction residuals (errors).

Global Moran's I: Tests whether residuals are spatially clustered overall.
  - I > 0: Positive spatial autocorrelation (similar values cluster together)
  - I < 0: Negative spatial autocorrelation (dissimilar values cluster together)
  - I ≈ 0: Random spatial distribution

Local Moran's I (LISA): Identifies specific spatial clusters and outliers.
  - High-High (HH): High residual surrounded by high residuals
  - Low-Low (LL): Low residual surrounded by low residuals  
  - High-Low (HL): High residual surrounded by low residuals (spatial outlier)
  - Low-High (LH): Low residual surrounded by high residuals (spatial outlier)
"""

import os
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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
    'font.size': 12,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Coastline shapefile path
COASTLINE_PATH = os.path.join(settings.BASE_DIR, 'datasets/other/Coastline/Coastline.shp')

# Configuration
STATIONS_PATH = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef_hiclimatedivision.csv')
CONDITIONS = [('Clear-Sky', 0), ('Cloudy-Sky', 1), ('All-Sky', None)]
MODELS = ['B-E', 'B', 'B-E-X', 'B-X']


def load_stations():
    """Load station coordinates."""
    stations = pl.read_csv(STATIONS_PATH)
    stations = stations.select(['station_id', 'lat', 'lon', 'name', 'HICLIMATEDIVISION'])
    return stations


def load_predictions(model_name):
    """Load prediction data for a model."""
    pred_path = os.path.join(settings.OUTPUT_DIR, 'xgb', model_name, f'{model_name}_ALL_predictions.csv')
    if os.path.exists(pred_path):
        print(f"Loading {model_name}...")
        df = pl.read_csv(pred_path)
        return df
    else:
        print(f"Warning: {pred_path} not found.")
        return None


def calculate_station_residuals(df, metric='rmse'):
    """
    Calculate residual metrics per station.
    
    Args:
        df: Polars DataFrame with predictions
        metric: 'rmse', 'mae', 'bias', or 'std'
    
    Returns:
        Polars DataFrame with station_id and residual metric
    """
    df = df.with_columns(
        (pl.col('LST_pred') - pl.col('LST_true')).alias('error')
    )
    
    if metric == 'rmse':
        station_stats = df.group_by('station_id').agg([
            (pl.col('error') ** 2).mean().sqrt().alias('residual'),
            pl.len().alias('count')
        ])
    elif metric == 'mae':
        station_stats = df.group_by('station_id').agg([
            pl.col('error').abs().mean().alias('residual'),
            pl.len().alias('count')
        ])
    elif metric == 'bias':
        station_stats = df.group_by('station_id').agg([
            pl.col('error').mean().alias('residual'),
            pl.len().alias('count')
        ])
    elif metric == 'std':
        station_stats = df.group_by('station_id').agg([
            pl.col('error').std().alias('residual'),
            pl.len().alias('count')
        ])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return station_stats.sort('station_id')


def create_spatial_weights(coords, weight_type='inverse_distance', k=None, threshold=None):
    """
    Create spatial weights matrix.
    
    Args:
        coords: Nx2 array of (lat, lon) coordinates
        weight_type: 'inverse_distance', 'knn', or 'distance_band'
        k: Number of neighbors for KNN
        threshold: Distance threshold for distance_band
    
    Returns:
        NxN spatial weights matrix (row-standardized)
    """
    n = len(coords)
    
    # Calculate distance matrix (in km using Haversine approximation)
    # For Hawaii, simple Euclidean on lat/lon is reasonable
    distances = cdist(coords, coords, metric='euclidean')
    
    # Convert degrees to approximate km (at Hawaii's latitude ~20°N)
    # 1 degree lat ≈ 111 km, 1 degree lon ≈ 104 km at 20°N
    lat_scale = 111.0
    lon_scale = 104.0
    
    # Recalculate with scaling
    scaled_coords = np.column_stack([
        coords[:, 0] * lat_scale,
        coords[:, 1] * lon_scale
    ])
    distances = cdist(scaled_coords, scaled_coords, metric='euclidean')
    
    if weight_type == 'inverse_distance':
        # Inverse distance weights (with small epsilon to avoid infinity)
        with np.errstate(divide='ignore'):
            w = 1.0 / (distances + 1e-10)
        np.fill_diagonal(w, 0)  # No self-neighbors
        
    elif weight_type == 'knn':
        k = k or 4  # Default 4 nearest neighbors
        w = np.zeros((n, n))
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            neighbors = np.argsort(distances[i])[1:k+1]
            w[i, neighbors] = 1.0
        # Make symmetric
        w = (w + w.T) / 2
        
    elif weight_type == 'distance_band':
        threshold = threshold or 50.0  # Default 50 km threshold
        w = (distances <= threshold).astype(float)
        np.fill_diagonal(w, 0)
    
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # Row-standardize
    row_sums = w.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    w = w / row_sums[:, np.newaxis]
    
    return w


def global_morans_i(values, weights, n_permutations=999):
    """
    Calculate Global Moran's I statistic.
    
    Args:
        values: Array of attribute values
        weights: Row-standardized spatial weights matrix
        n_permutations: Number of permutations for pseudo p-value
    
    Returns:
        dict with I statistic, expected I, variance, z-score, and p-values
    """
    n = len(values)
    y = np.array(values)
    y_mean = y.mean()
    y_dev = y - y_mean
    
    # Calculate Moran's I
    numerator = np.sum(weights * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    
    if denominator == 0:
        return {'I': np.nan, 'E_I': np.nan, 'V_I': np.nan, 'z': np.nan, 
                'p_norm': np.nan, 'p_sim': np.nan, 'n': n}
    
    I = (n / np.sum(weights)) * (numerator / denominator)
    
    # Expected value under null hypothesis
    E_I = -1.0 / (n - 1)
    
    # Variance under normality assumption
    S0 = np.sum(weights)
    S1 = 0.5 * np.sum((weights + weights.T) ** 2)
    S2 = np.sum((weights.sum(axis=1) + weights.sum(axis=0)) ** 2)
    
    k = (np.sum(y_dev ** 4) / n) / ((np.sum(y_dev ** 2) / n) ** 2)
    
    A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
    B = k * ((n**2 - n) * S1 - 2*n * S2 + 6 * S0**2)
    C = (n - 1) * (n - 2) * (n - 3) * S0**2
    
    V_I = (A - B) / C - E_I**2
    
    if V_I > 0:
        z = (I - E_I) / np.sqrt(V_I)
        from scipy import stats
        p_norm = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z = np.nan
        p_norm = np.nan
    
    # Permutation test for pseudo p-value
    if n_permutations > 0:
        permuted_I = np.zeros(n_permutations)
        for perm in range(n_permutations):
            y_perm = np.random.permutation(y)
            y_perm_dev = y_perm - y_perm.mean()
            num = np.sum(weights * np.outer(y_perm_dev, y_perm_dev))
            denom = np.sum(y_perm_dev ** 2)
            if denom > 0:
                permuted_I[perm] = (n / S0) * (num / denom)
            else:
                permuted_I[perm] = 0
        
        # Two-tailed p-value
        p_sim = (np.sum(np.abs(permuted_I) >= np.abs(I)) + 1) / (n_permutations + 1)
    else:
        p_sim = np.nan
    
    return {
        'I': I,
        'E_I': E_I,
        'V_I': V_I,
        'z': z,
        'p_norm': p_norm,
        'p_sim': p_sim,
        'n': n
    }


def local_morans_i(values, weights):
    """
    Calculate Local Moran's I (LISA) statistics.
    
    Args:
        values: Array of attribute values
        weights: Row-standardized spatial weights matrix
    
    Returns:
        dict with local I values, z-scores, p-values, and cluster labels
    """
    n = len(values)
    y = np.array(values)
    y_mean = y.mean()
    y_dev = y - y_mean
    ss = np.sum(y_dev ** 2) / n
    
    if ss == 0:
        return None
    
    # Local Moran's I for each location
    local_i = np.zeros(n)
    for i in range(n):
        lag_i = np.sum(weights[i] * y_dev)
        local_i[i] = (y_dev[i] / ss) * lag_i
    
    # Expected value under null
    E_I = -1.0 / (n - 1)
    
    # Variance (simplified)
    m2 = np.sum(y_dev ** 2) / n
    wi = weights.sum(axis=1)
    wi2 = (weights ** 2).sum(axis=1)
    
    b2 = (np.sum(y_dev ** 4) / n) / (m2 ** 2)
    
    V_I = np.zeros(n)
    for i in range(n):
        A = (n - b2) / (n - 1)
        B = (2 * b2 - n) / ((n - 1) * (n - 2))
        V_I[i] = A * wi2[i] + B * (wi[i]**2 - wi2[i]) - E_I**2
    
    # Z-scores
    from scipy import stats
    z_scores = np.where(V_I > 0, (local_i - E_I) / np.sqrt(V_I), np.nan)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Cluster classification
    # Standardized values
    z_y = y_dev / np.std(y_dev)
    # Spatial lag of standardized values
    z_lag = np.dot(weights, z_y)
    
    clusters = np.full(n, 'NS', dtype=object)  # Not Significant
    sig_threshold = 0.05
    
    for i in range(n):
        if p_values[i] < sig_threshold:
            if z_y[i] > 0 and z_lag[i] > 0:
                clusters[i] = 'HH'  # High-High
            elif z_y[i] < 0 and z_lag[i] < 0:
                clusters[i] = 'LL'  # Low-Low
            elif z_y[i] > 0 and z_lag[i] < 0:
                clusters[i] = 'HL'  # High-Low outlier
            elif z_y[i] < 0 and z_lag[i] > 0:
                clusters[i] = 'LH'  # Low-High outlier
    
    return {
        'local_i': local_i,
        'z_scores': z_scores,
        'p_values': p_values,
        'clusters': clusters,
        'z_y': z_y,
        'z_lag': z_lag
    }


def plot_moran_scatter(values, weights, model_name, condition, output_dir):
    """
    Create Moran scatterplot.
    
    Args:
        values: Array of residual values
        weights: Spatial weights matrix
        model_name: Name of the model
        condition: Sky condition name
        output_dir: Directory to save plot
    """
    y = np.array(values)
    y_mean = y.mean()
    y_dev = y - y_mean
    y_std = np.std(y_dev)
    
    if y_std == 0:
        return
    
    z_y = y_dev / y_std
    z_lag = np.dot(weights, z_y)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(z_y, z_lag, alpha=0.7, edgecolor='k', linewidth=0.5)
    
    # Add regression line
    slope = np.polyfit(z_y, z_lag, 1)[0]
    x_line = np.linspace(z_y.min(), z_y.max(), 100)
    ax.plot(x_line, slope * x_line, 'r-', linewidth=2, label=f"Moran's I = {slope:.4f}")
    
    # Add reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # Quadrant labels
    ax.text(0.9, 0.9, 'HH', transform=ax.transAxes, fontsize=14, fontweight='bold', color='red')
    ax.text(0.1, 0.9, 'LH', transform=ax.transAxes, fontsize=14, fontweight='bold', color='blue')
    ax.text(0.1, 0.1, 'LL', transform=ax.transAxes, fontsize=14, fontweight='bold', color='red')
    ax.text(0.9, 0.1, 'HL', transform=ax.transAxes, fontsize=14, fontweight='bold', color='blue')
    
    ax.set_xlabel('Standardized Residual', fontsize=12)
    ax.set_ylabel('Spatial Lag of Standardized Residual', fontsize=12)
    ax.set_title(f"Moran's I Scatterplot\n{model_name} - {condition}", fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'moran_scatter_{model_name}_{condition.replace("-", "_")}.jpg')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_lisa_map(stations_df, lisa_result, model_name, condition, output_dir):
    """
    Create LISA cluster map with Hawaiian islands coastline.
    
    Args:
        stations_df: DataFrame with station coordinates
        lisa_result: Output from local_morans_i
        model_name: Name of the model
        condition: Sky condition name
        output_dir: Directory to save plot
    """
    if lisa_result is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Add Hawaii coastline outline
    try:
        import geopandas as gpd
        if os.path.exists(COASTLINE_PATH):
            coastline = gpd.read_file(COASTLINE_PATH)
            # Reproject to EPSG:4326 (WGS84) to match station coordinates
            if coastline.crs is not None and coastline.crs != 'EPSG:4326':
                coastline = coastline.to_crs('EPSG:4326')
            elif coastline.crs is None:
                coastline = coastline.set_crs('EPSG:4326')
            coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, zorder=1)
        else:
            print(f"  Warning: Coastline shapefile not found at {COASTLINE_PATH}")
    except ImportError:
        print("  Warning: geopandas not installed. Skipping coastline.")
    except Exception as e:
        print(f"  Warning: Could not load coastline: {e}")
    
    coords = stations_df[['lon', 'lat']].to_numpy()
    clusters = lisa_result['clusters']
    
    # Color mapping for clusters
    colors = {'HH': 'red', 'LL': 'blue', 'HL': 'orange', 'LH': 'lightblue', 'NS': 'lightgray'}
    labels = {'HH': 'High-High', 'LL': 'Low-Low', 'HL': 'High-Low', 'LH': 'Low-High', 'NS': 'Not Significant'}
    
    # Plot each cluster type
    for cluster_type in ['NS', 'LH', 'HL', 'LL', 'HH']:  # Plot NS first so it's in background
        mask = clusters == cluster_type
        if np.any(mask):
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[cluster_type],
                s=100,
                label=f"{labels[cluster_type]} ({np.sum(mask)})",
                edgecolor='k',
                linewidth=0.5,
                alpha=0.8,
                zorder=10
            )
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f"LISA Cluster Map\n{model_name} - {condition}", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    
    # Set Hawaii bounds
    ax.set_xlim(-160.5, -154.5)
    ax.set_ylim(18.8, 22.5)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'lisa_map_{model_name}_{condition.replace("-", "_")}.jpg')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def run_analysis(metric='rmse', weight_type='inverse_distance', n_permutations=999):
    """
    Run full Moran's I analysis for all models and conditions.
    
    Args:
        metric: Residual metric to analyze ('rmse', 'mae', 'bias', 'std')
        weight_type: Type of spatial weights ('inverse_distance', 'knn', 'distance_band')
        n_permutations: Number of permutations for p-value estimation
    
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*60}")
    print(f"Moran's I Spatial Autocorrelation Analysis")
    print(f"Metric: {metric.upper()}, Weights: {weight_type}")
    print(f"{'='*60}")
    
    # Load station data
    stations = load_stations()
    print(f"Loaded {stations.height} stations")
    
    # Create output directory
    output_dir = os.path.join(settings.FIGURES_DIR, 'morans_i')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for model_name in MODELS:
        df = load_predictions(model_name)
        if df is None:
            continue
        
        for cond_name, cond_val in CONDITIONS:
            print(f"\n{model_name} - {cond_name}")
            
            # Filter by condition
            if cond_val is not None:
                df_cond = df.filter(pl.col('ACMC_BCM') == cond_val)
            else:
                df_cond = df  # All-Sky
            
            # Calculate station residuals
            station_residuals = calculate_station_residuals(df_cond, metric=metric)
            
            # Join with station coordinates
            merged = stations.join(station_residuals, on='station_id', how='inner')
            
            if merged.height < 5:
                print(f"  Insufficient stations (n={merged.height}). Skipping.")
                continue
            
            print(f"  Stations with data: {merged.height}")
            
            # Extract coordinates and values
            coords = merged.select(['lat', 'lon']).to_numpy()
            values = merged['residual'].to_numpy()
            
            # Create spatial weights
            weights = create_spatial_weights(coords, weight_type=weight_type)
            
            # Global Moran's I
            global_result = global_morans_i(values, weights, n_permutations=n_permutations)
            
            print(f"  Global Moran's I: {global_result['I']:.4f}")
            print(f"  Z-score: {global_result['z']:.4f}")
            print(f"  P-value (norm): {global_result['p_norm']:.4f}")
            print(f"  P-value (sim): {global_result['p_sim']:.4f}")
            
            # Interpret result
            if global_result['p_sim'] < 0.05:
                if global_result['I'] > 0:
                    interpretation = "Significant positive spatial autocorrelation (clustering)"
                else:
                    interpretation = "Significant negative spatial autocorrelation (dispersion)"
            else:
                interpretation = "No significant spatial autocorrelation (random)"
            print(f"  Interpretation: {interpretation}")
            
            # Store results
            results.append({
                'Model': model_name,
                'Condition': cond_name,
                'Metric': metric.upper(),
                'N_Stations': global_result['n'],
                'Morans_I': global_result['I'],
                'E_I': global_result['E_I'],
                'V_I': global_result['V_I'],
                'Z_Score': global_result['z'],
                'P_Norm': global_result['p_norm'],
                'P_Sim': global_result['p_sim'],
                'Interpretation': interpretation
            })
            
            # Local Moran's I
            lisa_result = local_morans_i(values, weights)
            
            if lisa_result is not None:
                # Count clusters
                clusters, counts = np.unique(lisa_result['clusters'], return_counts=True)
                cluster_summary = dict(zip(clusters, counts))
                print(f"  LISA Clusters: {cluster_summary}")
                
                # Save station-level LISA results
                lisa_df = merged.clone()
                lisa_df = lisa_df.with_columns([
                    pl.Series('local_i', lisa_result['local_i']),
                    pl.Series('z_score', lisa_result['z_scores']),
                    pl.Series('p_value', lisa_result['p_values']),
                    pl.Series('cluster', lisa_result['clusters'])
                ])
                lisa_path = os.path.join(output_dir, f'lisa_{model_name}_{cond_name.replace("-", "_")}.csv')
                lisa_df.write_csv(lisa_path)
            
            # Generate plots
            # Convert coords and merged to pandas for plotting
            merged_pd = merged.to_pandas()
            plot_moran_scatter(values, weights, model_name, cond_name, output_dir)
            plot_lisa_map(merged_pd, lisa_result, model_name, cond_name, output_dir)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(output_dir, 'morans_i_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Figures saved to: {output_dir}")
    
    return results_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Moran's I Spatial Autocorrelation Analysis")
    parser.add_argument('--metric', type=str, default='rmse',
                        choices=['rmse', 'mae', 'bias', 'std'],
                        help='Residual metric to analyze')
    parser.add_argument('--weights', type=str, default='inverse_distance',
                        choices=['inverse_distance', 'knn', 'distance_band'],
                        help='Type of spatial weights')
    parser.add_argument('--permutations', type=int, default=999,
                        help='Number of permutations for pseudo p-value')
    
    args = parser.parse_args()
    
    results = run_analysis(
        metric=args.metric,
        weight_type=args.weights,
        n_permutations=args.permutations
    )
    
    # Display summary table
    if not results.empty:
        print("\n" + "="*80)
        print("MORAN'S I ANALYSIS SUMMARY")
        print("="*80)
        
        # Format for display
        disp_df = results.copy()
        disp_df['Morans_I'] = disp_df['Morans_I'].apply(lambda x: f"{x:.4f}")
        disp_df['Z_Score'] = disp_df['Z_Score'].apply(lambda x: f"{x:.4f}")
        disp_df['P_Sim'] = disp_df['P_Sim'].apply(lambda x: f"{x:.4f}" if x >= 0.001 else f"{x:.2e}")
        
        display_cols = ['Model', 'Condition', 'N_Stations', 'Morans_I', 'Z_Score', 'P_Sim', 'Interpretation']
        print(disp_df[display_cols].to_markdown(index=False))


if __name__ == "__main__":
    main()
