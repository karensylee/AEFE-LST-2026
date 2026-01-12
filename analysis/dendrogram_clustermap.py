# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

print(f"NumPy version: {np.__version__}")
print(f"Polars version: {pl.__version__}")
print("--- Libraries Imported Successfully ---")

# ==============================================================================
# CONFIGURE PATHS
# ==============================================================================
RANDOM_STATE = 808
np.random.seed(RANDOM_STATE)

# Adjusted paths for local environment
DATA_PATH = '/home/ksylee/projects/lst/datasets/processed/ML_READY_mesonet_goes_embeddings_2024.csv'
# Using the standard model directory for analysis outputs if it exists, otherwise figures
base_model_dir = '/home/ksylee/projects/lst/models/xgb/B'
if os.path.exists(base_model_dir):
    OUTPUT_DIR = os.path.join(base_model_dir, 'shap_analysis')
else:
    OUTPUT_DIR = '/home/ksylee/projects/lst/figures/dendrogram_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output Directory: {OUTPUT_DIR}")

# ==============================================================================
# DEFINE FEATURE SET
# ==============================================================================
TARGET_VARIABLE = 'LST'
CMI_FEATURES = [f'CMI_C{i:02d}' for i in range(1, 17)]
AEFE_FEATURES = [f'A{i:02d}' for i in range(64)]
AUXILIARY_FEATURES = [
    'Elevation', 'HOUR_sin', 'HOUR_cos', 'DOY_sin', 'DOY_cos',
    'SZA_sin', 'SZA_cos', 'SAA_sin', 'SAA_cos'
]

FEATURE_SET = CMI_FEATURES + AEFE_FEATURES + AUXILIARY_FEATURES

# ==============================================================================
# LOAD DATA (POLARS OPTIMIZED)
# ==============================================================================
print("\n--- Loading Data with Polars ---")

cols_to_load = FEATURE_SET + ['SITE_ID', 'ACMC_BCM', 'LOCAL_TIME', TARGET_VARIABLE]

try:
    # Polars scan for lazy loading and efficient memory management
    data_df = pl.scan_csv(DATA_PATH).select(cols_to_load).with_columns([
        pl.col(FEATURE_SET).cast(pl.Float32),
        pl.col(TARGET_VARIABLE).cast(pl.Float32),
        pl.col('SITE_ID').cast(pl.String),
        pl.col('ACMC_BCM').cast(pl.Float32)
    ]).collect()

    print(f"Loaded data: {data_df.height:,} rows × {data_df.width} columns")
    print(f"Memory usage: {data_df.estimated_size('mb'):.1f} MB")

    station_ids = sorted(data_df['SITE_ID'].unique().to_list())
    print(f"Unique stations: {len(station_ids)}")

except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# ==============================================================================
# CORRELATION ANALYSIS
# ==============================================================================
print("\n--- Feature Correlation Analysis ---")

# Efficient sampling in Polars
sample_size = min(100000, data_df.height)
X_sample_corr = data_df.select(FEATURE_SET).sample(n=sample_size, seed=RANDOM_STATE).to_numpy()

# NumPy is used for corrcoef as it is highly optimized for 2D arrays
corr_matrix = np.corrcoef(X_sample_corr.T)

# Convert to Polars for CSV output/filtering
corr_pl = pl.from_numpy(corr_matrix, schema=FEATURE_SET).with_columns(
    Feature=pl.Series(FEATURE_SET)
)
corr_pl.write_csv(os.path.join(OUTPUT_DIR, 'feature_correlation_matrix.csv'))
print(f"Correlation matrix saved to {os.path.join(OUTPUT_DIR, 'feature_correlation_matrix.csv')}")

# Find high correlation pairs using Polars expressions
print("\nSearching for highly correlated pairs...")
# Unpivot the matrix to a long format for easier filtering
high_corr_df = (
    corr_pl.unpivot(index="Feature", variable_name="Feature_2", value_name="Correlation")
    .filter(
        (pl.col("Correlation").abs() > 0.8) &
        (pl.col("Feature") < pl.col("Feature_2")) # Avoid self-corr and duplicates
    )
    .sort(pl.col("Correlation").abs(), descending=True)
)

print(f"Highly correlated pairs (|r| > 0.8): {high_corr_df.height}")
if high_corr_df.height > 0:
    print(high_corr_df.head(15))

# ==============================================================================
# CLUSTERING & DENDROGRAM CLUSTERMAP (SEABORN)
# ==============================================================================
print("\n--- Generating Dendrogram Clustermap ---")

# Create correlation DataFrame for seaborn
corr_df = pl.DataFrame(corr_matrix, schema=FEATURE_SET).to_pandas()
corr_df.index = FEATURE_SET

# Use seaborn clustermap with adjusted spacing
try:
    g = sns.clustermap(
        corr_df,
        method='complete',
        metric='correlation',
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        figsize=(28, 28),
        dendrogram_ratio=(0.08, 0.08),  # Smaller dendrograms
        cbar_pos=(1.02, 0.3, 0.025, 0.4),  # Move colorbar to right side, outside
        tree_kws={'linewidths': 0.8},
        xticklabels=True,
        yticklabels=True
    )

    # Adjust tick label sizes
    g.ax_heatmap.tick_params(axis='x', labelsize=8, rotation=90, pad=2)
    g.ax_heatmap.tick_params(axis='y', labelsize=8, pad=2)

    # Move x-axis labels to BOTTOM (away from top dendrogram)
    g.ax_heatmap.xaxis.set_ticks_position('bottom')
    g.ax_heatmap.xaxis.set_label_position('bottom')
    g.ax_heatmap.tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True)

    # Y-axis labels on RIGHT (closer to heatmap, away from left dendrogram)
    g.ax_heatmap.yaxis.set_ticks_position('right')
    g.ax_heatmap.yaxis.set_label_position('right')
    g.ax_heatmap.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)

    # Title
    g.fig.suptitle('Feature Correlation Clustermap', fontsize=14, y=1.01)

    # Colorbar label
    g.cax.set_ylabel('Correlation', fontsize=10)

    output_path = os.path.join(OUTPUT_DIR, 'correlation_clustermap.jpg')
    plt.savefig(output_path, dpi=150, format='jpg', pil_kwargs={'quality': 90}, bbox_inches='tight')
    print(f"Clustermap saved to {output_path}")
    
    # Store the ordering
    order = g.dendrogram_row.reordered_ind
    features_ordered = [FEATURE_SET[i] for i in order]
    
    print("Clustermap generation complete.")

except Exception as e:
    print(f"Error producing clustermap: {e}")

gc.collect()
