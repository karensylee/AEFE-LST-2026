# Land Surface Temperature (LST) Estimation from GOES-18

Machine learning pipeline for estimating Land Surface Temperature using GOES-18 satellite imagery and Hawaii Mesonet ground station data.

## Overview

This project trains XGBoost models to predict land surface temperature (LST) by combining:

- **Ground truth**: Hawaii Mesonet station temperature measurements (Tsrf_1_Avg)
- **Satellite features**: GOES-18 MCMIPC Cloud Moisture Imagery bands (CMI_C01-C16)
- **Embeddings**: AlphaEarth geospatial embeddings (A00-A63)
- **Auxiliary**: Elevation and cyclical time: Day of the Year (DOY_sin, DOY_cos), Hour of the day (HOUR_sin, HOUR_cos), Solar Zenith Angle (SZA_sin, SZA_cos), and Solar Azimuth Angle (SAA_sin, SAA_cos)

## Requirements

```bash
pip install -r requirements.txt
```

**External dependencies**:

- Google Earth Engine account (for GOES data download)
- HCDP API key (for Hawaii Mesonet data)

## Setup

1. **Clone and configure environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Configure `.env`**:

   ```env
   HCDP_API_KEY=your_hcdp_api_key
   EEPROJECTNAME=your_gee_project
   BASE_PATH=/path/to/lst
   STATIONS_CSV=datasets/stations/stations.csv
   ```

## Workflow

### Step 1: Download Ground Station Data

```bash
python download.py
```

Downloads 5-minute averaged surface temperature (Tsrf_1_Avg) from Hawaii Mesonet stations.

**Output**: `datasets/raw/Tsrf_1_Avg/{station_id}_Tsrf_1_Avg.csv`

### Step 2: Download GOES-18 Satellite Data

```bash
python goes18mcmipc.py
```

Samples GOES-18 MCMIPC imagery at ground station locations and timestamps via Google Earth Engine.

**Output**: Exports to Google Drive → move to `datasets/raw/goes18cmipc/goes18_lst_samples_2024.csv`

### Step 3: Add Cloud Mask (ACMC)

```bash
python goes18acmc.py
```

Attaches GOES-18 ACMC cloud mask data (BCM, ACM, DQF) to the satellite samples.

**Output**: `datasets/raw/goes18acmc/goes18_lst_samples_2024_ACMC.csv`

### Step 4: Process and Merge Data

```bash
python -m src.process_data
```

Merges ground + satellite data, performs feature engineering:

- Converts LST from Celsius to Kelvin
- Filters unreasonable temperatures (240-373K)
- Calibrates CMI bands
- Computes solar position (SZA, SAA)
- Creates cyclical time features
- Merges AlphaEarth embeddings

**Output**: `datasets/processed/ML_READY_mesonet_goes_embeddings_2024.csv`

### Step 5: Train Models

```bash
python main.py --model_type SX
python main.py --model_type SXE
python main.py --model_type SXE-C
python main.py --model_type S
python main.py --model_type SE
```

```bash
python main.py --model_type SX && \
python main.py --model_type SXE && \
python main.py --model_type SXE-C && \
python main.py --model_type S && \
python main.py --model_type SE
```

Optuna hyperparameter tuning (100 trials) runs automatically for each model.

**Available model types** (defined in `config/settings.py`):

| Model | Description | Components |
|-------|-------------|------------|
| **SX** | Baseline Model | CMI Bands + Auxiliary |
| **SXE** | Baseline + AlphaEarth | CMI Bands + Auxiliary + Embeddings |
| **SXE-C** | Context Only | Auxiliary + Embeddings (No CMI) |
| **S** | Satellite Only | CMI Bands (No Auxiliary, No Embeddings) |
| **SE** | Satellite + AlphaEarth | CMI Bands + Embeddings (No Auxiliary) |

### Feature Details

Specific features included in the components above:

- **CMI Bands** (16 features): Channels 1-16 from GOES-18 (`CMI_C01` ... `CMI_C16`)
- **Auxiliary** (9 features):
  - **Elevation**: Station elevation (meters)
  - **Solar Position**: Zenith (`SZA_sin`, `SZA_cos`) and Azimuth (`SAA_sin`, `SAA_cos`)
  - **Cyclical Time**: Hour of day (`HOUR_sin`, `HOUR_cos`) and Day of year (`DOY_sin`, `DOY_cos`)
- **Embeddings** (64 features): AlphaEarth geospatial embeddings (`A00` ... `A63`)

### Training Pipeline Features

The training pipeline (`main.py`) includes:

- **Leave-One-Station-Out (LOSO)** cross-validation across all stations
- **Checkpointing**: Skips stations if predictions already exist (resume interrupted runs)
- **Per-fold outputs**: Model and scaler saved for each holdout station
- **Metrics**: R², RMSE, MAE, MRE, and **Bias** for Overall, Clear-sky, and Cloudy-sky conditions
- **Aggregated predictions**: Combined predictions file for analysis scripts

### Output Files

After training, outputs are saved to `models/xgb/{model_type}/`:

```text
models/xgb/{MODEL}/
├── {MODEL}_ALL_predictions.csv     # Aggregated predictions from all folds
├── {MODEL}_loso_results.csv        # Per-station metrics summary
├── {MODEL}_model_{station}.joblib  # Trained model for each fold
├── {MODEL}_scaler_{station}.joblib # Scaler for each fold
└── loso_temp_predictions/          # Per-station prediction files
    ├── preds_0115.csv
    ├── preds_0116.csv
    └── ...
```

---

## Step 6: Analysis & Visualization

The project includes comprehensive analysis scripts for evaluating model performance, generating publication-ready figures, and performing statistical tests.

### Analysis Scripts Overview

| Script | Description | Output Directory |
|--------|-------------|------------------|
| `analysis/density_plots.py` | Density scatter plots (True vs Predicted LST) | `figures/density_plots/` |
| `analysis/heatmap_plots.py` | Station-level error metric heatmaps | `figures/heatmaps/` |
| `analysis/gantt_chart.py` | Temporal data availability visualization | `figures/analysis/` |
| `analysis/hourly_daily_rmse.py` | Hourly and daily RMSE temporal analysis | `figures/rmse_temporal/` |
| `analysis/paired_dot_plots.py` | Paired dot (slope) plots for model comparisons | `figures/statistical_tests/` |
| `analysis/statistical_analysis.py` | Core statistical functions and metrics export | `figures/statistical_tests/` |

---

### Density Plots

```bash
python analysis/density_plots.py
```

Generates density scatter plots comparing True vs Predicted LST:

- **Individual plots**: Clear-Sky, Cloudy-Sky, and All-Sky per model
- **2x4 Compact Grid**: Rows (Clear-Sky, Cloudy-Sky) × Columns (SX, SXE, S, SE)
- **2x2 All-Sky Grid**: All-Sky comparison across all models

**Output**: `figures/density_plots/`

---

### Heatmap Plots

```bash
python analysis/heatmap_plots.py
```

Generates heatmaps comparing model performance across stations:

- **Metrics**: RMSE and STD differences (SXE - SX, SE - S)
- **Conditions**: Clear-Sky, Cloudy-Sky, All-Sky
- **Ordering options**: Station ID, Elevation, or Climate Division

**Output**: `figures/heatmaps/`

---

### Gantt Chart (Data Availability)

```bash
python analysis/gantt_chart.py
```

Visualizes temporal data availability for all stations throughout 2024:

- Identifies continuous data blocks and gaps
- Shows availability intervals per station
- Useful for understanding data coverage

**Output**: `figures/analysis/station_availability.jpg`

---

### Hourly & Daily RMSE Analysis

```bash
python analysis/hourly_daily_rmse.py
```

Generates temporal RMSE comparisons between SXE and SX:

- **Hourly Box Plots**: Paired box plots comparing SXE and SX RMSE by local hour (HST)
  - Includes secondary axis showing cloud percentage
  - Observation count tables below plots
- **Daily Line Plots**: Daily RMSE throughout 2024 with 7-day rolling mean
  - Cloud percentage overlay on secondary axis
  - Monthly observation count tables

**Output**: `figures/rmse_temporal/`

---

### Paired Dot Plots (Statistical Visualizations)

```bash
python analysis/paired_dot_plots.py
```

Generates paired dot (slope) plots for station-level model comparisons:

- **2x4 Compact Panel**:
  - Rows: Clear-Sky, Cloudy-Sky
  - Columns: SX vs SXE (RMSE), SX vs SXE (STD), S vs SE (RMSE), S vs SE (STD)
- **Individual metric plots**: Separate 2x2 grids for RMSE and STD
- **Features**:
  - Wilcoxon signed-rank test p-values in titles
  - Improvement vs No Improvement counts
  - Mean markers (diamonds)

**Output**: `figures/statistical_tests/`

---

### Statistical Analysis

```bash
python analysis/statistical_analysis.py
```

Core statistical module for computing and exporting model metrics:

- **Metrics computed**: RMSE, R², STD, Median Bias
- **Conditions**: Clear-Sky, Cloudy-Sky, All-Sky
- **Statistical tests**: Wilcoxon signed-rank test, Cohen's q effect size
- **Output**: `model_metrics_summary.csv`

**Output**: `figures/statistical_tests/model_metrics_summary.csv`

---
<!-- currently deprecated -->
<!-- ## Data Aggregation

The `src/` directory includes temporal aggregation utilities: -->

<!-- ### 5-Minute to Hourly/Daily/Monthly Aggregation

```bash
python -m src.aggregate_5min_to_hourly_daily_monthly
```

Converts 5-minute observations to aggregated time scales with quality control thresholds based on Lucas et al. (2020):

| Aggregation | QC Threshold |
|-------------|--------------|
| **Hourly** | ≥10 of 12 possible 5-min observations |
| **Daily** | ≥22 of 24 possible hourly observations |
| **Monthly** | 100% daily completeness required | -->

---

## Project Structure

```
lst/
├── config/
│   └── settings.py              # Feature definitions, paths, model params
├── src/
│   ├── process_data.py          # Data merging and feature engineering
│   ├── data_loader.py           # Data loading utilities
│   ├── trainer.py               # XGBoost training logic
│   ├── evaluation.py            # Metrics calculation
│   ├── aggregation.py           # Temporal aggregation (5min → hourly/daily/monthly)
│   ├── aggregate_5min_to_hourly_daily_monthly.py  # Aggregation runner script
│   ├── process_climate.py       # Climate division processing
│   └── station_processing.py    # Station data utilities
├── analysis/
│   ├── density_plots.py         # Density scatter plots
│   ├── gantt_chart.py           # Data availability visualization
│   ├── heatmap_plots.py         # Station metric heatmaps
│   ├── hourly_daily_rmse.py     # Temporal RMSE analysis
│   ├── paired_dot_plots.py      # Statistical visualization
│   └── statistical_analysis.py  # Core statistical functions
├── notebooks/
│   └── example_gantt_chart.ipynb  # Example Jupyter notebook
├── datasets/
│   ├── raw/                     # Downloaded data (gitignored)
│   ├── processed/               # ML-ready CSVs (gitignored)
│   └── stations/                # Station metadata
├── models/
│   └── xgb/                     # XGBoost models by type
│       ├── SX/
│       ├── SXE/
│       ├── SXE-C/
│       ├── S/
│       └── SE/
├── figures/                     # Generated analysis figures
│   ├── analysis/                # General analysis plots
│   ├── density_plots/           # Density scatter plots
│   ├── heatmaps/                # Station heatmaps
│   ├── rmse_temporal/           # Hourly/daily RMSE plots
│   └── statistical_tests/       # Statistical visualizations & CSVs
├── download.py                  # Ground data download
├── goes18mcmipc.py              # GOES MCMIPC download
├── goes18acmc.py                # GOES ACMC cloud mask
├── main.py                      # Training pipeline
└── requirements.txt
```

## Quick Start

Run all analysis scripts after training:

```bash
# Generate all visualizations
python analysis/density_plots.py
python analysis/heatmap_plots.py
python analysis/gantt_chart.py
python analysis/hourly_daily_rmse.py
python analysis/paired_dot_plots.py
python analysis/statistical_analysis.py
```

Or run them sequentially:

```bash
python analysis/density_plots.py && \
python analysis/heatmap_plots.py && \
python analysis/gantt_chart.py && \
python analysis/hourly_daily_rmse.py && \
python analysis/paired_dot_plots.py && \
python analysis/statistical_analysis.py
```

## Acknowledgments

- **Brian Blaylock** for [goes2go](https://github.com/blaylockbk/goes2go)
- **Hawaii Climate Data Portal** (HCDP) for Mesonet API access and station metadata
- **Google Earth Engine** for MCMIPC and AEFE cloud computation and access
- **Tom Giambelluca & Han Tseng** for providing Mesonet Instrument Details

## References

- See manuscript
