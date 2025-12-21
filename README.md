# Land Surface Temperature (LST) Estimation from GOES-18

Machine learning pipeline for estimating Land Surface Temperature using GOES-18 satellite imagery and Hawaii Mesonet ground station data.

## Overview

This project trains XGBoost models to predict land surface temperature (LST) by combining:

- **Ground truth**: Hawaii Mesonet station temperature measurements (Tsrf_1_Avg)
- **Satellite features**: GOES-18 MCMIPC Cloud Moisture Imagery bands (CMI_C01-C16)
- **Embeddings**: AlphaEarth geospatial embeddings (64-dim)
- **Auxiliary**: Elevation, cyclical time and solar position features (two components per feature: DOY, hour of day, SZA, SAA)

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
python main.py --model_type BLM
python main.py --model_type BLAM
python main.py --model_type BLAM-C
python main.py --model_type CIM
python main.py --model_type CIAM
```

Optuna hyperparameter tuning (100 trials) runs automatically for each model.

**Available model types** (defined in `config/settings.py`):

| Model | Description | Components |
|-------|-------------|------------|
| **BLM** | Baseline Model | CMI Bands + Auxiliary |
| **BLAM** | Baseline + AlphaEarth | CMI Bands + Auxiliary + Embeddings |
| **BLAM-C** | Context Only | Auxiliary + Embeddings (No CMI) |
| **CIM** | Satellite Only | CMI Bands (No Auxiliary, No Embeddings) |
| **CIAM** | Satellite + AlphaEarth | CMI Bands + Embeddings (No Auxiliary) |

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

### Step 6: Generate Analysis Plots

```bash
python analysis/density_plots.py
```

**Output**: `figures/densityplots/`

### Step 6: Analysis & Aggregation

The project includes several scripts for aggregating data and analyzing model performance:

| Script | Description |
|--------|-------------|
| **`src/aggregate_5min_to_hourly_daily_monthly.py`** | **Data Aggregation**: Converts 5-minute samples to Hourly, Daily, and Monthly means. Applies QC thresholds (e.g., ≥10 obs/hr, ≥22 hrs/day) based on Lucas et al. (2020). |
| **`analysis/density_plots.py`** | **Validation**: Generates density scatter plots comparing True vs Predicted LST. Stratifies results by Clear-Sky, Cloudy-Sky, and All-Sky conditions. |
| **`analysis/monthly_analysis.py`** | **Performance Constraints**: Computes publication-ready monthly metrics (RMSE, R², Bias) and visualizes them alongside sample counts and cloudiness proportions. |
| **`analysis/heatmap_plots.py`** | **Spatial Analysis**: Generates heatmaps of error metrics across all stations, ordered by station ID, elevation, or climate division. Useful for identifying spatial bias. |
| **`analysis/gantt_chart.py`** | **Data Availability**: Visualizes temporal data availability for 2024, highlighting gaps and active periods for each station. |

```bash
# Example: Generate density plots
python analysis/density_plots.py

# Example: Generate monthly analysis figures
python analysis/monthly_analysis.py

# Example: Generate availability chart
python analysis/gantt_chart.py
```

**Output**: All analysis figures are saved to `figures/`.

## Project Structure

```
lst/
├── config/
│   └── settings.py          # Feature definitions, paths, model params
├── src/
│   ├── process_data.py      # Data merging and feature engineering
│   ├── data_loader.py       # Data loading utilities
│   ├── trainer.py           # XGBoost training logic
│   └── evaluation.py        # Metrics calculation
├── analysis/
│   └── density_plots.py     # Visualization
├── datasets/
│   ├── raw/                  # Downloaded data (gitignored)
│   ├── processed/            # ML-ready CSVs (gitignored)
│   └── stations/             # Station metadata
├── models/
│   └── xgb/                  # XGBoost models by type
│       ├── BLM/
│       ├── BLAM/
│       ├── BLAM-C/
│       ├── CIM/
│       └── CIAM/
├── figures/                  # Generated plots
├── download.py               # Ground data download
├── goes18mcmipc.py           # GOES MCMIPC download
├── goes18acmc.py             # GOES ACMC cloud mask
├── main.py                   # Training pipeline
└── requirements.txt
```

## Acknowledgments

- **Brian Blaylock** for [goes2go](https://github.com/blaylockbk/goes2go)
- **Hawaii Climate Data Portal** (HCDP) for Mesonet API access
- **Google Earth Engine** for satellite data access
- **Tom Giambelluca & Han Tseng** for providing Mesonet Instrument Details
- **Hawaii State Climate Office** for providing station metadata

## References

- See manuscript
