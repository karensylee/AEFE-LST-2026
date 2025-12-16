# Land Surface Temperature (LST) Estimation from GOES-18

Machine learning pipeline for estimating Land Surface Temperature using GOES-18 satellite imagery and Hawaii Mesonet ground station data.

## Overview

This project trains XGBoost models to predict land surface temperature (Tsrf) by combining:
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
   ```
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
- Converts Tsrf from Celsius to Kelvin
- Filters unreasonable temperatures (240-373K)
- Calibrates CMI bands
- Computes solar position (SZA, SAA)
- Creates cyclical time features
- Merges AlphaEarth embeddings

**Output**: `datasets/processed/ML_READY_lst_2024.csv`

### Step 5: Train Models
```bash
python main.py --model_type BLM
python main.py --model_type BLAM --tune
```

**Available model types** (defined in `config/settings.py`):
| Model | Features |
|-------|----------|
| BLM | Baseline (CMI C13-C16 + auxiliary) |
| BLAM | Baseline + AlphaEarth embeddings |
| CIM | All CMI bands + auxiliary |
| CIAM | All CMI + embeddings |
| BLHIM | Baseline + climate divisions |

**Output**: `models/` (trained models + results CSVs)

### Step 6: Generate Analysis Plots
```bash
python analysis/density_plots.py
```
**Output**: `figures/densityplots/`

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
├── models/                   # Trained models (gitignored)
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

## References

- ATBD reference: [TBD]