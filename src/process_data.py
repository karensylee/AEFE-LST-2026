import pandas as pd
import numpy as np
import os
import argparse
import time
import logging
from pvlib.solarposition import get_solarposition

# Project Imports (assuming running from root)
from config import settings

def setup_logging(log_path):
    """Setup logging to both file and console."""
    # Create logger
    logger = logging.getLogger('process_data')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def process_data(target_year=2024, tolerance_minutes=3):
    start_time = time.time()
    
    # Define Paths
    BASE_DIR = settings.BASE_DIR
    GROUND_DATA_PATH = os.path.join(BASE_DIR, 'datasets/raw/Tsrf_1_Avg')
    GOES_DATA_PATH = os.path.join(BASE_DIR, 'datasets/raw/goes18acmc/goes18_lst_samples_2024_ACMC.csv')
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'datasets/raw/aef/hawaii_station_alphaearth_embeddings_2024.csv')
    FINAL_OUTPUT_PATH = settings.DATA_PATH
    
    # Setup logging
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'process_data_{target_year}.log')
    logger = setup_logging(log_path)
    
    logger.info("=" * 60)
    logger.info("LST DATA PROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Target Year: {target_year}")
    logger.info(f"Tolerance: {tolerance_minutes} minutes")
    logger.info(f"Ground Data Dir: {GROUND_DATA_PATH}")
    logger.info(f"GOES Data Path: {GOES_DATA_PATH}")
    logger.info(f"Embeddings Path: {EMBEDDINGS_PATH}")
    logger.info(f"Output Path: {FINAL_OUTPUT_PATH}")
    logger.info(f"Log File: {log_path}")

    # --- 1. Load GOES Data ---
    logger.info("\n--- Step 1: Loading GOES Data ---")
    goes_df = pd.read_csv(GOES_DATA_PATH)
    logger.info(f"Loaded {len(goes_df)} total GOES rows.")
    
    # Rename time column
    goes_df.rename(columns={'ground_obs_time_utc': 'sample_time'}, inplace=True)
    
    # Clean station_id
    goes_df['station_id'] = goes_df['station_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(4)
    
    # Convert time
    logger.info("Converting GOES time...")
    goes_df['sample_time'] = pd.to_datetime(goes_df['sample_time'], utc=True, errors='coerce')
    goes_df.dropna(subset=['sample_time'], inplace=True)
    
    # Filter by year
    goes_df = goes_df[goes_df['sample_time'].dt.year == target_year].copy()
    logger.info(f"Filtered GOES data to {len(goes_df)} rows in {target_year}.")
    
    # Sort for merge_asof
    logger.info("Sorting GOES data...")
    goes_df.sort_values('sample_time', inplace=True)
    
    # Group by station for efficiency
    logger.info("Pre-grouping GOES data by station...")
    goes_groups = {sid: g.copy() for sid, g in goes_df.groupby('station_id')}
    logger.info(f"Created {len(goes_groups)} station groups.")
    logger.info("✅ Full GOES data prepared and sorted.")
    
    # --- 2. Process Ground Data ---
    logger.info("\n--- Step 2: Processing Ground Stations ---")
    all_merged_data = []
    station_stats = []  # Track per-station statistics
    
    ground_files = [f for f in os.listdir(GROUND_DATA_PATH) if f.endswith('_Tsrf_1_Avg.csv')]
    logger.info(f"Found {len(ground_files)} ground station files to process.")
    
    # Time window
    ground_start = pd.Timestamp(f'{target_year}-01-01 00:00:00', tz='utc')
    ground_end = pd.Timestamp(f'{target_year + 1}-01-01 00:03:00', tz='utc')
    logger.info(f"Ground data window: {ground_start} to {ground_end}")
    
    stations_processed = 0
    stations_merged = 0
    
    for f in ground_files:
        # Extract ID
        station_id = f.split('_')[0].zfill(4)
        stations_processed += 1
        
        # Check if we have GOES data
        goes_station_df = goes_groups.get(station_id)
        if goes_station_df is None or goes_station_df.empty:
            logger.info(f"Skipping Station: {station_id} (No GOES data found)")
            continue
            
        logger.info(f"\nProcessing Station: {station_id} ({stations_processed}/{len(ground_files)})...")
        
        # Load Ground
        g_path = os.path.join(GROUND_DATA_PATH, f)
        try:
            ground_df = pd.read_csv(g_path, usecols=['timestamp', 'value', 'flag', 'station_id'])
        except ValueError:
            ground_df = pd.read_csv(g_path)
            
        ground_df['station_id'] = ground_df['station_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(4)
        
        # Rename
        ground_df.rename(columns={'timestamp': 'date_time', 'value': 'LST'}, inplace=True)
        ground_df['date_time'] = pd.to_datetime(ground_df['date_time'], utc=True, errors='coerce')
        
        # Filter
        mask = (
            (ground_df['flag'] == 0) & 
            (ground_df['station_id'] == station_id) &
            (ground_df['date_time'] >= ground_start) &
            (ground_df['date_time'] <= ground_end)
        )
        ground_df_filtered = ground_df[mask].copy()
        
        # Dedup and sort
        ground_df_filtered.drop_duplicates(subset=['date_time'], inplace=True)
        ground_df_filtered.dropna(subset=['date_time', 'LST'], inplace=True)
        ground_df_filtered.sort_values('date_time', inplace=True)
        
        goes_rows = len(goes_station_df)
        ground_rows = len(ground_df_filtered)
        logger.info(f"  - Found {goes_rows} GOES rows and {ground_rows} valid ground rows in window.")
        
        if ground_df_filtered.empty:
            continue
            
        # Merge AsOf
        merged_df = pd.merge_asof(
            left=goes_station_df,
            right=ground_df_filtered[['date_time', 'LST', 'station_id']],
            left_on='sample_time',
            right_on='date_time',
            by='station_id',
            direction='nearest',
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
            allow_exact_matches=True
        )
        
        merged_df.dropna(subset=['date_time', 'LST'], inplace=True)
        merged_rows = len(merged_df)
        
        logger.info(f"  - Merged {merged_rows} rows for station {station_id}.")
        
        # Track stats
        station_stats.append({
            'station_id': station_id,
            'goes_rows': goes_rows,
            'ground_rows': ground_rows,
            'merged_rows': merged_rows
        })
        
        all_merged_data.append(merged_df)
        stations_merged += 1
        
    if not all_merged_data:
        logger.error("No data merged!")
        return

    final_df = pd.concat(all_merged_data, ignore_index=True)
    rows_before_dqf = len(final_df)
    logger.info(f"\nRow count before DQF filter: {rows_before_dqf}")
    
    # Filter ACMC_DQF
    final_df = final_df[final_df['ACMC_DQF'] == 0]
    rows_after_dqf = len(final_df)
    logger.info(f"Row count after DQF filter (ACMC_DQF == 0): {rows_after_dqf}")
    
    # Convert LST from Celsius to Kelvin
    logger.info("Converting LST from Celsius to Kelvin...")
    final_df['LST'] = pd.to_numeric(final_df['LST'], errors='coerce')
    final_df['LST'] = final_df['LST'] + 273.15
    
    # Filter unreasonable temperatures (240K to 373K)
    logger.info("Filtering unreasonable temperatures (240K - 373K)...")
    rows_before_temp_filter = len(final_df)
    final_df = final_df[final_df['LST'].between(240, 373)]
    rows_after_temp_filter = len(final_df)
    logger.info(f"Row count after temperature filter: {rows_after_temp_filter} (removed {rows_before_temp_filter - rows_after_temp_filter} rows)")
    
    # --- 3. Feature Engineering ---
    logger.info("\n--- Step 3: Feature Engineering ---")
    
    # Solar Position
    logger.info("Engineering solar position features (SZA, SAA)...")
    time_index = pd.DatetimeIndex(final_df['sample_time'])
    solar_pos = get_solarposition(time_index, latitude=final_df['lat'], longitude=final_df['lon'])
    final_df['SZA'] = solar_pos['zenith'].values
    final_df['SAA'] = solar_pos['azimuth'].values
    
    # CMI Calibration
    logger.info("Calibrating CMI bands...")
    scaling_params = {
        'CMI_C01': {'offset': 0, 'scale': 0.00031746001}, 'CMI_C02': {'offset': 0, 'scale': 0.00031746001},
        'CMI_C03': {'offset': 0, 'scale': 0.00031746001}, 'CMI_C04': {'offset': 0, 'scale': 0.00031746001},
        'CMI_C05': {'offset': 0, 'scale': 0.00031746001}, 'CMI_C06': {'offset': 0, 'scale': 0.00031746001},
        'CMI_C07': {'offset': 197.31, 'scale': 0.01309618}, 'CMI_C08': {'offset': 138.05, 'scale': 0.042249858},
        'CMI_C09': {'offset': 137.7, 'scale': 0.042339109}, 'CMI_C10': {'offset': 126.91, 'scale': 0.049889188},
        'CMI_C11': {'offset': 127.69, 'scale': 0.05216432}, 'CMI_C12': {'offset': 117.49, 'scale': 0.047270339},
        'CMI_C13': {'offset': 89.620003, 'scale': 0.06145332}, 'CMI_C14': {'offset': 96.190002, 'scale': 0.059850749},
        'CMI_C15': {'offset': 97.379997, 'scale': 0.05956082}, 'CMI_C16': {'offset': 92.699997, 'scale': 0.055081531},
    }
    for col, params in scaling_params.items():
        if col in final_df.columns:
            final_df[col] = (final_df[col] - params['offset']) * params['scale']
            
    # Cyclical Features
    logger.info("Creating DOY, HOUR, and cyclical features...")
    final_df['DOY'] = final_df['sample_time'].dt.dayofyear
    final_df['HOUR'] = final_df['sample_time'].dt.hour
    
    # 2024 is leap year -> 366
    final_df['DOY_sin'] = np.sin(2 * np.pi * final_df['DOY'] / 366.0)
    final_df['DOY_cos'] = np.cos(2 * np.pi * final_df['DOY'] / 366.0)
    final_df['HOUR_sin'] = np.sin(2 * np.pi * final_df['HOUR'] / 24.0)
    final_df['HOUR_cos'] = np.cos(2 * np.pi * final_df['HOUR'] / 24.0)
    
    final_df['SZA_sin'] = np.sin(np.deg2rad(final_df['SZA']))
    final_df['SZA_cos'] = np.cos(np.deg2rad(final_df['SZA']))
    final_df['SAA_sin'] = np.sin(np.deg2rad(final_df['SAA']))
    final_df['SAA_cos'] = np.cos(np.deg2rad(final_df['SAA']))
    
    # Local Time
    final_df['LOCAL_TIME'] = final_df['sample_time'] - pd.Timedelta(hours=10)
    
    # Rename columns to match config
    final_df.rename(columns={'station_id': 'SITE_ID'}, inplace=True)
    
    logger.info("✅ Feature engineering complete.")
    
    # --- 4. Merge Embeddings ---
    logger.info("\n--- Step 4: Merging Embeddings ---")
    emb_df = pd.read_csv(EMBEDDINGS_PATH)
    emb_df.rename(columns={'station_id': 'SITE_ID'}, inplace=True)
    
    # Ensure ID types match
    final_df['SITE_ID'] = final_df['SITE_ID'].astype(str)
    emb_df['SITE_ID'] = emb_df['SITE_ID'].astype(str)
    
    cols_to_merge = ['SITE_ID'] + [f'A{i:02d}' for i in range(64)] + ['elevation']
    available_emb_cols = [c for c in cols_to_merge if c in emb_df.columns]
    
    merged_final = pd.merge(final_df, emb_df[available_emb_cols], on='SITE_ID', how='left')
    
    if 'elevation' in merged_final.columns:
        merged_final.rename(columns={'elevation': 'Elevation'}, inplace=True)
    
    logger.info(f"Merged {len(available_emb_cols)-1} embedding features.")
        
    # --- 4.5 Merge Climate Divisions ---
    logger.info("\n--- Step 4.5: Merging Climate Divisions ---")
    CLIMATE_PATH = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef_hiclimatedivision.csv')
    if os.path.exists(CLIMATE_PATH):
        logger.info(f"Loading climate divisions from {CLIMATE_PATH}")
        cd_df = pd.read_csv(CLIMATE_PATH)
        cd_df.rename(columns={'station_id': 'SITE_ID'}, inplace=True)
        cd_df['SITE_ID'] = cd_df['SITE_ID'].astype(str)
        
        cols_to_use = ['SITE_ID'] + [c for c in settings.CLIMATE_DIVISIONS if c in cd_df.columns]
        
        if len(cols_to_use) > 1:
            merged_final = pd.merge(merged_final, cd_df[cols_to_use], on='SITE_ID', how='left')
            logger.info(f"Merged {len(cols_to_use)-1} climate division features.")
        else:
            logger.warning("No climate division columns found in file matching settings.")
    else:
        logger.warning(f"Climate division file not found at {CLIMATE_PATH}. Skipping.")
        
    # --- 5. Save ---
    logger.info("\n--- Step 5: Saving ---")
    
    os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)
    merged_final.to_csv(FINAL_OUTPUT_PATH, index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # --- Final Statistics ---
    logger.info("\n" + "=" * 60)
    logger.info("✅✅✅ PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    # Per-station summary
    logger.info("\n--- Per-Station Row Counts ---")
    stats_df = pd.DataFrame(station_stats)
    for _, row in stats_df.iterrows():
        logger.info(f"Station {row['station_id']}: GOES={row['goes_rows']}, Ground={row['ground_rows']}, Merged={row['merged_rows']}")
    
    # Post-processing per-station counts
    logger.info("\n--- Final Rows Per Station (After All Processing) ---")
    final_station_counts = merged_final.groupby('SITE_ID').size()
    for sid, count in final_station_counts.items():
        logger.info(f"Station {sid}: {count} rows")
    
    # Totals
    logger.info("\n--- Summary Statistics ---")
    logger.info(f"Total rows before DQF filter: {rows_before_dqf}")
    logger.info(f"Total rows after DQF filter: {rows_after_dqf}")
    logger.info(f"Total rows in final output: {len(merged_final)}")
    logger.info(f"Total stations processed: {stations_processed}")
    logger.info(f"Total stations with data merged: {stations_merged}")
    logger.info(f"Total columns in output: {len(merged_final.columns)}")
    
    # Feature counts
    logger.info("\n--- Feature Counts ---")
    logger.info(f"CMI Bands: {len(settings.CMI_BANDS)} ({', '.join(settings.CMI_BANDS[:3])}...)")
    logger.info(f"Embeddings: {len(settings.EMBEDDINGS)} (A00-A63)")
    logger.info(f"Auxiliary Features: {len(settings.AUXILIARY_FEATURES)}")
    for f in settings.AUXILIARY_FEATURES:
        logger.info(f"  - {f}")
    logger.info(f"Climate Divisions: {len(settings.CLIMATE_DIVISIONS)}")
    
    logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
    logger.info(f"💾 ML-ready data saved to: {FINAL_OUTPUT_PATH}")
    logger.info(f"📋 Log saved to: {log_path}")
    
    logger.info("\n--- Script Finished ---")

if __name__ == "__main__":
    process_data()
