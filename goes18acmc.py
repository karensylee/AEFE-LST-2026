import os
import pandas as pd
import xarray as xr
import fsspec
import h5netcdf
import numpy as np
import warnings
from google.cloud import storage
from goes2go import tools
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Suppress pandas concatenation warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. SETUP ---
load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
BUCKET_NAME = 'gcp-public-data-goes-18'
CHECKPOINT_INTERVAL = 500

STATIONS_PATH = os.path.join(BASE_PATH, os.getenv("STATIONS_CSV"))
CSV_IN_PATH = os.path.join(BASE_PATH, 'datasets/raw/goes18cmipc/goes18_lst_samples_2024.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'datasets/raw/goes18acmc')
CSV_OUT_PATH = os.path.join(OUTPUT_DIR, 'goes18_lst_samples_2024_ACMC.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize GCS
storage_client = storage.Client.create_anonymous_client()
bucket = storage_client.bucket(BUCKET_NAME)
url_cache = {}

print("Loading Data...")
# Load IDs as strings to match your successful diagnostic
df_in = pd.read_csv(CSV_IN_PATH, dtype={'GOES_img_id': str, 'station_id': str})
df_stations = pd.read_csv(STATIONS_PATH, dtype={'station_id': str})

# Normalize IDs
df_in['station_id'] = df_in['station_id'].str.zfill(4)
df_stations['station_id'] = df_stations['station_id'].str.zfill(4)

# Handle lat/lon column names
if 'lng' in df_stations.columns:
    df_stations.rename(columns={'lng': 'lon'}, inplace=True)

print("Merging station coordinates...")
df_in = df_in.merge(df_stations[['station_id', 'lat', 'lon']], on='station_id', how='left')

# Drop invalid rows
if df_in['lat'].isnull().any():
    print(f"Dropping {df_in['lat'].isnull().sum()} records missing coordinates.")
    df_in = df_in.dropna(subset=['lat', 'lon'])

# --- 2. CORE LOGIC ---

def get_acmc_url(img_id, bucket_obj):
    if img_id in url_cache: return url_cache[img_id]
    
    # Logic from your successful diagnostic
    s_id = str(img_id).strip()
    
    if len(s_id) < 14: return None

    scan_start = "s" + s_id[:14]
    prefix = f"ABI-L2-ACMC/{s_id[:4]}/{s_id[4:7]}/{s_id[7:9]}/"
    
    blobs = bucket_obj.list_blobs(prefix=prefix)
    # Find the file that contains the scan start time
    match = next((b.name for b in blobs if scan_start in b.name), None)
    
    url = f"https://storage.googleapis.com/{bucket_obj.name}/{match}" if match else None
    url_cache[img_id] = url
    return url

def process_image_group(group_df, bucket_obj):
    img_id = group_df['GOES_img_id'].iloc[0]
    acmc_url = get_acmc_url(img_id, bucket_obj)
    
    # Initialize columns with NaN
    for col in ['ACMC_BCM', 'ACMC_ACM', 'ACMC_DQF']:
        group_df[col] = np.nan
    group_df['ACMC_time_start'] = pd.NaT
    group_df['ACMC_time_end'] = pd.NaT

    if not acmc_url: 
        return group_df

    try:
        with fsspec.open(acmc_url) as fp:
            with xr.open_dataset(fp, engine="h5netcdf") as ds:
                # Calculate scan angles using station lat/lon
                x, y = tools.lat_lon_to_scan_angles(
                    group_df['lat'].values, 
                    group_df['lon'].values, 
                    ds.goes_imager_projection
                )
                
                # Create DataArrays for indexing
                x_idx = xr.DataArray(x, dims="z")
                y_idx = xr.DataArray(y, dims="z")
                
                # Extract values nearest to station location
                group_df['ACMC_BCM'] = ds['BCM'].sel(x=x_idx, y=y_idx, method='nearest').values
                group_df['ACMC_ACM'] = ds['ACM'].sel(x=x_idx, y=y_idx, method='nearest').values
                group_df['ACMC_DQF'] = ds['DQF'].sel(x=x_idx, y=y_idx, method='nearest').values
                
                # Add Metadata
                group_df['ACMC_time_start'] = pd.Timestamp(ds.time_coverage_start)
                group_df['ACMC_time_end'] = pd.Timestamp(ds.time_coverage_end)
                
    except Exception as e:
        # Just log error and return the dataframe with NaNs
        # print(f"Error processing {img_id}: {e}") 
        pass

    return group_df

# --- 3. EXECUTION LOOP ---

all_ids = set(df_in['GOES_img_id'].unique())
processed_ids = set()

# Resume Logic
if os.path.exists(CSV_OUT_PATH):
    print(f"Resuming from {CSV_OUT_PATH}")
    try:
        # Load processed IDs as strings
        processed_ids = set(pd.read_csv(CSV_OUT_PATH, usecols=['GOES_img_id'], dtype={'GOES_img_id': str})['GOES_img_id'])
        print(f"Found {len(processed_ids)} processed images.")
    except: 
        print("Could not read progress file. Starting fresh.")

ids_to_process = list(all_ids - processed_ids)
print(f"Remaining images: {len(ids_to_process)}")

new_data = []
# Determine file mode: 'w' (overwrite) if fresh, 'a' (append) if resuming
write_header = not os.path.exists(CSV_OUT_PATH) or len(processed_ids) == 0
file_mode = 'a' if not write_header else 'w'

for i, img_id in tqdm(enumerate(ids_to_process), total=len(ids_to_process)):
    group = df_in[df_in['GOES_img_id'] == img_id].copy()
    result = process_image_group(group, bucket)
    new_data.append(result)
    
    # Save Checkpoint
    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        pd.concat(new_data).to_csv(CSV_OUT_PATH, mode=file_mode, header=write_header, index=False)
        new_data = [] # Clear memory
        file_mode = 'a' # Always append after first write
        write_header = False

# Save Final Batch
if new_data:
    pd.concat(new_data).to_csv(CSV_OUT_PATH, mode=file_mode, header=write_header, index=False)

print("\nDone.")