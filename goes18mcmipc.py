# This will save GOES-18 MCMIPC data sampled at ground station locations with -5 minute offset
# using LWout_1_Avg ground observations from 2024 only.
# The output CSV will be saved to Google Drive.
# the output CSV will be named goes18_lwout_samples_2024.csv, place this into datasets/raw/goes18cmipc/

import os
import pandas as pd
import ee
from dotenv import load_dotenv

# ==============================================================================
# 1. SETUP
# ==============================================================================
load_dotenv()
PROJECT = os.getenv("EEPROJECTNAME")
BASE_PATH = os.getenv("BASE_PATH")
STATION_LIST_PATH = os.path.join(BASE_PATH, os.getenv("STATIONS_CSV"))
GROUND_DATA_PATH = os.path.join(BASE_PATH, "datasets", "raw", "Tsrf_1_Avg")
EXPORT_FOLDER = "datasets/raw/goes18cmipc"
OUTPUT_FILENAME = "goes18_lst_samples_2024"

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project=PROJECT)

# ==============================================================================
# 2. PREPARE GROUND DATA (2024) UTC
# ==============================================================================
print("--- Loading and Filtering Ground Data (2024) ---")
station_lookup_df = pd.read_csv(STATION_LIST_PATH)

# Construct file paths
all_ground_files = [
    os.path.join(GROUND_DATA_PATH, f"{int(row['station_id']):04d}_Tsrf_1_Avg.csv")
    for _, row in station_lookup_df.iterrows()
]

# Read files (Scanning for timestamp and flag)
df_list = []
for f in all_ground_files:
    if os.path.exists(f):
        # We need 'flag' to filter quality, even if the snippet only showed timestamp
        try:
            # Check headers first to identify flag column
            headers = pd.read_csv(f, nrows=0).columns.tolist()
            flag_col = next((c for c in headers if 'flag' in c or 'qcode' in c), None)
            
            cols_to_use = ['timestamp', flag_col] if flag_col else ['timestamp']
            
            temp_df = pd.read_csv(f, usecols=cols_to_use, parse_dates=['timestamp'], on_bad_lines='skip')
            
            # Filter quality if flag column exists
            if flag_col:
                temp_df = temp_df[temp_df[flag_col] == 0]
            
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

# Combine
ground_df_combined = pd.concat(df_list, ignore_index=True).dropna(subset=['timestamp'])
ground_df_combined.rename(columns={'timestamp': 'date_time_utc'}, inplace=True)

# Ensure Timezone Awareness (UTC)
ground_df_combined['date_time_utc'] = pd.to_datetime(ground_df_combined['date_time_utc']).dt.tz_convert('UTC')
print("  - Timestamps normalized to UTC.")

# --- HARDCODED 2024 FILTER ---
print("  - Applying hardcoded 2024 filter (2024-01-01 00:05:00 to 2025-01-01 00:00:00)...")
start_2024_utc = pd.Timestamp('2024-01-01 00:00:00+00:00', tz='UTC')
end_2024_utc = pd.Timestamp('2025-01-01 00:00:00+00:00', tz='UTC')

ground_df_combined_2024_utc = ground_df_combined[
    (ground_df_combined['date_time_utc'] >= pd.Timestamp('2024-01-01 00:05:00+00:00', tz='UTC')) &
    (ground_df_combined['date_time_utc'] <= pd.Timestamp('2025-01-01 00:00:00+00:00', tz='UTC'))
].copy()

# Extract Unique Timestamps for GEE
timestamp_list_str = sorted([
    ts.strftime('%Y-%m-%dT%H:%M:%S') 
    for ts in ground_df_combined_2024_utc['date_time_utc'].unique()
])
ee_date_list = ee.List(timestamp_list_str)

print(f"✅ Prepared {len(timestamp_list_str)} unique 2024 UTC timestamps for GEE processing.")

# ==============================================================================
# 3. GEE STATION FEATURES
# ==============================================================================
def df_to_ee_feature(row):
    # Ensure station_id is string '0123'
    sid = str(int(row['station_id'])).zfill(4) 
    return ee.Feature(
        ee.Geometry.Point([row['lon'], row['lat']]), 
        {'station_id': sid}
    )

stations_fc = ee.FeatureCollection(
    station_lookup_df.apply(df_to_ee_feature, axis=1).tolist()
)

# ==============================================================================
# 4. SAMPLING LOGIC
# ==============================================================================
satellite = ee.ImageCollection("NOAA/GOES/18/MCMIPC").filterDate('2024-01-01', '2025-01-01')

def sample_goes_neg5min(timestamp_str):
    currentDate = ee.Date(timestamp_str)
    # Window: [-5 min, 0 min)
    windowStart = currentDate.advance(-5, 'minute') #important: advance(-5 goes back -5minutes.
    
    # Get most recent image in window
    image = satellite.filterDate(windowStart, currentDate).sort('system:time_start', False).first()

    def process_image(img):
        return img.reduceRegions(
            collection=stations_fc,
            reducer=ee.Reducer.first(),
            scale=2000
        ).map(lambda f: f.set({
            'ground_obs_time_utc': timestamp_str,
            'GOES_img_id': img.id(),
            'GOES_time_start': img.get('system:time_start')
        }))

    return ee.FeatureCollection(ee.Algorithms.If(
        image, 
        process_image(image), 
        ee.FeatureCollection([])
    ))

# Flatten results
final_collection = ee.FeatureCollection(ee_date_list.map(sample_goes_neg5min)).flatten()

# ==============================================================================
# 5. EXPORT
# ==============================================================================
print(f"--- Starting Export Task: {OUTPUT_FILENAME} ---")
export_task = ee.batch.Export.table.toDrive(
    collection=final_collection,
    description='GOES_Samples_Tsrf_Hardcoded2024',
    folder=os.path.basename(EXPORT_FOLDER),
    fileNamePrefix=OUTPUT_FILENAME,
    fileFormat='CSV'
)
export_task.start()
print(f"🚀 Task started with ID: {export_task.id}")