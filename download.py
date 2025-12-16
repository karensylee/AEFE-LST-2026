# This downloads 5min LWout_1_Avg data from HCDP for specified stations and saves them as CSV files in datasets/raw/LWout_1_Avg/

import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load configuration
load_dotenv()
API_KEY = os.getenv("HCDP_API_KEY")
BASE_PATH = os.getenv("BASE_PATH")

# Construct paths dynamically
STATIONS_LIST_PATH = os.path.join(BASE_PATH, os.getenv("STATIONS_CSV"))
OUTPUT_DIR = os.path.join(BASE_PATH, "datasets", "raw", "Tsrf_1_Avg")
VAR_ID = "Tsrf_1_Avg"
API_URL = "https://api.hcdp.ikewai.org/mesonet/db/measurements"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prep station list
stations = pd.read_csv(STATIONS_LIST_PATH)
stations['station_id'] = stations['station_id'].astype(str).str.zfill(4)
stations['start_date'] = pd.to_datetime(stations['start_date']).dt.tz_convert('UTC')

# Download loop
for _, row in stations.iterrows():
    sid = row['station_id']
    s_date = row['start_date']
    e_date = pd.Timestamp.now(tz='UTC')
    
    print(f"Processing Station: {sid}")

    # Generate monthly chunks to prevent timeouts
    months = pd.date_range(start=s_date, end=e_date, freq='MS')
    
    # List comprehension to fetch data chunks
    data_frames = [
        pd.DataFrame(requests.get(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            params={
                'station_ids': sid,
                'var_ids': VAR_ID,
                'start_date': m.isoformat(),
                'end_date': (m + pd.offsets.MonthEnd(0)).isoformat()
            }
        ).json())
        for m in months
    ]

    # Concatenate and Save
    if data_frames:
        full_df = pd.concat(data_frames, ignore_index=True)
        if not full_df.empty:
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
            full_df.sort_values(by='timestamp', inplace=True)
            
            save_path = os.path.join(OUTPUT_DIR, f"{sid}_{VAR_ID}.csv")
            full_df.to_csv(save_path, index=False)
            print(f"Saved {len(full_df)} records to {save_path}")
        else:
            print(f"Station {sid}: No data found.")
    else:
        print(f"Station {sid}: No months to process.")

print("Download sequence complete.")