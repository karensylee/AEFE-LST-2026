# Downloads 5-min Tsrf_1_Avg data from HCDP for specified stations
# Includes retry logic and error handling for API failures

import os
import time
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

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10
API_TIMEOUT_SECONDS = 60


def fetch_data_with_retries(url, params, headers, max_retries=MAX_RETRIES):
    """Attempts to fetch data from the API with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            
            # Check if response is valid JSON
            data = response.json()
            if isinstance(data, list):
                return data
            else:
                print(f"    - WARNING: Unexpected response type: {type(data)}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"    - WARNING: Request timed out (Attempt {attempt + 1}/{max_retries}).")
        except requests.exceptions.RequestException as e:
            print(f"    - WARNING: Request failed (Attempt {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            print(f"    - WARNING: JSON decode error (Attempt {attempt + 1}/{max_retries}): {e}")

        if attempt + 1 < max_retries:
            print(f"    - Waiting {RETRY_WAIT_SECONDS} seconds before retrying...")
            time.sleep(RETRY_WAIT_SECONDS)
        else:
            print(f"    - ERROR: Final attempt failed. Skipping this batch.")
            return None

    return None


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prep station list
stations = pd.read_csv(STATIONS_LIST_PATH)
stations['station_id'] = stations['station_id'].astype(str).str.zfill(4)
stations['start_date'] = pd.to_datetime(stations['start_date']).dt.tz_convert('UTC')

# Filter out stations without start dates
stations = stations.dropna(subset=['start_date'])
print(f"Loaded {len(stations)} stations with start dates.")

# Resume logic: check which stations are already downloaded
completed_files = os.listdir(OUTPUT_DIR)
completed_station_ids = {f.split('_')[0] for f in completed_files if f.endswith(f'_{VAR_ID}.csv')}
print(f"Found {len(completed_station_ids)} stations already completed.")

stations_to_process = stations[~stations['station_id'].isin(completed_station_ids)]
print(f"Remaining stations to process: {len(stations_to_process)}")

if stations_to_process.empty:
    print("All stations already downloaded. Nothing to do.")
    exit()

# Download loop
for idx, row in stations_to_process.iterrows():
    sid = row['station_id']
    s_date = row['start_date']
    e_date = pd.Timestamp.now(tz='UTC')
    
    print(f"\nProcessing Station: {sid}")

    # Generate monthly chunks to prevent timeouts
    months = pd.date_range(start=s_date, end=e_date, freq='MS')
    
    monthly_data_frames = []
    failed_periods = []
    
    for m in months:
        month_end = m + pd.offsets.MonthEnd(0)
        if month_end > e_date:
            month_end = e_date
            
        print(f"  Fetching {m.strftime('%Y-%m')}...", end=" ")
        
        params = {
            'station_ids': sid,
            'var_ids': VAR_ID,
            'start_date': m.isoformat(),
            'end_date': month_end.isoformat()
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        data = fetch_data_with_retries(API_URL, params, headers)
        
        if data:
            monthly_data_frames.append(pd.DataFrame(data))
            print(f"✓ ({len(data)} rows)")
        elif data is None:
            failed_periods.append(m.date())
            print("✗ (failed)")
        else:
            print("- (no data)")
        
        time.sleep(0.25)  # Rate limiting

    # Concatenate and Save
    if monthly_data_frames:
        full_df = pd.concat(monthly_data_frames, ignore_index=True)
        if not full_df.empty:
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
            full_df.sort_values(by='timestamp', inplace=True)
            
            save_path = os.path.join(OUTPUT_DIR, f"{sid}_{VAR_ID}.csv")
            full_df.to_csv(save_path, index=False)
            print(f"✅ Saved {len(full_df)} records to {save_path}")
            
            if failed_periods:
                print(f"   ⚠️  Missing data for months: {failed_periods}")
        else:
            print(f"Station {sid}: No data found.")
    else:
        print(f"Station {sid}: No data retrieved.")

print("\n--- Download Complete ---")