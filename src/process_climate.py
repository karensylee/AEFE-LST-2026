import pandas as pd
import os
from config import settings

def process_climate_divisions():
    # Paths
    # We are working with stations_aef.csv which is in datasets/stations/
    STATIONS_AEF_PATH = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef.csv')
    OUTPUT_PATH = os.path.join(settings.BASE_DIR, 'datasets/stations/stations_aef_hiclimatedivision.csv')
    
    print(f"Loading {STATIONS_AEF_PATH}...")
    df = pd.read_csv(STATIONS_AEF_PATH)
    
    if 'HICLIMATEDIVISION' not in df.columns:
        raise ValueError("HICLIMATEDIVISION column missing from input file.")
        
    print(f"Found {len(df)} stations.")
    print(f"Unique Climate Divisions: {df['HICLIMATEDIVISION'].unique()}")
    
    # One Hot Encoding
    # Prefix 'CD_' for Climate Division
    dummies = pd.get_dummies(df['HICLIMATEDIVISION'], prefix='CD')
    
    # Convert bool to int (optional but standard for ML)
    dummies = dummies.astype(int)
    
    # Concatenate with original dataframe
    # User said "HICLIMATEDIVSION column... turned into a one hot encoded and saved..." 
    # Usually implies keeping other features or at least ID. 
    # merging everything + dummies is safest.
    df_encoded = pd.concat([df, dummies], axis=1)
    
    # Save
    df_encoded.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved encoded data to {OUTPUT_PATH}")
    print("New columns added:", list(dummies.columns))

if __name__ == "__main__":
    process_climate_divisions()
