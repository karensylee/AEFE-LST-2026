import pandas as pd
import os
import argparse

def merge_stations_with_embeddings(stations_path, embeddings_path, output_path):
    """
    Merges station metadata with valid embeddings using station_id as the key.
    
    Args:
        stations_path (str): Path to stations.csv
        embeddings_path (str): Path to embeddings csv
        output_path (str): Path to save the merged output
    """
    print(f"Reading stations from: {stations_path}")
    stations_df = pd.read_csv(stations_path)
    
    print(f"Reading embeddings from: {embeddings_path}")
    embeddings_df = pd.read_csv(embeddings_path)
    
    # Ensure station_id is same type (usually integer or string)
    # Checking types first
    print(f"Stations ID dtype: {stations_df['station_id'].dtype}")
    print(f"Embeddings ID dtype: {embeddings_df['station_id'].dtype}")
    
    # Merge
    # We use inner join to keep only stations that have embeddings, 
    # or left join if we want to keep all stations? 
    # Usually for ML we need the features, so inner join is safer unless specified otherwise.
    # However, standard practice might be to keep station metadata and just have nulls if missing, 
    # but here the goal is likely to prepare the feature file for these stations.
    # Let's assume we want to attach embeddings to stations.
    
    # Check for overlapping columns to avoid duplication (suffixes)
    # Both have 'name', 'full_name', 'status', 'lat', 'lon'/'lng', 'elevation', 'timezone', 'start_date'
    # We probably trust stations.csv for metadata more, or they match.
    # Let's drop overlapping columns from embeddings before merge, or handle suffixes.
    
    cols_to_use_from_embeddings = ['station_id'] + [f'A{i:02d}' for i in range(64)]
    
    # Filter embeddings to just ID and A00-A63
    embeddings_subset = embeddings_df[cols_to_use_from_embeddings]
    
    merged_df = pd.merge(stations_df, embeddings_subset, on='station_id', how='left')
    
    print(f"Merged shape: {merged_df.shape}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged file to: {output_path}")

if __name__ == "__main__":
    # Default paths based on project structure
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    STATIONS_FP = os.path.join(BASE_DIR, 'datasets/stations/stations.csv')
    EMBEDDINGS_FP = os.path.join(BASE_DIR, 'datasets/raw/aef/hawaii_station_alphaearth_embeddings_2024.csv')
    OUTPUT_FP = os.path.join(BASE_DIR, 'datasets/stations/stations_aef.csv')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stations', default=STATIONS_FP)
    parser.add_argument('--embeddings', default=EMBEDDINGS_FP)
    parser.add_argument('--output', default=OUTPUT_FP)
    
    args = parser.parse_args()
    
    merge_stations_with_embeddings(args.stations, args.embeddings, args.output)
