import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config.settings import SCALING_COLS

def load_data(path, target_col, feature_cols):
    """
    Loads specific columns from the CSV and optimizes types.
    
    Args:
        path (str): Path to the CSV file.
        target_col (str): Name of the target variable column.
        feature_cols (list): List of feature column names.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    # Ensure we load essential columns even if not in features (like SITE_ID for splitting)
    # The config feature sets usually include METADATA which has SITE_ID.
    # We remove duplicates just in case.
    cols_to_load = list(set(feature_cols + [target_col]))
    
    # Define dtypes for memory optimization
    dtype_map = {col: np.float32 for col in cols_to_load if col not in ['SITE_ID', 'LOCAL_TIME']}
    if 'SITE_ID' in cols_to_load:
        dtype_map['SITE_ID'] = str
    if 'LOCAL_TIME' in cols_to_load:
        dtype_map['LOCAL_TIME'] = str
        
    try:
        df = pd.read_csv(path, usecols=cols_to_load, dtype=dtype_map)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}")

def scale_features(df, features_to_scale=SCALING_COLS, scaler=None):
    """
    Scales specified features to [0, 1] range.
    
    Args:
        df (pd.DataFrame): Dataframe containing features.
        features_to_scale (list): List of columns to scale.
        scaler (MinMaxScaler, optional): Existing scaler to use. If None, fits a new one.
        
    Returns:
        tuple: (scaled_df, scaler)
    """
    # Filter features that actully exist in the dataframe
    valid_cols = [c for c in features_to_scale if c in df.columns]
    
    if not valid_cols:
        return df, scaler
        
    if scaler is None:
        scaler = MinMaxScaler()
        df[valid_cols] = scaler.fit_transform(df[valid_cols])
    else:
        df[valid_cols] = scaler.transform(df[valid_cols])
        
    return df, scaler

def get_station_split(df, holdout_id):
    """
    Returns train/test split for Leave-One-Station-Out (LOSO).
    
    Args:
        df (pd.DataFrame): Full dataframe.
        holdout_id (str): SITE_ID to hold out for testing.
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df = df[df['SITE_ID'] != holdout_id]
    test_df = df[df['SITE_ID'] == holdout_id]
    return train_df, test_df
