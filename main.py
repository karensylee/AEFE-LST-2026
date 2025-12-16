import argparse
import os
import joblib
import pandas as pd
import numpy as np
from config import settings
from src import data_loader, trainer, evaluation

def main():
    parser = argparse.ArgumentParser(description="Train LST XGBoost Models")
    parser.add_argument('--model_type', type=str, required=True, choices=settings.FEATURE_SETS.keys(), help="Feature set to use")
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning, otherwise use defaults")
    args = parser.parse_args()

    print(f"--- Starting Pipeline for {args.model_type} ---")
    
    # Setup - Create nested directory structure for model type
    model_output_dir = os.path.join(settings.OUTPUT_DIR, 'xgb', args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(settings.TEMP_PRED_DIR, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    features = settings.FEATURE_SETS[args.model_type]
    # We load target + features + metadata needed for splitting/eval
    cols_needed = list(set(features + [settings.TARGET_COL] + settings.METADATA))
    
    df = data_loader.load_data(settings.DATA_PATH, settings.TARGET_COL, cols_needed)
    print(f"Data loaded: {df.shape}")
    
    # Scale Data
    print("Scaling features...")
    df, _ = data_loader.scale_features(df)

    # Global Tuning
    # For simplicity in this script, we'll do a simple split if tuning is requested.
    # Ideally, this should respect station groups (Leave-Group-Out) as per the notebook.
    best_params = settings.DEFAULT_XGB_PARAMS.copy()
    
    if args.tune:
        print("Starting Hyperparameter Tuning...")
        # Get list of stations to split by station
        stations = df['SITE_ID'].unique()
        # Simple random shuffle of stations for 80/20 Tuning Split
        np.random.shuffle(stations)
        n_train = int(len(stations) * 0.8)
        train_stations = stations[:n_train]
        
        tuning_df = df[df['SITE_ID'].isin(train_stations)]
        
        X_tune = tuning_df[features].to_numpy()
        y_tune = tuning_df[settings.TARGET_COL].to_numpy()
        
        best_params = trainer.tune_hyperparameters(X_tune, y_tune)
        print(f"Best Params found: {best_params}")
    else:
        print("Using default hyperparameters.")

    # LOSO Loop
    stations = sorted(df['SITE_ID'].unique())
    all_results = []
    
    for station in stations:
        print(f"Processing station: {station}")
        
        # Checkpoint check
        pred_file = os.path.join(settings.TEMP_PRED_DIR, f"pred_{station}.csv")
        # Removing skip logic to ensure re-training "for real"
        # if os.path.exists(pred_file):
        #     print(f"Skipping {station}, predictions exist.")
        #     continue
            
        train_df, test_df = data_loader.get_station_split(df, station)
        
        X_train = train_df[features].to_numpy()
        y_train = train_df[settings.TARGET_COL].to_numpy()
        X_test = test_df[features].to_numpy()
        y_test = test_df[settings.TARGET_COL].to_numpy()
        
        model = trainer.train_final_model(X_train, y_train, best_params)
        
        # Inference
        preds = model.predict(X_test)
        
        # Metrics
        y_test_flat = y_test.ravel()
        cloud_mask = test_df['ACMC_BCM'].to_numpy() # Assuming column exists
        metrics = evaluation.calculate_metrics(y_test_flat, preds, cloud_mask)
        metrics['station_id'] = station
        all_results.append(metrics)
        
        print(f"  R2: {metrics['r2_overall']:.4f}, RMSE: {metrics['rmse_overall']:.4f}, MAE: {metrics['mae_overall']:.4f}, MRE: {metrics['mre_overall']:.2f}%")
        
        # Save Predictions
        output_df = pd.DataFrame({
            'station_id': station,
            'true': y_test_flat,
            'pred': preds,
            'sky_condition': cloud_mask,
            'local_time': test_df['LOCAL_TIME'].values if 'LOCAL_TIME' in test_df else None
        })
        output_df.to_csv(pred_file, index=False)
        
        # Save Model
        joblib.dump(model, os.path.join(model_output_dir, f"model_{station}.joblib"))

    # Save Aggregate Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(model_output_dir, f"results_{args.model_type}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved aggregate results to {results_path}")

if __name__ == "__main__":
    main()
