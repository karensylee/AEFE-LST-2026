import argparse
import os
import time
import joblib
import pandas as pd
import numpy as np
from config import settings
from src import data_loader, trainer, evaluation


def get_model_description(model_type):
    """Returns a description string for each model type."""
    descriptions = {
        'BLM': 'Baseline Model (Auxiliary + 16 CMI bands)',
        'BLAM': 'Baseline + AlphaEarth Foundation Embeddings (Auxiliary + 16 CMI + 64 AEFE)',
        'BLAM-C': 'Baseline + AEFE - CMI (Auxiliary + 64 AEFE only)',
        'CIM': 'CMI-Only Model (16 CMI bands, no auxiliary)',
        'CIAM': 'CMI + AEFE Model (16 CMI + 64 AEFE, no auxiliary)',
    }
    return descriptions.get(model_type, 'Unknown Model')


def print_header(model_type, features):
    """Prints a formatted header banner for the model."""
    print("=" * 80)
    print(f"{model_type}: {get_model_description(model_type)}")
    print("=" * 80)
    
    # Count feature categories
    cmi_count = sum(1 for f in features if f.startswith('CMI_'))
    aefe_count = sum(1 for f in features if f.startswith('A') and f[1:].isdigit())
    aux_count = len(features) - cmi_count - aefe_count
    
    print(f"\nModel Configuration:")
    print(f"    - CMI bands: {cmi_count} features")
    print(f"    - AEFE embeddings: {aefe_count} features")
    print(f"    - Auxiliary features: {aux_count} features")
    print(f"    - Total: {len(features)} features")
    
    print(f"\nBCM (Binary Cloud Mask) Values:")
    print(f"    - BCM = 0: Clear-sky conditions")
    print(f"    - BCM = 1: Cloudy-sky conditions")
    print("=" * 80)


def print_feature_breakdown(features):
    """Prints detailed feature breakdown by category."""
    print("\n--- Feature Set Breakdown ---")
    
    cmi_features = [f for f in features if f.startswith('CMI_')]
    aefe_features = [f for f in features if f.startswith('A') and f[1:].isdigit()]
    aux_features = [f for f in features if f not in cmi_features and f not in aefe_features]
    
    print(f"\nCMI Bands ({len(cmi_features)} features):")
    if cmi_features:
        print(f"    {cmi_features}")
    else:
        print("    (none)")
    
    print(f"\nAuxiliary Features ({len(aux_features)} features):")
    if aux_features:
        print(f"    {aux_features}")
    else:
        print("    (none)")
    
    print(f"\nAEFE Embeddings ({len(aefe_features)} features):")
    if aefe_features:
        if len(aefe_features) > 10:
            print(f"    {aefe_features[:5]} ... {aefe_features[-5:]}")
        else:
            print(f"    {aefe_features}")
    else:
        print("    (none)")
    
    print(f"\nTotal Features: {len(features)}")


def print_paths(model_output_dir):
    """Prints output directory paths."""
    print("\n--- Output Directories ---")
    print(f"    Input Data: {settings.DATA_PATH}")
    print(f"    Model Output: {model_output_dir}")
    print(f"    Temp Predictions: {settings.TEMP_PRED_DIR}")


def print_final_summary(all_results, model_type, total_time):
    """Prints final summary table with aggregated metrics."""
    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY: {model_type}")
    print("=" * 80)
    
    print(f"\nExecution Time: {total_time/60:.2f} minutes")
    
    if not all_results:
        print("\nNo results to summarize.")
        return
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    
    # Overall metrics
    for metric in ['r2_overall', 'rmse_overall', 'mae_overall', 'mre_overall']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric:<25} {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    # Clear-sky metrics (BCM=0)
    for metric in ['r2_clear', 'rmse_clear', 'mae_clear']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric} (BCM=0)"[:25].ljust(25) + f" {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    # Cloudy-sky metrics (BCM=1)
    for metric in ['r2_cloudy', 'rmse_cloudy', 'mae_cloudy']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric} (BCM=1)"[:25].ljust(25) + f" {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train LST XGBoost Models")
    parser.add_argument('--model_type', type=str, required=True, choices=settings.FEATURE_SETS.keys(), help="Feature set to use")
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning, otherwise use defaults")
    args = parser.parse_args()

    total_start_time = time.time()
    
    # Get features for this model
    features = settings.FEATURE_SETS[args.model_type]
    
    # --- Print Header Banner ---
    print_header(args.model_type, features)
    
    # Setup - Create nested directory structure for model type
    model_output_dir = os.path.join(settings.OUTPUT_DIR, 'xgb', args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(settings.TEMP_PRED_DIR, exist_ok=True)
    
    # Print paths
    print_paths(model_output_dir)
    
    # Print feature breakdown
    print_feature_breakdown(features)
    
    # --- Load Data ---
    print("\n--- Loading Data ---")
    cols_needed = list(set(features + [settings.TARGET_COL] + settings.METADATA))
    
    df = data_loader.load_data(settings.DATA_PATH, settings.TARGET_COL, cols_needed)
    print(f"✓ Successfully loaded data")
    print(f"    - Rows: {df.shape[0]:,}")
    print(f"    - Columns: {df.shape[1]}")
    print(f"    - Memory Usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # --- Scale Data ---
    print("\n--- Scaling Features ---")
    df, _ = data_loader.scale_features(df)
    print(f"✓ Features scaled")

    # --- Hyperparameter Tuning ---
    best_params = settings.DEFAULT_XGB_PARAMS.copy()
    
    if args.tune:
        print("\n--- Starting Hyperparameter Tuning with Optuna ---")
        stations = df['SITE_ID'].unique()
        np.random.shuffle(stations)
        n_train = int(len(stations) * 0.8)
        train_stations = stations[:n_train]
        val_stations = stations[n_train:]
        
        print(f"\nStation Split for Tuning:")
        print(f"    - Training Stations: {len(train_stations)}")
        print(f"    - Validation Stations: {len(val_stations)}")
        
        tuning_df = df[df['SITE_ID'].isin(train_stations)]
        
        X_tune = tuning_df[features].to_numpy()
        y_tune = tuning_df[settings.TARGET_COL].to_numpy()
        
        best_params = trainer.tune_hyperparameters(X_tune, y_tune)
        
        print(f"\n✓ Optuna Tuning Complete")
        print(f"    - Best Parameters:")
        for key, value in best_params.items():
            print(f"        {key}: {value}")
    else:
        print("\n--- Using Default Hyperparameters ---")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

    # --- LOSO Loop ---
    print("\n--- Starting Leave-One-Station-Out (LOSO) Cross-Validation ---")
    stations = sorted(df['SITE_ID'].unique())
    print(f"Total Stations: {len(stations)}")
    
    all_results = []
    
    for i, station in enumerate(stations):
        fold_start_time = time.time()
        print(f"\n[{i+1}/{len(stations)}] Hold-out Station: {station}")
        
        # Checkpoint check
        pred_file = os.path.join(settings.TEMP_PRED_DIR, f"pred_{station}.csv")
            
        train_df, test_df = data_loader.get_station_split(df, station)
        
        X_train = train_df[features].to_numpy()
        y_train = train_df[settings.TARGET_COL].to_numpy()
        X_test = test_df[features].to_numpy()
        y_test = test_df[settings.TARGET_COL].to_numpy()
        
        print(f"    - Training samples: {len(X_train):,}")
        print(f"    - Test samples: {len(X_test):,}")
        
        model = trainer.train_final_model(X_train, y_train, best_params)
        
        # Inference
        preds = model.predict(X_test)
        
        # Get BCM values for stratified metrics
        # BCM = 0: Clear-sky, BCM = 1: Cloudy-sky
        cloud_mask = test_df['ACMC_BCM'].to_numpy()
        n_clear = np.sum(cloud_mask == 0)
        n_cloudy = np.sum(cloud_mask == 1)
        
        # Calculate metrics using evaluation module
        y_test_flat = y_test.ravel()
        metrics = evaluation.calculate_metrics(y_test_flat, preds, cloud_mask)
        metrics['station_id'] = station
        all_results.append(metrics)
        
        # Print BCM-stratified results
        print(f"    - Overall:         R²={metrics['r2_overall']:.4f}, RMSE={metrics['rmse_overall']:.4f} K, MAE={metrics['mae_overall']:.4f} K")
        
        if 'r2_clear' in metrics and not np.isnan(metrics['r2_clear']):
            print(f"    - Clear-sky (BCM=0, n={n_clear:,}):  R²={metrics['r2_clear']:.4f}, RMSE={metrics['rmse_clear']:.4f} K")
        else:
            print(f"    - Clear-sky (BCM=0, n={n_clear:,}):  (no data)")
            
        if 'r2_cloudy' in metrics and not np.isnan(metrics['r2_cloudy']):
            print(f"    - Cloudy-sky (BCM=1, n={n_cloudy:,}): R²={metrics['r2_cloudy']:.4f}, RMSE={metrics['rmse_cloudy']:.4f} K")
        else:
            print(f"    - Cloudy-sky (BCM=1, n={n_cloudy:,}): (no data)")
        
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
        
        fold_time = time.time() - fold_start_time
        print(f"    - Fold completed in {fold_time:.2f} seconds")

    # --- Save Aggregate Results ---
    print("\n--- Saving Results ---")
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(model_output_dir, f"results_{args.model_type}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✓ Per-station results saved to: {results_path}")

    # --- Print Final Summary ---
    total_time = time.time() - total_start_time
    print_final_summary(all_results, args.model_type, total_time)
    
    print(f"\n✅ Pipeline complete for {args.model_type}")
    print(f"All models saved to: {model_output_dir}")


if __name__ == "__main__":
    main()
