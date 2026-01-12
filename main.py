import argparse
import os
import time
import gc
import joblib
import pandas as pd
import numpy as np
from config import settings
from src import data_loader, trainer, evaluation


def get_model_description(model_type):
    """Returns a description string for each model type."""
    descriptions = {
        'B-E': 'Baseline Model (Auxiliary + 16 CMI bands)',
        'B': 'Baseline + AlphaEarth Foundation Embeddings (Auxiliary + 16 CMI + 64 AEFE)',
        'B-C': 'Baseline + AEFE - CMI (Auxiliary + 64 AEFE only)',
        'B-E-X': 'CMI-Only Model (16 CMI bands, no auxiliary)',
        'B-X': 'CMI + AEFE Model (16 CMI + 64 AEFE, no auxiliary)',
        'TTM': 'TopTenModel (Top 10 features)',
        'B-ALL': 'Full B Model (Trained on ALL data, no CV)',
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


def print_paths(model_output_dir, temp_pred_dir):
    """Prints output directory paths."""
    print("\n--- Output Directories ---")
    print(f"    Input Data: {settings.DATA_PATH}")
    print(f"    Model Output: {model_output_dir}")
    print(f"    Temp Predictions: {temp_pred_dir}")


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
    for metric in ['r2_overall', 'rmse_overall', 'mae_overall', 'mre_overall', 'bias_overall']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric:<25} {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    # Clear-sky metrics (BCM=0)
    for metric in ['r2_clear', 'rmse_clear', 'mae_clear', 'bias_clear']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric} (BCM=0)"[:25].ljust(25) + f" {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    # Cloudy-sky metrics (BCM=1)
    for metric in ['r2_cloudy', 'rmse_cloudy', 'mae_cloudy', 'bias_cloudy']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric} (BCM=1)"[:25].ljust(25) + f" {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train LST XGBoost Models")
    parser.add_argument('--model_type', type=str, required=True, choices=settings.FEATURE_SETS.keys(), help="Feature set to use")
    parser.add_argument('--best_params', type=str, help="Path to JSON file containing best parameters (skips tuning)")
    args = parser.parse_args()

    total_start_time = time.time()
    
    # Get features for this model
    features = settings.FEATURE_SETS[args.model_type]
    
    # --- Print Header Banner ---
    print_header(args.model_type, features)
    
    # Setup - Create nested directory structure for model type
    model_output_dir = os.path.join(settings.OUTPUT_DIR, 'xgb', args.model_type)
    temp_pred_dir = os.path.join(model_output_dir, 'loso_temp_predictions')
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(temp_pred_dir, exist_ok=True)
    
    # Print paths
    print_paths(model_output_dir, temp_pred_dir)
    
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
    
    # --- Scale Data Globally ---
    # --- Scale Data Globally ---
    print("\n--- Scaling Features ---")
    df, global_scaler = data_loader.scale_features(df)
    print(f"✓ Features scaled globally")

    # --- Hyperparameter Tuning (Mandatory) ---
    print("\n--- Starting Hyperparameter Tuning with Optuna ---")
    stations = df['SITE_ID'].unique()
    np.random.seed(settings.RANDOM_STATE)  # Reproducible split
    np.random.shuffle(stations)
    n_train = int(len(stations) * 0.8)
    train_stations = sorted(stations[:n_train].tolist())
    val_stations = sorted(stations[n_train:].tolist())
    
    print(f"\nStation Split for Tuning (seed={settings.RANDOM_STATE}):")
    print(f"    - Training Stations ({len(train_stations)}): {train_stations}")
    print(f"    - Validation Stations ({len(val_stations)}): {val_stations}")
    
    # Save station split to JSON for reference
    import json
    split_info = {
        'random_seed': settings.RANDOM_STATE,
        'train_stations': train_stations,
        'val_stations': val_stations,
        'train_count': len(train_stations),
        'val_count': len(val_stations)
    }
    split_path = os.path.join(model_output_dir, f"{args.model_type}_optuna_station_split.json")
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"    - Split saved to: {split_path}")
    
    tuning_df = df[df['SITE_ID'].isin(train_stations)]
    
    X_tune = tuning_df[features].to_numpy()
    y_tune = tuning_df[settings.TARGET_COL].to_numpy()
    
    if args.best_params and os.path.exists(args.best_params):
        print(f"\n--- Loading Custom Parameters (Skipping Tuning) ---")
        import json
        with open(args.best_params, 'r') as f:
            best_params = json.load(f)
        print(f"✓ Loaded parameters from: {args.best_params}")
    else:
        best_params = trainer.tune_hyperparameters(X_tune, y_tune)
    
    print(f"\n✓ Optuna Tuning Complete (or Loaded)")
    print(f"    - Best Parameters:")
    for key, value in best_params.items():
        print(f"        {key}: {value}")

    # --- Training Execution Path ---
    if args.model_type.startswith('B-ALL'):
        # === FULL DATASET TRAINING (No Cross-Validation) ===
        print(f"\n--- Starting Full Dataset Training (No CV) ---")
        print(f"Model Type: {args.model_type}")
        
        # Filter Data based on Model Type Variant
        if 'CLEAR' in args.model_type:
            print("Filtering for CLEAR SKY only (ACMC_BCM = 0)")
            train_df = df[df['ACMC_BCM'] == 0]
        elif 'CLOUDY' in args.model_type:
            print("Filtering for CLOUDY SKY only (ACMC_BCM = 1)")
            train_df = df[df['ACMC_BCM'] == 1]
        else:
            print("Using ALL Sky Conditions")
            train_df = df
            
        print(f"Training Data Size: {len(train_df):,} rows")
        
        # Train on Selected Data
        X_all = train_df[features].to_numpy()
        y_all = train_df[settings.TARGET_COL].to_numpy()
        
        print(f"Training final model on available data...")
        model = trainer.train_final_model(X_all, y_all, best_params)
        
        # Save Final Model
        model_path = os.path.join(model_output_dir, f"{args.model_type}_model.joblib")
        joblib.dump(model, model_path)
        print(f"✓ Final model saved to: {model_path}")
        
        # Save Global Scaler (Only if it exists)
        if global_scaler is not None:
            scaler_path = os.path.join(model_output_dir, f"{args.model_type}_scaler.joblib")
            joblib.dump(global_scaler, scaler_path)
            print(f"✓ Global scaler saved to: {scaler_path}")
        else:
             print(f"✓ No scaler used/saved for {args.model_type}")
        
        # Calculate Training Metrics (Optional but good for sanity check)
        print("\nCalculating Training Metrics (Self-Prediction)...")
        preds = model.predict(X_all)
        cloud_mask = train_df['ACMC_BCM'].to_numpy()
        
        metrics = evaluation.calculate_metrics(y_all, preds, cloud_mask)
        print(f"    - Overall Training R²={metrics['r2_overall']:.4f}, RMSE={metrics['rmse_overall']:.4f} K")
        
        # Save Training Predictions 
        pred_file = os.path.join(temp_pred_dir, f"preds_{args.model_type}_TRAIN.csv")
        output_df = pd.DataFrame({
            'LST_true': y_all,
            'LST_pred': preds,
            'station_id': df['SITE_ID'],
            'LOCAL_TIME': df['LOCAL_TIME'].values if 'LOCAL_TIME' in df else None,
            'ACMC_BCM': cloud_mask
        })
        output_df.to_csv(pred_file, index=False)
        print(f"✓ Training predictions saved to: {pred_file}")

    else:
        # === LEAVE-ONE-STATION-OUT (LOSO) CV ===
        print("\n--- Starting Leave-One-Station-Out (LOSO) Cross-Validation ---")
    stations = sorted(df['SITE_ID'].unique())
    print(f"Total Stations: {len(stations)}")
    
    all_results = []
    
    for i, station in enumerate(stations):
        fold_start_time = time.time()
        print(f"\n[{i+1}/{len(stations)}] Hold-out Station: {station}")
        
        # Checkpoint check - skip if predictions already exist
        pred_file = os.path.join(temp_pred_dir, f"preds_{station}.csv")
        if os.path.exists(pred_file):
            print(f"  → Skipping (predictions already exist)")
            continue
            
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
        print(f"    - Overall:         R²={metrics['r2_overall']:.4f}, RMSE={metrics['rmse_overall']:.4f} K, Bias={metrics['bias_overall']:.4f} K")
        
        if 'r2_clear' in metrics and not np.isnan(metrics['r2_clear']):
            print(f"    - Clear-sky (BCM=0, n={n_clear:,}):  R²={metrics['r2_clear']:.4f}, RMSE={metrics['rmse_clear']:.4f} K")
        else:
            print(f"    - Clear-sky (BCM=0, n={n_clear:,}):  (no data)")
            
        if 'r2_cloudy' in metrics and not np.isnan(metrics['r2_cloudy']):
            print(f"    - Cloudy-sky (BCM=1, n={n_cloudy:,}): R²={metrics['r2_cloudy']:.4f}, RMSE={metrics['rmse_cloudy']:.4f} K")
        else:
            print(f"    - Cloudy-sky (BCM=1, n={n_cloudy:,}): (no data)")
        
        # Save Predictions with consistent column names
        output_df = pd.DataFrame({
            'LST_true': y_test_flat,
            'LST_pred': preds,
            'station_id': station,
            'LOCAL_TIME': test_df['LOCAL_TIME'].values if 'LOCAL_TIME' in test_df else None,
            'ACMC_BCM': cloud_mask
        })
        output_df.to_csv(pred_file, index=False)
        
        # Save Model
        model_path = os.path.join(model_output_dir, f"{args.model_type}_model_{station}.joblib")
        joblib.dump(model, model_path)
        
        # Save Scaler (global scaler, saved per fold for reproducibility/reference)
        scaler_path = os.path.join(model_output_dir, f"{args.model_type}_scaler_{station}.joblib")
        joblib.dump(global_scaler, scaler_path)
        
        fold_time = time.time() - fold_start_time
        print(f"    - Fold completed in {fold_time:.2f} seconds")
        
        # Memory cleanup
        del model, X_train, y_train, X_test, y_test, preds, train_df, test_df
        gc.collect()

    # --- Aggregate All Predictions ---
    print("\n--- Aggregating All Predictions ---")
    all_preds_list = []
    for station in stations:
        pred_path = os.path.join(temp_pred_dir, f"preds_{station}.csv")
        if os.path.exists(pred_path):
            all_preds_list.append(pd.read_csv(pred_path))
    
    if all_preds_list:
        all_preds_df = pd.concat(all_preds_list, ignore_index=True)
        all_preds_path = os.path.join(model_output_dir, f"{args.model_type}_ALL_predictions.csv")
        all_preds_df.to_csv(all_preds_path, index=False)
        print(f"✓ All predictions aggregated to: {all_preds_path}")
        print(f"    - Total predictions: {len(all_preds_df):,}")

    # --- Save Aggregate Results ---
    print("\n--- Saving Results ---")
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(model_output_dir, f"{args.model_type}_loso_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✓ Per-station results saved to: {results_path}")

    # --- Print Final Summary ---
    total_time = time.time() - total_start_time
    print_final_summary(all_results, args.model_type, total_time)
    
    print(f"\n✅ Pipeline complete for {args.model_type}")
    print(f"All models saved to: {model_output_dir}")


if __name__ == "__main__":
    main()
