import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import gc
from config import settings


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective for XGBoost.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5, log=True),
        'n_estimators': 1000,  # Max value, early stopping will find optimum
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': settings.RANDOM_STATE,
        'early_stopping_rounds': 20
    }
    
    model = xgb.XGBRegressor(**params)
    
    # XGBoost requires evaluation set for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    # Store metrics for retrieval
    trial.set_user_attr('best_iteration', model.best_iteration)
    trial.set_user_attr('val_rmse', rmse)
    
    return r2


def tune_hyperparameters(X, y, n_trials=100, random_state=None):
    """
    Runs Optuna study to find best hyperparameters.
    Splits input X, y into internal train/val for the study.
    
    Args:
        X: Feature array
        y: Target array
        n_trials: Number of Optuna trials (default 100)
        random_state: Random state for reproducibility
        
    Returns:
        dict: Best hyperparameters including optimal n_estimators
    """
    from sklearn.model_selection import train_test_split
    
    if random_state is None:
        random_state = settings.RANDOM_STATE
    
    # Create an internal split for tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"\n    - Tuning split: {len(X_train):,} train / {len(X_val):,} val samples")
    print(f"    - Running {n_trials} Optuna trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    # Suppress Optuna's default logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_params['n_estimators'] = study.best_trial.user_attrs['best_iteration']
    
    # Print best results
    print(f"\n✓ Optuna Tuning Complete")
    print(f"    - Best R²: {study.best_value:.6f}")
    print(f"    - Best RMSE: {study.best_trial.user_attrs['val_rmse']:.4f}")
    print(f"    - Best n_estimators: {best_params['n_estimators']}")
    
    return best_params


def train_final_model(X_train, y_train, params):
    """
    Trains final model with best params on provided data.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: XGBoost parameters dict
        
    Returns:
        Trained XGBRegressor model
    """
    # Ensure critical params are set
    params = params.copy()
    params['device'] = 'cuda'
    params['tree_method'] = 'hist'
    
    # Remove early_stopping_rounds if present, as we use fixed n_estimators from tuning
    if 'early_stopping_rounds' in params:
        del params['early_stopping_rounds']
        
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model
