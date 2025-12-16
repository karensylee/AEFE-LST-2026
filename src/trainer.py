import xgboost as xgb
import optuna
from sklearn.metrics import r2_score
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
        # 'device': 'cuda', # Using settings or auto-detect to avoid crash in CPU env
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
    
    # Store best iteration
    trial.set_user_attr('best_iteration', model.best_iteration)
    
    return r2

def tune_hyperparameters(X, y, n_trials=50, random_state=42):
    """
    Runs Optuna study to find best hyperparameters.
    Splits input X, y into internal train/val for the study.
    """
    from sklearn.model_selection import train_test_split
    
    # Create an internal split for tuning
    # Note: In a real scenario, this should be group-aware (by station) like in the notebook.
    # For this function, we assume X and y are already prepped (e.g., from train_station_ids).
    # If passed directly, standard random split is a simplification.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    best_params = study.best_params.copy()
    best_params['n_estimators'] = study.best_trial.user_attrs['best_iteration']
    
    return best_params

def train_final_model(X_train, y_train, params):
    """
    Trains final model with best params on provided data.
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
