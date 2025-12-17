import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def mean_relative_error(y_true, y_pred):
    """
    Calculates Mean Relative Error (MRE).
    MRE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
        
    Returns:
        float: Mean Relative Error as a percentage.
    """
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100

def calculate_metrics(y_true, y_pred, cloud_mask=None):
    """
    Calculates R2, RMSE, MAE, and MRE for Overall, Clear, and Cloudy conditions.
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
        cloud_mask (np.array): Binary Cloud Mask (BCM) array (0=Clear-sky, 1=Cloudy-sky).
        
    Returns:
        dict: Dictionary containing metrics.
    """
    
    # Overall
    metrics = {
        'r2_overall': r2_score(y_true, y_pred),
        'rmse_overall': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae_overall': mean_absolute_error(y_true, y_pred),
        'mre_overall': mean_relative_error(y_true, y_pred)
    }
    
    if cloud_mask is not None:
        # BCM convention: 0 = clear-sky, 1 = cloudy-sky
        clear_mask = (cloud_mask == 0)
        cloudy_mask = (cloud_mask == 1)
        
        # Clear Sky
        if np.sum(clear_mask) > 0:
            metrics['r2_clear'] = r2_score(y_true[clear_mask], y_pred[clear_mask])
            metrics['rmse_clear'] = np.sqrt(mean_squared_error(y_true[clear_mask], y_pred[clear_mask]))
            metrics['mae_clear'] = mean_absolute_error(y_true[clear_mask], y_pred[clear_mask])
            metrics['mre_clear'] = mean_relative_error(y_true[clear_mask], y_pred[clear_mask])
        else:
            metrics['r2_clear'] = np.nan
            metrics['rmse_clear'] = np.nan
            metrics['mae_clear'] = np.nan
            metrics['mre_clear'] = np.nan
            
        # Cloudy Sky
        if np.sum(cloudy_mask) > 0:
            metrics['r2_cloudy'] = r2_score(y_true[cloudy_mask], y_pred[cloudy_mask])
            metrics['rmse_cloudy'] = np.sqrt(mean_squared_error(y_true[cloudy_mask], y_pred[cloudy_mask]))
            metrics['mae_cloudy'] = mean_absolute_error(y_true[cloudy_mask], y_pred[cloudy_mask])
            metrics['mre_cloudy'] = mean_relative_error(y_true[cloudy_mask], y_pred[cloudy_mask])
        else:
            metrics['r2_cloudy'] = np.nan
            metrics['rmse_cloudy'] = np.nan
            metrics['mae_cloudy'] = np.nan
            metrics['mre_cloudy'] = np.nan
            
    return metrics
