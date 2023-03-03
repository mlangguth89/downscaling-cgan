import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter

def crps_ensemble(observation, forecasts):
    """
    This function is currently unused, but is equivalent to properscoring's crps_ensemble.

    If numba is unavailable, properscoring will fall back to a more naive algorithm that is very inefficient in time and memory.  Use this instead!
    """

    fc = forecasts.copy()
    fc.sort(axis=-1)
    obs = observation
    fc_below = fc < obs[..., None]
    crps = np.zeros_like(obs)

    for i in range(fc.shape[-1]):
        below = fc_below[..., i]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[..., i][below])

    for i in range(fc.shape[-1] - 1, -1, -1):
        above = ~fc_below[..., i]
        k = fc.shape[-1] - 1 - i
        weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
        crps[above] += weight * (fc[..., i][above] - obs[above])

    return crps

def calculate_pearsonr(array_1, array_2):
    if isinstance(array_1, list):
        array_1 = np.concatenate(array_1)
    if isinstance(array_2, list):
        array_2 = np.concatenate(array_2)
    vals_1 = array_1.flatten()
    vals_2 = array_2.flatten()
    
    return pearsonr(vals_1, vals_2)

def mae(x, y):
    return np.mean(np.abs(x - y))

def mse(x, y):
    return np.mean(np.square(x - y))

def rmse(x, y):
    return np.sqrt(mse(x, y))

def mae_95(y_true, y_pred):
    
    ''' 
    Mean absolute error above the 95th percentile
    '''
    
    cutoff_true = np.quantile(y_true, 0.95)
    cutoff_pred = np.quantile(y_pred, 0.95)
    
    # Account for degenerate values
    true_vals_ix = np.where(y_true >= cutoff_true)[0]
    pred_vals_ix = np.where(y_pred >= cutoff_pred)[0]
    min_index = max(true_vals_ix.min(), pred_vals_ix.min())
        
    mae_95 = np.abs(y_true[min_index:] - y_pred[min_index:]).mean()
    
    return mae_95


##
# The code below is based off the pysteps code, please refer to their license

def fss(obs_array, fcst_array, scale, thr, mode='constant'):

    """Accumulate ensemble forecast-observation pairs to an FSS object.
    Does ensemble mean of thresholded arrays.
    Parameters
    -----------

    fcst_array: array_like
        Array of shape (s, m, n, en) containing an ensemble of c forecast fields, s samples, or (s,m,n) if just
        one ensemble member (will be converted to the right shape)
    obs_array: array_like
        Array of shape (s, m, n) containing s samples of the observation field
    """
    fcst_array_shape = fcst_array.shape
    obs_array_shape = obs_array.shape
    
    if len(fcst_array_shape) == len(obs_array_shape):
        assertion = fcst_array.shape != obs_array.shape
        fcst_array = np.expand_dims(fcst_array, axis=-1)
    else:
        assertion = fcst_array.shape[:-1] != obs_array.shape
        
    if assertion:
        message = "fcst_array and obs_array must havethe same image dimensions"
        raise ValueError(message)

    sum_obs_sq = 0
    sum_fct_obs = 0
    sum_fct_sq = 0

    for n in range(obs_array.shape[0]):
        X_o = obs_array[n, :, :]
        X_f = fcst_array[n, :, :, :]

        # Convert to binary fields with the given intensity threshold
        I_f = (X_f >= thr).astype(np.single)
        I_o = (X_o >= thr).astype(np.single)

        # Compute fractions of pixels above the threshold within a square
        # neighboring area by applying a 2D moving average to the binary fields
        if scale > 1:
            if mode == 'constant':
                S_o = uniform_filter(I_o, size=scale, mode="constant", cval=0.0)
            else:
                S_o = uniform_filter(I_o, size=scale, mode=mode)
        else:
            S_o = I_o
            
        for ii in range(X_f.shape[-1]):
            if scale > 1:
                if mode == 'constant':
                    S_f = uniform_filter(I_f[:, :, ii], size=scale, mode="constant", cval=0.0)
                else:
                        
                    S_f = uniform_filter(I_f[:, :, ii], size=scale, mode=mode)
                
            else:
                S_f = I_f[:, :, ii]
        
            sum_obs_sq += np.sum(S_o ** 2)
            sum_fct_obs += np.sum(S_f * S_o)
            sum_fct_sq += np.sum(S_f ** 2)
    
    numer =  2.0 * sum_fct_obs
    denom = sum_fct_sq + sum_obs_sq

    return numer / denom
    
    