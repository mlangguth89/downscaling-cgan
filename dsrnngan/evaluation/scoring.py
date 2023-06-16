import numpy as np
from tqdm import tqdm
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

def mae_above_threshold(y_true, y_pred, percentile_threshold=0.95):
    
    ''' 
    Mean absolute error above the 95th percentile
    '''
    
    cutoff_true = np.quantile(y_true, percentile_threshold)
    cutoff_pred = np.quantile(y_pred, percentile_threshold)
    
    # Account for degenerate values
    true_vals_ix = np.where(y_true >= cutoff_true)[0]
    pred_vals_ix = np.where(y_pred >= cutoff_pred)[0]
    min_index = max(true_vals_ix.min(), pred_vals_ix.min())
        
    mae_95 = np.abs(y_true[min_index:] - y_pred[min_index:]).mean()
    
    return mae_95

def get_spread_error_data(n_samples: int, observation_array: np.ndarray, ensemble_array: np.ndarray,
                          upper_limit: int=None, quantile_step_size: float=0.01):
    """
    Calculates data for plotting a binned spread-error plot

    Args:
        n_samples (int): Number of samples to use from the dataset
        observation_array (np.ndarray): array of observations
        ensemble_array (np.ndarray): array of ensemble predictions
        quantile_step_size (float, optional): step size of quantiles for binning, defaults to 0.01
    Returns:
        list(tuple): List of tuples, where each tuple is (average variance in bin, mse of corresponding observation points relative to ensemble mean)
    """
    (n_sample_total, _, _) = observation_array.shape

    # Sample the data
    ensmean_array = np.mean(ensemble_array, axis=-1)
    sample_indexes = np.random.choice(n_sample_total, n_samples)
    ensemble_array = ensemble_array[sample_indexes, :,:,:].copy()
    ensmean_array = ensmean_array[sample_indexes, :,:]

    # cap data at large value
    if upper_limit is not None:
        ensemble_array = np.clip(ensemble_array, 0, upper_limit)
        ensmean_array = np.clip(ensmean_array, 0, upper_limit)

    observation_array = observation_array[sample_indexes, :, :].copy()

    # First calculate the ensemble variances for each sample point
    sample_variances = np.var(ensemble_array, axis=-1)

    # find percentiles of the variances
    variance_boundaries = np.quantile(sample_variances, np.arange(0, 1, quantile_step_size))
    
    binned_variances = np.digitize(sample_variances, variance_boundaries, right=False)

    # Calculate bin centres
    variance_bin_centres = [0.5*(variance_boundaries[n]+variance_boundaries[n+1]) for n in range(len(variance_boundaries) -1)] + [0.5*(variance_boundaries[-1] + sample_variances.max())]

    variance_mse_pairs = []
    for bin_num in tqdm(set(binned_variances.flatten())):
        
        relevant_truth_data = observation_array[binned_variances ==bin_num]
        relevant_ensmean_data = ensmean_array[binned_variances == bin_num]
        
        tmp_mse = np.power(relevant_truth_data - relevant_ensmean_data, 2).mean()
        
        variance_mse_pairs.append((variance_bin_centres[bin_num-1], tmp_mse))
        
    return variance_mse_pairs


##
# The code below is based off the pysteps code, please refer to their license

def get_filtered_array(int_array: np.ndarray, size: int, mode: str='constant'):
    
    if size > 1:
        if mode == 'constant':
            S = uniform_filter(int_array, size=size, mode="constant", cval=0.0)
        else:
            S = uniform_filter(int_array, size=size, mode=mode)
    else:
        S = int_array.copy()
        
    return S
    
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
        
        S_o = get_filtered_array(int_array=I_o, mode=mode, size=scale)
            
        for ii in range(X_f.shape[-1]):
            
            S_f = get_filtered_array(int_array=I_f[:, :, ii], mode=mode, size=scale)
                   
            sum_obs_sq += np.sum(S_o ** 2)
            sum_fct_obs += np.sum(S_f * S_o)
            sum_fct_sq += np.sum(S_f ** 2)
    
    numer =  2.0 * sum_fct_obs
    denom = sum_fct_sq + sum_obs_sq

    return numer / denom
    

def get_metric_by_hour(metric_fn, obs_array, fcst_array, hours, bin_width=1):
    
    hour_bin_edges = np.arange(0, 24, bin_width)

    digitized_hours = np.digitize(hours, bins=hour_bin_edges)

    metric_by_hour = {}
    for hour in range(24):
        digitized_hour = np.digitize(hour, bins=hour_bin_edges)
        hour_indexes = np.where(np.array(digitized_hours) == digitized_hour)[0]
        
        metric_by_hour[digitized_hour] = metric_fn(obs_array[hour_indexes,...], fcst_array[hour_indexes,...])
    return metric_by_hour, hour_bin_edges