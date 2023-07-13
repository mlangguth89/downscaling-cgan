import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter
from sklearn.metrics import confusion_matrix

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

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float):

    y_true_int = (y_true > threshold).astype(np.int0).flatten()
    y_pred_int = (y_pred > threshold).astype(np.int0).flatten()
    
    # Note that the conf mat rows indicate true label, and columns indicate predicted label
    conf_mat = confusion_matrix(y_true=y_true_int, y_pred=y_pred_int, labels=[0,1])
    
    return conf_mat
    
def mae(x, y):
    return np.mean(np.abs(x - y))

def mse(x, y):
    return np.mean(np.square(x - y))

def rmse(x, y):
    return np.sqrt(mse(x, y))

def recall(c):
    return c[1][1] / (c[1][0] + c[1][1])

def precision(c):
    return c[1][1] / (c[0][1] + c[1][1])

def f1(c):
    r = recall(c)
    p = precision(c)
    
    f1 = 2*p*r / (p+r)
    
    return f1



def _pierce_skill_score(c):
    return hit_rate(c) - false_alarm_rate(c)
    
def get_contingency_table_values(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, mask: np.ndarray=None):
    
    
    if mask is not None:
        shape = y_true.shape
        if len(shape) == 3 and len(mask.shape) == 2:
            mask = np.stack([mask]*shape[0], axis=0)
        
        y_true = y_true[mask].flatten()
        y_pred = y_pred[mask].flatten()
    else:     
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
    y_true_int = (y_true > threshold).astype(np.int32)
    not_y_true_int = (y_true <= threshold).astype(np.int32)
    y_pred_int = (y_pred > threshold).astype(np.int32)
    not_y_pred_int = (y_pred <= threshold).astype(np.int32)
    
    hits = np.multiply(y_true_int, y_pred_int).sum()
    misses = np.multiply(y_true_int, not_y_pred_int).sum()
    false_alarms = np.multiply(not_y_true_int, y_pred_int).sum()
    correct_negatives = np.multiply(not_y_true_int, not_y_pred_int).sum()
    
    output_dict = {
            'hits': hits,
            'misses': misses,
            'false_alarms': false_alarms,
            'hits_random': (hits + misses)*(hits + false_alarms) / y_true.size,
            'correct_negatives': correct_negatives
            }
    
    return output_dict

def critical_success_index(y_true: np.ndarray, y_pred: np.ndarray, threshold):
    
    vals = get_contingency_table_values(y_true, y_pred, threshold)

    if vals['hits'] + vals['misses'] + vals['false_alarms'] == 0:
        csi = np.nan
    else:
        csi = vals['hits'] / (vals['hits'] + vals['misses'] + vals['false_alarms'])

    return csi

def equitable_threat_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, mask=None):
    
    vals = get_contingency_table_values(y_true, y_pred, threshold=threshold, mask=mask)

    if vals['hits'] + vals['misses'] + vals['false_alarms'] + vals['hits_random'] == 0:
        ets = np.nan
    else:
        ets = (vals['hits'] - vals['hits_random'] ) / (vals['hits'] + vals['misses'] + vals['false_alarms'] - vals['hits_random'])

    return ets

def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, mask: np.ndarray=None):
    
    vals = get_contingency_table_values(y_true, y_pred, threshold, mask=mask)
    
    return vals['hits'] / (vals['hits'] + vals['misses'])

def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, mask=None):
    vals = get_contingency_table_values(y_true, y_pred, threshold, mask=mask)
    
    return vals['false_alarms'] / (vals['hits'] + vals['false_alarms'])

def pierce_skill_score(y_true: np.ndarray, y_pred: np.ndarray, threshold):
    
    conf_mat = calculate_confusion_matrix(y_true=y_true, y_pred=y_pred, threshold=threshold)
    pss_result = _pierce_skill_score(conf_mat)

    return pss_result

def get_metric_by_grid_cell(metric_fn, y_true, y_pred, **kwargs):
    
    (_, W, H) = y_true.shape
    metric_by_grid_cell = np.empty((W,H))
    metric_by_grid_cell[:,:] = np.nan

    for w in range(W):
        for h in range(H):
            
            metric_by_grid_cell[w,h] = metric_fn(y_true=y_true[:,w,h], y_pred=y_pred[:,w,h], **kwargs)
    
    return metric_by_grid_cell

def get_skill_score_results(
                skill_score_function,
                data_dict: dict, obs_array: np.ndarray,
             hours,
             hourly_thresholds: list
             ):
    
    csi_results = []
        
    for threshold in hourly_thresholds:
        
        tmp_results_dict = {'threshold': threshold}
        for k, v in data_dict.items():
            tmp_results_dict[k] = skill_score_function(obs_array, v, threshold=threshold)
            
            metric_fn = lambda x,y: skill_score_function(x, y, threshold=threshold)
            
            tmp_results_dict[k + '_hourly'] = get_metric_by_hour(metric_fn=metric_fn, 
                                                                         obs_array=obs_array, 
                                                                         fcst_array=v,
                                                                         hours=hours,
                                                                         bin_width=1)
        
        csi_results.append(tmp_results_dict)
        
    return csi_results


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
    (n_sample_total, _, _, ensemble_size) = ensemble_array.shape
    ensmean_array = np.mean(ensemble_array, axis=-1)

    if n_samples < n_sample_total:
        # Sample the data
        sample_indexes = np.random.choice(n_sample_total, n_samples)
        ensemble_array = ensemble_array[sample_indexes, :,:,:].copy()
        ensmean_array = ensmean_array[sample_indexes, :,:]
        observation_array = observation_array[sample_indexes, :, :].copy()
    else:
        ensemble_array = ensemble_array.copy()
        observation_array = observation_array.copy()
        
    # cap data at large value
    if upper_limit is not None:
        ensemble_array = np.clip(ensemble_array, 0, upper_limit)
        ensmean_array = np.clip(ensmean_array, 0, upper_limit)

    # First calculate the spread values of the ensemble
    spreads = np.var(ensemble_array, axis=-1)
    # Apply correction factor; see e.g. Leutbecker and Palmer, 2008
    spreads = ((ensemble_size+1) / (ensemble_size-1)) * spreads

    # find percentiles of the variances
    spread_boundaries = np.quantile(spreads, np.arange(0, 1, quantile_step_size))
    binned_spreads = np.digitize(spreads, spread_boundaries, right=False)

    # Calculate bin centres
    spread_bin_centres = [0.5*(spread_boundaries[n]+spread_boundaries[n+1]) for n in range(len(spread_boundaries) -1)] + [0.5*(spread_boundaries[-1] + spreads.max())]

    variance_mse_pairs = []
    for bin_num in tqdm(set(binned_spreads.flatten())):
        
        relevant_truth_data = observation_array[binned_spreads ==bin_num]
        relevant_ensmean_data = ensmean_array[binned_spreads == bin_num]
        
        tmp_mse = np.power(relevant_truth_data - relevant_ensmean_data, 2).mean()
        
        variance_mse_pairs.append((spread_bin_centres[bin_num-1], tmp_mse))
        
    return variance_mse_pairs

def get_metric_by_hour(metric_fn, obs_array, fcst_array, hours, bin_width=1):
    
    hour_bin_edges = np.arange(0, 24, bin_width)

    digitized_hours = np.digitize(hours, bins=hour_bin_edges)

    metric_by_hour = {}
    for hour in range(24):
        digitized_hour = np.digitize(hour, bins=hour_bin_edges)
        hour_indexes = np.where(np.array(digitized_hours) == digitized_hour)[0]
  
        if len(hour_indexes) == 0:
            metric_by_hour[digitized_hour] = np.nan
        else:
            metric_by_hour[digitized_hour] = metric_fn(obs_array[hour_indexes,...], fcst_array[hour_indexes,...])

    return metric_by_hour, hour_bin_edges
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
    

