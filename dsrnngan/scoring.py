import numpy as np
from scipy.stats import pearsonr


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

# def FSS(truth_array, forecast_array):
    
    