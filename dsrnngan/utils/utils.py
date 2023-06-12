import os
import sys
from pathlib import Path
import hashlib
import json
import yaml
import re
import random
from glob import glob
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from calendar import monthrange
from typing import Iterable, Tuple, Callable, List
from timezonefinder import TimezoneFinder
from dateutil import tz

tz_finder = TimezoneFinder()
from_zone = tz.gettz('UTC')

def hash_dict(params: dict):
    h = hashlib.shake_256()
    h.update(json.dumps(params, sort_keys=True).encode('utf-8'))
    return h.hexdigest(8)


def load_yaml_file(fpath: str):  
    with open(fpath, 'r') as f:
        data = yaml.safe_load(f)
        
    return data

def write_to_yaml(fpath: str, data: dict):
    
    with open(fpath, 'w+') as ofh:
        yaml.dump(data, ofh, default_flow_style=False)
        
def date_range_from_year_month_range(year_month_range):
    
    start_date = datetime.datetime(year=int(year_month_range[0][:4]), month=int(year_month_range[0][4:6]), day=1)
    
    end_year = int(year_month_range[-1][:4])
    end_month = int(year_month_range[-1][4:6])
    end_date = datetime.datetime(year=end_year, 
                                 month=end_month, 
                                 day=monthrange(end_year, end_month)[1])
    
    return [item.date() for item in pd.date_range(start=start_date, end=end_date)]

def get_checkpoint_model_numbers(log_folder: str) -> List[int]:
    """Get list of iteration points that have been saved as checkpoints

    Args:
        log_folder (str): Base folder; must contain a 'models' directory
        
    Returns:
        list[int]: list of interger model numbers
    """
    models_glob = str(Path(log_folder).parents[0] / 'models/*.h5')

    fps = glob(models_glob)
    
    if fps:
        model_numbers = [int(re.search(r'([0-9]+).h5', fp).groups()[0]) for fp in fps]
    else:
        return []
    
    return model_numbers
    

def get_best_model_number(log_folder: str, metric_column_name: str='CRPS_no_pooling'):
    """
    Get model number that has lowest value according to metric specified (defaults to CRPS)

    Args:
        log_folder (str): Path to evaluation results
        metric_column_name (str): Name of column to optimise on. Defaults to CRPS_no_pooling 

    Returns:
        int: Model number
    """
    df = pd.read_csv(os.path.join(log_folder, 'eval_validation.csv'))

    # find model_number with lowest CRPS
    min_crps = df['CRPS_no_pooling'].min()
    model_number = df[df.CRPS_no_pooling == min_crps]['N'].values[0]
    
    return model_number


def get_valid_quantiles(data_size: int, min_data_points_per_quantile: int, raw_quantile_locations: list) -> list:
    """Returns a list of quantiles q such that (1-q)*input_data.size > min_data_points_per_quantile

    Args:
        data_size (int): Total number of data points
        min_data_points_per_quantile (int): Minimum number of data points per quantile
        raw_quantile_locations (list): The raw quantile locations, to be truncated

    Returns:
        list: quantile locations with necessary quantiles removed
    """

    quantile_locs = [val for val in raw_quantile_locations[:-1] if (1-val)*data_size > min_data_points_per_quantile]
    
    return quantile_locs

def resample_array(input_array: np.ndarray, time_resample: bool=False) -> np.ndarray:
    """Resample a single array with replacement

    Args:
        input_array (np.ndarray): array to resample
        time_resample (bool, optional): If True then will only resample along the 0 axis. Defaults to False.

    Returns:
        np.ndarray: resampled array
    """
    shape = input_array.shape
    if time_resample:

        idx = np.random.randint(0, shape[0], shape[0])

        return input_array[idx, ...]
    else:
        
        return np.random.choice(input_array.flatten(), size=shape, replace=True) 

    

def resample_paired_arrays(fcst_array: np.ndarray, 
                           obs_array: np.ndarray, time_resample: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function for resampling paired arrays (e.g. for bootstrapping)

    Args:
        fcst_array (np.ndarray): forecast array
        obs_array (np.ndarray): observation array
        time_resample (bool, optional): If True, will only resample along the 0 axis. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: resampled forecast and observation arrays
    """
    shape = fcst_array.shape
    
    if time_resample:
        
        idx = np.random.randint(0, shape[0], shape[0])

        resampled_fcst = fcst_array[idx, ...]
        resampled_obs = obs_array[idx, ...]
    else:
        idx = np.random.randint(0,fcst_array.size, shape)

        resampled_fcst = np.take(fcst_array, idx)
        resampled_obs = np.take(obs_array, idx)

    return resampled_fcst, resampled_obs

def bootstrap_summary_statistic(statistic_func: Callable[..., float],
                                input_array: np.ndarray,
                                n_bootstrap_samples: int,
                                time_resample: bool=False,
                                **kwargs):
    
    
    tmp_results = []
    for _ in tqdm(range(n_bootstrap_samples)):
        resampled_array = resample_array(input_array, time_resample)
        tmp_results.append(statistic_func(resampled_array, **kwargs))
        
    tmp_results = np.stack(tmp_results, axis=0)
    # calculate mean and standard deviation
    output_dict = {'mean': tmp_results.mean(axis=0), 'std': tmp_results.std(axis=0)}
    
    return output_dict

def subsample_summary_statistic(statistic_func: Callable[..., float],
                                input_array: np.ndarray,
                                n_subsamples: int,
                                axis: int=0,
                                **kwargs):
    
        
    split_arrays = np.array_split(input_array, n_subsamples, axis=axis)

    tmp_results = []

    for arr in tqdm(split_arrays):
        
        tmp_results.append(statistic_func(arr, **kwargs))
            
    tmp_results = np.stack(tmp_results, axis=0)
    
    # calculate mean and standard deviation
    output_dict = {'mean': tmp_results.mean(axis=0), 'std': tmp_results.std(axis=0)}
    
    return output_dict 

    
def bootstrap_metric_function(metric_func: Callable[..., float], 
                              fcst_array: np.ndarray, 
                              obs_array: np.ndarray, 
                              n_bootstrap_samples: int, 
                              time_resample: bool=False,
                              **kwargs) -> dict:
    """Wrapper to run bootstrap on metric function
    The function must accept a forecast array and observation array, plus keyword arguments

    Args:
        metric_func (Callable[..., float]): Metric function
        fcst_array (np.ndarray): Forecast array
        obs_array (np.ndarray): Observation array
        n_bootstrap_samples (int): Number of bootstrap samples to generate

    Returns:
        dict: Summary statistics of metric
    """
    tmp_results = []
    for _ in tqdm(range(n_bootstrap_samples)):
        resampled_fcst_array, resampled_obs_array = resample_paired_arrays(fcst_array, obs_array, time_resample)
        tmp_results.append(metric_func(resampled_obs_array, resampled_fcst_array, **kwargs))

    # calculate mean and standard deviation
    output_dict = {'mean': np.mean(tmp_results), 'std': np.std(tmp_results)}
    
    return output_dict


def get_local_datetime(utc_datetime: datetime.datetime, longitude: float, latitude: float) -> datetime.datetime:
    """Get datetime at locality defined by lat,long, from UTC datetime

    Args:
        utc_datetime (datetime.datetime): UTC datetime
        longitude (float): longitude
        latitude (float): latitude

    Returns:
        datetime.datetime: Datetime in local time
    """
    utc_datetime.replace(tzinfo=from_zone)
    
    timezone = tz_finder.timezone_at(lng=longitude, lat=latitude)
    to_zone = tz.gettz(timezone)

    local_datetime = utc_datetime.astimezone(to_zone)
    
    return local_datetime

def get_local_hour(hour: int, longitude: float, latitude: float):
    """Convert hour to hour in locality.
    
    This is not as precise as get_local_datetime, as it won't contain information about e.g. BST. 
    But it is useful when the date is not known but the hour is

    Args:
        hour (int): UTC hour
        longitude (float): longitude
        latitude (float): latitude
    Returns:
        int: approximate hour in local time
    """
    utc_datetime = datetime.datetime(year=2000, month=1, day=1, hour=hour)
    utc_datetime.replace(tzinfo=from_zone)
    
    timezone = tz_finder.timezone_at(lng=longitude, lat=latitude)
    to_zone = tz.gettz(timezone)

    local_hour = utc_datetime.astimezone(to_zone).hour
    
    return local_hour