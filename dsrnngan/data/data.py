""" Functions for handling data loading and saving. """
from genericpath import isfile
import os
import re
from pathlib import Path
import pickle
import logging
import netCDF4
from calendar import monthrange
from tqdm import tqdm
from glob import glob
from typing import Iterable, Union, List, Dict, Tuple

from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import xarray as xr
from argparse import ArgumentParser

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

from dsrnngan.utils import read_config

data_config = read_config.read_data_config()
DATA_PATHS = read_config.get_data_paths()

# Raw paths for data that has not been regridded or cut down to correct lat/lon range

IMERG_PATH = DATA_PATHS["GENERAL"].get("IMERG", '')
NIMROD_PATH = DATA_PATHS["GENERAL"].get("NIMROD", '')
ERA5_PATH = DATA_PATHS["GENERAL"].get("ERA5", '')
IFS_PATH = DATA_PATHS["GENERAL"].get("IFS", '')
OROGRAPHY_PATH = DATA_PATHS["GENERAL"].get("OROGRAPHY")
LSM_PATH = DATA_PATHS["GENERAL"].get("LSM")
CONSTANTS_PATH = DATA_PATHS["GENERAL"].get("CONSTANTS")
### For processing data on JSC cluster ###

# monthly files of CERRA, ERA5 and IMERG
CERRA_MONTHLY_PATH = DATA_PATHS["GENERAL"].get("CERRA_MONTHLY", '')
ERA5_MONTHLY_PATH = DATA_PATHS["GENERAL"].get("ERA5_MONTHLY", '')
IMERG_MONTHLY_PATH = DATA_PATHS["GENERAL"].get("IMERG_MONTHLY", '')
NORMALISATION_STRATEGY = data_config.input_normalisation_strategy

### For processing data on JSC cluster ###

FIELD_TO_HEADER_LOOKUP_IFS = {'tp': 'sfc',
                              'cp': 'sfc',
                              'sp': 'sfc',
                              'tisr': 'sfc',
                              'cape': 'sfc',
                              'tclw': 'sfc',
                              'tcwv': 'sfc',
                              'u700': 'winds',
                              'v700': 'winds',
                              #   'cdir': 'missing',
                              #   'tcrw': 'missing'
                              }
VAR_LOOKUP_ERA5_JSC = {"t": {"longname": "temperature"}, 
                       "u": {"longname": "velocity_u"},
                       "v": {"longname": "velocity_v"},
                       "q": {"longname": "specific_humidity"},
                       "cape": {"longname": "cape"},
                       "sp": {"longname": "surface_pressure"},
                       "cp": {"longname": "convective_precip"},
                       "tisr": {"longname": "toa_solar_rad"},
                       "tclw": {"longname": "total_cloud_liquid_water"},
                       "tcwv":{"longname": "vertically_int_water_vapour"},
                       "tp": {"longname": "total_precip"}}

# 'cin', Left out for the moment as contains a lot of nulls
all_ifs_fields = ['2t', 'cape',  'cp', 'r200', 'r700', 'r950', 
                  'sp', 't200', 't700', 'tclw', 'tcwv', 'tisr', 'tp', 
                  'u200', 'u700', 'v200', 'v700', 'w200', 'w500', 'w700', 'cin']
input_fields = data_config.input_fields
constant_fields = data_config.constant_fields
all_fcst_hours = np.array(range(24))

# TODO: change this to behave like the IFS data load (to allow other vals of v, u etc)
# TODO: move to a config
VAR_LOOKUP_ERA5 = {'tp': {'folder': 'total_precipitation', 'suffix': 'day',
                               'negative_vals': False},
                        'q': {'folder': 'shum', 'subfolder': 'shum700', 'suffix': 'day_1deg',
                              'normalisation': 'standardise', 'negative_vals': False},
                        't': {'folder': 'ta', 'subfolder': 'tas700', 'suffix': 'day_1deg',
                              'normalisation': 'minmax', 'negative_vals': False},
                        'u': {'folder': 'u', 'subfolder': 'u700', 'suffix': 'day_1deg',
                              'normalisation': 'minmax'},
                        'v': {'folder': 'v', 'subfolder': 'v700', 'suffix': 'day_1deg',
                              'normalisation': 'minmax'}}

IFS_NORMALISATION_STRATEGY = data_config.input_normalisation_strategy

VAR_LOOKUP_IFS = {field: IFS_NORMALISATION_STRATEGY[re.sub(r'([0-9]*[a-z]+)[0-9]*', r'\1', field)] 
                  for field in all_ifs_fields}

all_era5_fields = list(VAR_LOOKUP_ERA5.keys())

input_field_lookup = {'ifs': all_ifs_fields, 'era5': all_era5_fields}

NORMALISATION_YEAR = data_config.normalisation_year

DEFAULT_LATITUDE_RANGE=np.arange(data_config.min_latitude, data_config.max_latitude + data_config.latitude_step_size, data_config.latitude_step_size)
DEFAULT_LONGITUDE_RANGE=np.arange(data_config.min_longitude, data_config.max_longitude + data_config.longitude_step_size, data_config.longitude_step_size)

char_integer_re = re.compile(r'[a-zA-Z]*([0-9]+)')

def denormalise(x: np.float64, normalisation_type: str):
    """
    Undo normalisation

    Args:
        x (np.float64): normalised float
        normalisation_type (str): type of normalisation
    Returns:
        np.float64: denormalised float
    """
        
    if normalisation_type == 'log':
        return 10 ** x - 1
    elif normalisation_type == 'sqrt':
        return np.power(x,2)
    else:
        raise NotImplementedError(f'normalisation type {normalisation_type} not recognised')
    

def normalise_precipitation(data_array: xr.DataArray,
                            normalisation_type: str):
    
    if normalisation_type == 'log':
        return np.log10(1 + data_array)
    elif normalisation_type == 'sqrt':
        return np.sqrt(data_array)
    else:
        raise NotImplementedError(f'normalisation type {normalisation_type} not recognised')

def log_plus_1(data_array: xr.DataArray):
    """
    Transform data according to x -> log10(1 + x)

    Args:
        data_array (xr.DataArray): Data array to transform

    Returns:
        xr.DataArray: Transformed data array
    """
    return np.log10(1 + data_array)

def infer_lat_lon_names(ds: xr.Dataset):
    """
    Infer names of latitude / longitude coordinates from the dataset

    Args:
        ds (xr.Dataset): dataset (containing one latitude coordinate 
        and one longitude coordinate)

    Returns:
        tuple: (lat name, lon name)
    """
    
    coord_names = list(ds.coords)
    lat_var_name = [item for item in coord_names if item.startswith('lat')]
    lon_var_name = [item for item in coord_names if item.startswith('lon')]
    
    assert (len(lat_var_name) == 1) and (len(lon_var_name) == 1), IndexError('Cannot infer latitude and longitude names from this dataset')

    return lat_var_name[0], lon_var_name[0]

def order_coordinates(ds: xr.Dataset):
    """
    Order coordinates of dataset -> (time, lat, lon)

    Args:
        ds (xr.Dataset): Dataset to order coordinates of

    Returns:
        xr.Dataset: Ordered dataset
    """
    
    lat_var_name, lon_var_name = infer_lat_lon_names(ds)
    
    if 'time' in list(ds.dims):
        return ds.transpose('time', lat_var_name, lon_var_name)
    else:
        return ds.transpose(lat_var_name, lon_var_name)

def make_dataset_consistent(ds: xr.Dataset):
    """
    Ensure longitude and latitude are ordered in ascending order

    Args:
        ds (xr.Dataset): dataset

    Returns:
        xr.Dataset: Dataset with data reordered
    """
    
    latitude_var, longitude_var = infer_lat_lon_names(ds)
    ds = ds.sortby(latitude_var, ascending=True)
    ds = ds.sortby(longitude_var, ascending=True)
    
    return ds

def get_obs_dates(date_range: list,  hour: int, obs_data_source: str, 
                  data_paths=DATA_PATHS,
                  ):
    """
    Get dates for which there is observational data available
    Args:
        date_range (list): list of candidate dates
        obs_data_source (str): Name of data source
        data_paths (dict, optional): Dict containing data paths. Defaults to DATA_PATHS.

    Returns:
        list: list of dates
    """

    obs_dates = set([item for item in date_range if file_exists(data_source=obs_data_source, year=item.year,
                                                    month=item.month, day=item.day,
                                                    data_paths=data_paths, hour=hour)])
    
    return sorted(obs_dates)

def get_dates(years, obs_data_source: str, 
              fcst_data_source: str,
              data_paths=DATA_PATHS):
    """
    Get dates for which there is observational and forecast data

    Args:
        years (int or list): list of years, or single integer year
        obs_data_source (str): name of observational data source
        fcst_data_source (str): name of forecast data source
        data_paths (dict, optional): Dict containing paths to data. Defaults to DATA_PATHS.

    Returns:
        list: list of valid date strings
    """
    if isinstance(years, int):
        years = [years]

    years = sorted(years)
    
    date_range = pd.date_range(start=date(years[0], 1, 1), end=date(years[-1], 12, 31))
    date_range = [item.date() for item in date_range]
    
    obs_dates = set([item for item in date_range if file_exists(data_source=obs_data_source, year=item.year,
                                                    month=item.month, day=item.day,
                                                    data_paths=data_paths)])
    fcst_dates = set([item for item in date_range if file_exists(data_source=fcst_data_source, year=item.year,
                                                    month=item.month, day=item.day,
                                                    data_paths=data_paths)])
    dates = sorted(obs_dates.intersection(fcst_dates))
        
    return [item.strftime('%Y%m%d') for item in dates]


def file_exists(data_source: str, year: int,
                month: int, day:int, hour='random',
                data_paths=DATA_PATHS):
    """
    Check if file exists

    Args:
        data_source (str): Name of data source
        year (int): year
        month (int): month
        day (int): day
        data_paths (dict, optional): dict of data paths. Defaults to DATA_PATHS.

    Returns:
        bool: True if file exists
    """
    
    if hour == 'random':
        hour = 0
        
    data_path = data_paths["GENERAL"].get(data_source.upper()) 
    
    if not data_path:
        raise ValueError(f'No path specified for {data_source} in data_paths')

    if data_source == 'nimrod':
        glob_str = os.path.join(data_path, f"{year}/*.nc")
        if len(glob(glob_str)) > 0:
            return True
        
    elif data_source == 'imerg':
        for file_type in ['.HDF5', '.nc']:
            fps = get_imerg_filepaths(year, month, day, hour, file_ending=file_type, imerg_data_dir=data_path)
            if os.path.isfile(fps[0]):
                return True

    elif data_source == 'ifs':
        fp = get_ifs_filepath('tp', loaddate=datetime(year, month, day), 
                                  loadtime='12', fcst_dir=data_path)
        if os.path.isfile(fp):
            return True

    elif data_source == 'era5':
        # These are just meaningless dates to get the filepath
        era5_fp = get_era5_path('tp', year=year, month=month, era_data_dir=data_path)
        glob_str = era5_fp
        if len(glob(glob_str)) > 0:
            return True
    else:
        raise ValueError(f'Unrecognised data source: {data_source}')
    
    return False
                
def filter_by_lat_lon(ds: xr.Dataset, 
                      lon_range: list, 
                      lat_range: list):
    """
    Filter dataset by latitude / longitude range

    Args:
        ds (xr.Dataset): Dataset to filter
        lon_range (list): min and max longitude values
        lat_range (list): min and max latitude values
        lon_var_name (str, optional): Name of longitude variable in dataset. Defaults to 'lon'.
        lat_var_name (str, optional): Name of latitude variable in dataset. Defaults to 'lat'.

    Returns:
        xr.Dataset: Filtered dataset
    """
    lat_var_name, lon_var_name = infer_lat_lon_names(ds)
    
    all_lat_vals = ds[lat_var_name].values
    all_lon_vals = ds[lon_var_name].values
    
    overlapping_lat_vals = all_lat_vals[all_lat_vals >= min(lat_range)]
    overlapping_lat_vals = overlapping_lat_vals[overlapping_lat_vals <= max(lat_range)]
    
    overlapping_lon_vals = all_lon_vals[all_lon_vals >= min(lon_range)]
    overlapping_lon_vals = overlapping_lon_vals[overlapping_lon_vals <= max(lon_range)]
        
    ds = ds.sel({lat_var_name: overlapping_lat_vals})
    ds = ds.sel({lon_var_name: overlapping_lon_vals})
   
    return ds


def interpolate_dataset_on_lat_lon(ds: xr.Dataset, 
                                   latitude_vals: list, 
                                   longitude_vals: list,
                                   interp_method:str ='bilinear'):
    """
    Interpolate dataset to new lat/lon values. Requires xesmf-package.

    Args:
        ds (xr.Dataset): Datast to interpolate
        latitude_vals (list): list of latitude values to interpolate to
        longitude_vals (list): list of longitude values to interpolate to
        interp_method (str, optional): name of interpolation method. Defaults to 'bilinear'._

    Returns:
        xr,Dataset: interpolated dataset
    """
    import xesmf as xe
    
    lat_var_name, lon_var_name = infer_lat_lon_names(ds)
    
    min_lat = min(latitude_vals)
    max_lat = max(latitude_vals)

    min_lon = min(longitude_vals)
    max_lon = max(longitude_vals)
    
    existing_latitude_vals = ds[lat_var_name].values
    existing_longitude_vals = ds[lon_var_name].values
    
    if all([item in existing_latitude_vals for item in latitude_vals]) and all([item in existing_longitude_vals for item in longitude_vals]):
        ds = ds.sel({lat_var_name: latitude_vals}).sel({lon_var_name: longitude_vals})
        return ds
    
    # Filter to correct lat/lon range (faster this way)
    # Add a buffer around it for interpolation
    ds = filter_by_lat_lon(ds, [min_lon - 2, max_lon+2], [min_lat - 2, max_lat +2])

    # check enough data to interpolate
    min_actual_lat = min(ds.coords[lat_var_name].values)
    max_actual_lat = max(ds.coords[lat_var_name].values)
    min_actual_lon = min(ds.coords[lon_var_name].values)
    max_actual_lon = max(ds.coords[lon_var_name].values)

    if not ((min_actual_lat < min_lat < max_actual_lat) and (min_actual_lon < min_lon < max_actual_lon)
            and (min_actual_lat < max_lat < max_actual_lat) and (min_actual_lon < max_lon < max_actual_lon)):
        raise ValueError('Larger buffer area needed to ensure good interpolation')

    ds_out = xr.Dataset(
        {
            lat_var_name: ([lat_var_name], latitude_vals),
            lon_var_name: ([lon_var_name], longitude_vals),
        }
    )

    # Use conservative to preserve global precipitation
    regridder = xe.Regridder(ds, ds_out, interp_method)
    regridded_ds = ds.copy()
    
    # Make float vars C-contiguous (to avoid warning message and potentially improve performance)
    for var in list(regridded_ds.data_vars):
        if regridded_ds[var].values.dtype.kind == 'f':
            regridded_ds[var].values = np.ascontiguousarray(regridded_ds[var].values)
            
    regridded_ds = regridder(regridded_ds)

    return regridded_ds


def load_hdf5_file(fp: str, group_name:str ='Grid'):
    """
    Load HDF5 file

    Args:
        fp (str): file path
        group_name (str, optional): name of group to load. Defaults to 'Grid'.

    Returns:
        xr.Dataset: Dataset
    """
    ncf = netCDF4.Dataset(fp, diskless=True, persist=False)
    nch = ncf.groups.get(group_name)
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nch))

    return ds


def preprocess(variable: str, 
               ds: xr.Dataset, 
               normalisation_strategy: dict, 
               stats_dict: dict=None):
    """
    Preprocess data

    Args:
        variable (str): name of variable.
        ds (xr.Dataset): dataset containing data for named variable
        normalisation_strategy (dict): dict containing normalisation strategy for each variable
        stats_dict (dict, optional): dict of values required for normalisation. Defaults to None.

    Returns:
        xr.Dataset: processed dataset
    """
    
    
    var_name = list(ds.data_vars)[0]

    normalisation_type = normalisation_strategy[variable].get('normalisation')

    if normalisation_type:

        if normalisation_type == 'standardise':
            ds[var_name] = (ds[var_name] - stats_dict['mean']) / stats_dict['std']

        elif normalisation_type == 'minmax':
            ds[var_name] = (ds[var_name] - stats_dict['min']) / (stats_dict['max'] - stats_dict['min'])
        
        elif normalisation_type == 'log':
            ds[var_name] = log_plus_1(ds[var_name])
            
        elif normalisation_type == 'max':
            ds[var_name] = ds[var_name] / stats_dict['max']
            
        elif normalisation_type == 'sqrt':
            ds[var_name] = normalise_precipitation(ds[var_name], 'sqrt')

        else:
            raise ValueError(f'Unrecognised normalisation type for variable {var_name}')

    return ds


def load_observational_data(data_source: str, *args, **kwargs):
    """
    Function to pick between various different sources of observational data

    Args:
        data_source (str): anme of data source

    Returns:
        np.ndarray: array of data
    """
    if data_source.lower() == 'imerg':
        return load_imerg(*args, **kwargs)
    elif data_source.lower() == 'imerg_monthly':
        return load_imerg_from_ds(*args, **kwargs)
    else:
        raise NotImplementedError(f'Data source {data_source} not implemented yet')


def load_orography(filepath: str=OROGRAPHY_PATH, 
                   latitude_vals: list=None, 
                   longitude_vals: list=None,
                   varname_oro: str = "orog",
                   interpolate: bool=False):
    """
    Load orography values

    Args:
        filepath (str, optional): path to orography data. Defaults to OROGRAPHY_PATH.
        latitude_vals (list, optional): list of latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): list of longitude values to filter/interpolate to. Defaults to None.
        varname_oro (str, optional): name of orography variable. Defaults to "orog".
        interpolate (bool, optional): Whether or not to interpolate. Defaults to True.

    Returns:
        np.ndarray: orography data array
    """
    ds = xr.load_dataset(filepath)
    
    # Note that this assumes the orography is somewhat filtered already 
    # If it is worldwide orography then normalised values will probably be too small!
    max_val = ds[varname_oro].values.max()
       
    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method='bilinear')
        else:
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, lat_range=latitude_vals)

    ds = make_dataset_consistent(ds)

    # Normalise and clip below to remove spectral artefacts
    h_vals = ds[varname_oro].values[0, :, :]
    h_vals[h_vals < 5] = 5.0
    h_vals = h_vals / max_val

    ds.close()

    return h_vals

def load_land_sea_mask(filepath=LSM_PATH, 
                       latitude_vals=None, 
                       longitude_vals=None,
                       varname_lsm:str = "lsm",
                       interpolate=False):
    """
    Load land-sea mask values

    Args:
        filepath (str, optional): path to land-sea masj data. Defaults to LSM_PATH.
        latitude_vals (list, optional): list of latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): list of longitude values to filter/interpolate to. Defaults to None.
        varname_lsm (str, optional): name of land-sea mask variable. Defaults to "lsm".
        interpolate (bool, optional): Whether or not to interpolate. Defaults to True.

    Returns:
        np.ndarray: land-sea mask data array
    """
    ds = xr.load_dataset(filepath)
    
    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method='bilinear')
        else:
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, lat_range=latitude_vals)
            
    ds = make_dataset_consistent(ds)
    
    lsm = ds[varname_lsm].values[0, :, :]
    
    ds.close()
    
    return lsm


def load_hires_constants(
                         fields: Iterable,
                         data_paths: dict,
                         batch_size: int=1,
                         latitude_vals: list=None, 
                         longitude_vals: list=None):
    """
    Load hi resolution constants (land sea mask, orography)
    
    Note; this currently interpolates by default

    Args:
        batch_size (int, optional): Size of batch. Defaults to 1.
        lsm_path (str, optional): path to land sea mask data. Defaults to LSM_PATH.
        oro_path (str, optional): path to orography data. Defaults to OROGRAPHY_PATH.
        latitude_vals (list, optional): list of latitude values to interpolate to. Defaults to None.
        longitude_vals (list, optional): list of longitude values to interpolate to. Defaults to None.

    Returns:
        np.ndarray: array of data
    """
    
    function_lookup = {'lsm': load_land_sea_mask,
                       'orography': load_orography,
                       'lakes': load_land_sea_mask,
                       'sea': load_land_sea_mask}
    
    unrecognised_fields = [f for f in fields if f not in function_lookup]

    if len(unrecognised_fields) > 0:
        raise ValueError(f'Unrecognised constant field names: {unrecognised_fields}')
    
    constant_data = []
    for field in fields:
        tmp_array = function_lookup[field.lower()](filepath=data_paths[field.upper()],
                                                            latitude_vals=latitude_vals,
                                                            longitude_vals=longitude_vals)
        tmp_array = np.expand_dims(tmp_array, axis=0)
        
        constant_data.append(tmp_array)
    
    return np.repeat(np.stack(constant_data, axis=-1), batch_size, axis=0)


### These functions work with IFS / Nimrod.
# TODO: unify the functions that load data from different sources

def load_fcst_radar_batch(batch_dates: Iterable, 
                          fcst_fields: list, 
                          fcst_data_source: str, 
                          obs_data_source: str, 
                          fcstdir_or_ds: Union[str, xr.Dataset],
                          obsdir_or_ds: Union[str, xr.Dataset],
                          normalisation_strategy: dict,
                          latitude_range: Iterable[float]=None,
                          longitude_range: Iterable[float]=None,
                          constants_dir: str=CONSTANTS_PATH,
                          constant_fields: list=None, 
                          hour: Union[str, int] = 0, 
                          normalise_inputs: bool=False,
                          output_normalisation: bool=False):
    batch_x = []
    batch_y = []

    if type(hour) == str:
        if hour == 'random':
            all_fcst_hours = np.array(range(24))
            hours = all_fcst_hours[np.random.randint(24, size=[len(batch_dates)])]
        else:
            assert False, f"Not configured for {hour}"
    elif np.issubdtype(type(hour), np.integer):
        hours = len(batch_dates) * [hour]
    else:
        hours = hour

    for i, date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_fcst_stack(fcst_data_source, fcst_fields, date, h,
                                       latitude_vals=latitude_range, longitude_vals=longitude_range, fcstdir_or_ds=fcstdir_or_ds,
                                       norm=normalise_inputs, constants_dir=constants_dir,
                                       normalisation_strategy=normalisation_strategy))
        
        if obs_data_source is not None:
            batch_y.append(load_observational_data(obs_data_source, date, h, obsdir_or_ds, normalisation_type=output_normalisation,
                                                latitude_vals=latitude_range, longitude_vals=longitude_range))
    if constant_fields is None:
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(batch_size=len(batch_dates), fields=constant_fields)], np.array(batch_y)

def get_ifs_filepath(field: str, loaddate: datetime, 
                     loadtime: int, fcst_dir: str=IFS_PATH):
    """
    Get ifs filepath for time/data combination

    Args:
        field (str): name of field
        loaddate (datetime): load datetime
        loadtime (int): load time
        fcst_dir (str, optional): directory of forecast data. Defaults to IFS_PATH.

    Returns:
        str: filepath
    """
    filename = f"{field}_HRES_1h_EAfrica_{loaddate.strftime('%Y-%m-%d')}_{loadtime}h.nc"

    top_level_folder = re.sub(r'([0-9]*[a-z]+)[0-9]*', r'\1', field)
    subdirectory = True
    
    if top_level_folder == field:
        subdirectory = False
    
    if subdirectory:
        fp = os.path.join(fcst_dir, top_level_folder, field, filename)
    else:
        fp = os.path.join(fcst_dir, top_level_folder, filename)
        
    return fp

def get_ifs_forecast_time(year: int, 
                          month: int, 
                          day:int, hour: int):
    """
    Get the closest IFS forecast load (between 6-18h lead time) to the target
    year/month/day/hour combination

    Args:
        year (int): year
        month (int): month
        day (int): day
        hour (int): hour

    Returns:
        tuple: tuple containing (load date, load time)
    """
    
    time = datetime(year=year, month=month, day=day, hour=hour)
    
    if time.hour < 6:
        loaddate = time - timedelta(days=1)
        loadtime = '12'
    elif 6 <= time.hour < 18:
        loaddate = time
        loadtime = '00'
    elif 18 <= time.hour < 24:
        loaddate = time
        loadtime = '12'
    else:
        raise ValueError("Not acceptable time")
    
    return loaddate, loadtime
    
def load_ifs_raw(field: str, 
                 year: int, 
                 month: int,
                 day: int, hour: int, ifs_data_dir: str=IFS_PATH,
                 latitude_vals: list=None, longitude_vals: list=None, 
                 interpolate: bool=True,
                 convert_to_float_64: bool=False):
    """
    Load raw IFS data (i.e without any normalisation or conversion to mm/hr)

    Args:
        field (str): name of field
        year (int): year
        month (int): month
        day (int): day  
        hour (int): hour
        ifs_data_dir (str, optional): folder with IFS data in. Defaults to IFS_PATH.
        latitude_vals (list, optional): latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): longitude values to filter/interpolate to. Defaults to None.
        interpolate (bool, optional): whether or not to interpolate. Defaults to True.
        convert_to_float_64 (bool, optional): Whether or not to convert to float 64,
    Returns:
        xr.Dataset: dataset
    """
    
    if not isinstance(latitude_vals[0], np.float32):
        latitude_vals = np.array(latitude_vals).astype(np.float32)
        longitude_vals = np.array(longitude_vals).astype(np.float32)

    assert field in all_ifs_fields, ValueError(f"field must be one of {all_ifs_fields}")
    
    t = datetime(year=year, month=month, day=day, hour=hour)
    t_plus_one = datetime(year=year, month=month, day=day, hour=hour) + timedelta(hours=1)

    # Get the nearest forecast starttime
    loaddate, loadtime = get_ifs_forecast_time(year, month, day, hour)
    
    fp = get_ifs_filepath(field=field,
                          loaddate=loaddate,
                          loadtime=loadtime,
                          fcst_dir=ifs_data_dir
                          )

    ds = xr.open_dataset(fp)
    var_names = list(ds.data_vars)
    
    if np.round(ds.longitude.values.max(), 6) < np.round(max(longitude_vals), 6) or np.round(ds.longitude.values.min(), 6) > np.round(min(longitude_vals), 6):
        raise ValueError('Longitude range outside of data range')
    
    if np.round(ds.latitude.values.max(), 6) < np.round(max(latitude_vals),6) or np.round(ds.latitude.values.min(),6) > np.round(min(latitude_vals), 6):
        raise ValueError('Latitude range outside of data range')
    
    assert len(var_names) == 1, ValueError('More than one variable found; cannot automatically infer variable name')
    var_name = list(ds.data_vars)[0]
    
    if convert_to_float_64:
        # Multiplication with float32 leads to some discrepancies
        # But in some cases this is outweighed by speed 
        ds[var_name] = ds[var_name].astype(np.float64)
       
    # Account for cumulative fields
    if var_name in ['tp', 'cp', 'cdir', 'tisr']:
        # Output rainfall during the following hour
        ds =  ds.sel(time=t_plus_one) - ds.sel(time=t)
    else:
        ds = ds.sel(time=t)

    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            if var_name in ['tp', 'tclw', 'cape', 'tisr', 'tcwv', 'cp', 'cin']:
                interpolation_method = 'conservative'
            else:
                interpolation_method = 'bilinear'
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method=interpolation_method)
        else:
            ds = ds.sel(longitude=longitude_vals, method='backfill')
            ds = ds.sel(latitude=latitude_vals, method='backfill')
    
    ds = make_dataset_consistent(ds)
    ds = ds.transpose('latitude', 'longitude')
             
    return ds

def load_ifs(field: str, 
             date, hour: int,
             fcst_dir: str,
             normalisation_strategy: dict,
             norm: bool=False, 
             latitude_vals: list=None, 
             longitude_vals: list=None, 
             constants_path: str=CONSTANTS_PATH):
    """
    Load IFS (including normalisation and conversion to mm/hr)

    Args:
        field (str): name of field
        date (str or datetime): YYYYMMDD string or datetime to forecast
        hour (int): hour to forecast
        norm (bool, optional): whether or not to normalise the data. Defaults to False.
        fcst_dir (str, optional): forecast data directory. Defaults to IFS_PATH.
        normalisation_strategy (str, optional): dict with normalisation details for variables. Defaults to VAR_LOOKUP_IFS.
        latitude_vals (list, optional): latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): longitude_vals to filter/interpolate to. Defaults to None.
        constants_path (str, optional): path to constant data. Defaults to CONSTANTS_PATH.

    Returns:
        np.ndarray: data for the specified field
    """
    
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y%m%d")
        
    ds = load_ifs_raw(field, date.year, date.month, date.day, hour, ifs_data_dir=fcst_dir,
                      latitude_vals=latitude_vals, longitude_vals=longitude_vals, interpolate=True)
    
    var_name = list(ds.data_vars)[0]
    
    if not normalisation_strategy[field].get('negative_vals', True):
        # Make sure no negative values
        ds[var_name] = ds[var_name].clip(min=0)
        
    if field in ['tp', 'cp', 'pr', 'prl', 'prc']:
        # precipitation is measured in metres, so multiply up
        ds[var_name] = 1000 * ds[var_name]
        
    if field == 'cin':
        # Replace null values with 0
        ds[var_name] = ds[var_name].fillna(0)
        
    if norm:
        stats_dict = get_ifs_stats(field, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                            use_cached=True, ifs_data_dir=fcst_dir,
                            output_dir=constants_path)
        # Normalisation here      
        ds = preprocess(field, ds, stats_dict=stats_dict, normalisation_strategy=normalisation_strategy)
    
    y = np.array(ds[var_name][:, :])
    
    return y


def load_fcst_stack(data_source: str, fields: list, 
                    date: str, hour: int, 
                    fcstdir_or_ds: Union[str, xr.Dataset], 
                    normalisation_strategy: dict,
                    constants_dir:str=CONSTANTS_PATH,
                    norm:bool=False,
                    latitude_vals:list=None, longitude_vals:list=None):
    """
    Load forecast 'stack' of all variables

    Args:
        data_source (str): source of data (e.g. ifs)
        fields (list): list of fields to load
        date (str): YYYYMMDD date string to forecast for
        hour (int): hour to forecast for
        fcstdir_or_ds (str): folder with forecast data files or list of datasets (for load_<dataset>_monthly-methods)
        normalisation_strategy (dict): normalisation strategy
        constants_dir (str, optional): folder with constants data in. Defaults to CONSTANTS_PATH.
        norm (bool, optional): whether or not to normalise the data. Defaults to False.
        latitude_vals (list, optional): list of latitude values. Defaults to None.
        longitude_vals (list, optional): list of longitude values. Defaults to None.

    Returns:
        np.ndarray: stacked array of specified variables
    """
    field_arrays = []

    append_fields = True
    if data_source == 'ifs':
        load_function = load_ifs
    elif data_source == 'era5':
        load_function = load_era5
    elif data_source == "era5_monthly":
        load_function = load_era5_from_ds
        append_fields = False
    elif data_source == "cerra_monthly":
        load_function = load_cerra_from_ds
        append_fields = False
    else:
        raise ValueError(f'Unknown data source {data_source}')

    if append_fields:
        for f in fields:
            field_arrays.append(load_function(f, date, hour, fcstdir_or_ds,
                                            latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                                            constants_path=constants_dir, norm=norm,
                                            normalisation_strategy=normalisation_strategy))
        return np.stack(field_arrays, -1)
    else:
        return load_function(fields, date, hour, fcstdir_or_ds, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                             constants_path=constants_dir, norm=norm, normalisation_strategy=normalisation_strategy)


def get_ifs_stats(field: str, latitude_vals: list, longitude_vals: list, output_dir: str=None, 
                   use_cached: bool=True, year: int=NORMALISATION_YEAR,
                   ifs_data_dir: str=IFS_PATH, hours: list=all_fcst_hours):

    min_lat = int(min(latitude_vals))
    max_lat = int(max(latitude_vals))
    min_lon = int(min(longitude_vals))
    max_lon = int(max(longitude_vals))


    # Filepath is specific to make sure we don't use the wrong normalisation stats
    fp = f'{output_dir}/IFS_norm_{field}_{year}_lat{min_lat}-{max_lat}lon{min_lon}-{max_lon}.pkl'

    var_name = None

    if use_cached and os.path.isfile(fp):
        logger.debug('Loading stats from cache')

        with open(fp, 'rb') as f:
            stats = pickle.load(f)

    else:
        print('Calculating stats')
        all_dates = list(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='D'))
        all_dates = [item.date() for item in all_dates]

        datasets = []
        
        for date in all_dates:
            year = date.year
            month = date.month
            day = date.day
            
            for hour in hours:
            
                tmp_ds = load_ifs_raw(field, year, month, day, hour, 
                                    latitude_vals=latitude_vals,
                                    longitude_vals=longitude_vals, 
                                    ifs_data_dir=ifs_data_dir,
                                    interpolate=False)
                if not var_name:
                    var_names = list(tmp_ds.data_vars)
                    assert len(var_names) == 1, ValueError('More than one variable found; cannot automatically infer variable name')
                    var_name = list(tmp_ds.data_vars)[0]

                datasets.append(tmp_ds)
        concat_ds = xr.concat(datasets, dim='time')

        stats = {'min': np.abs(concat_ds[var_name]).min().values,
                'max': np.abs(concat_ds[var_name]).max().values,
                'mean': concat_ds[var_name].mean().values,
                'std': concat_ds[var_name].std().values}

        if output_dir:
            with open(fp, 'wb') as f:
                pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    return stats

### Functions that work with ERA5, CERRA and IMERG data at JSC (from the AtmoRep-project)

def get_era5_monthly_path(variable, lvl, year_month, era5_basedir=ERA5_MONTHLY_PATH, var_name_lookup=VAR_LOOKUP_ERA5_JSC):
    """"
    Get path to monthly ERA5-data grib-files as organized in JSC's file system.
    :param variable: str, variable name
    :param lvl: str, level string of variable (e.g. ml0 for surface data, 'ml137' for model level 137 or 'pl700' for 700 hPa pressure level)
    :param year_month: datetime, year and month of data
    :param era5_basedir: str, base directory of ERA5 data
    :param var_name_lookup: dict, lookup table for retrieving longmanes variables
    :return: str, path to grib-file
    """
    assert isinstance(lvl, str), f"level parameter lvl must be a string, e.g. 'ml0', 'ml137' or 'pl700', but is of type: {type(lvl)}"
    varname_long = var_name_lookup[variable]['longname']
    ym_str = year_month.strftime("y%Y_m%m")

    fname = os.path.join(era5_basedir, varname_long, lvl, f"era5_{varname_long}_{ym_str}_{lvl}.nc")
    if not os.path.exists(fname):
        fname = fname.replace(".nc", ".grib")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Could not find the following data file: {fname.replace('.grib', '')}[.nc, .grib]")
    return fname

def get_cerra_monthly_path(year_month, cerra_basedir=CERRA_MONTHLY_PATH):
    """
    Get path to monthly CERRA-data grib-files as organized in JSC's file system.
    :param year_month: datetime, year and month of data
    :param cerra_basedir: str, base directory of CERRA data
    :return: str, path to grib-file
    """
    ym_str = year_month.strftime("y%Y_m%m")

    return os.path.join(cerra_basedir, f"cerra_{ym_str}.grib")

def get_imerg_monthly_path(year_month, imerg_basedir=IMERG_MONTHLY_PATH):
    """
    Get path to monthly IMERG-data grib-files as organized in JSC's file system.
    :param year_month: datetime, year and month of data
    :param imerg_basedir: str, base directory of IMERG data
    :return: str, path to grib-file
    """
    ym_str = year_month.strftime("y%Y_m%m")

    return os.path.join(imerg_basedir, f"3B-HHR.MS.MRG.3IMERG.{ym_str}.nc")
    
def load_era5_monthly(variables: dict, year_month, latitude_vals=None, longitude_vals=None, era5_datadir=ERA5_MONTHLY_PATH):
    """
    Load data from monthly ERA5 files. Note that the data for all variables and levels are stored in separate files (unlike the CERRA data files)
    :param variables: dict, variables to load with corresponding model levels, example: {"t": [106, 101]}
    :param year_month: datetime, year and month of data
    :param latitude_vals: list, latitude values to filter
    :param longitude_vals: list, longitude values to filter
    :param era5_datadir: str, path to ERA5 data
    """
    # retrieve variables of interest while handling level-dimension 
    # (e.g. {"t": [106, 101]} results into variables named "t_ml106" and "t_ml101", respectively)
    # Note that the data for all variables and levels are stored in separate files
    da_dict = {}
    for var, vls in variables.items():
        for vl in vls:
            fname = get_era5_monthly_path(var, vl, year_month, era5_datadir)
            logger.info(f"Read file {fname}...")
            print(f"Read file {fname}...")
            if fname.endswith(".grib"):
                engine = "cfgrib"
                backend_kwargs = backend_kwargs={"indexpath": ""}
            else:
                engine, backend_kwargs = None, None
                
            with xr.open_dataset(fname, engine=engine, backend_kwargs=backend_kwargs) as ds_era5:
                ds_era5 = make_dataset_consistent(ds_era5)
        
                if latitude_vals is not None and longitude_vals is not None:
                    ds_era5 = filter_by_lat_lon(ds_era5, lon_range=longitude_vals, lat_range=latitude_vals)
                
                da_dict[var] = ds_era5[var].squeeze().load()#.drop_vars("hybrid")
                if var in ["cp", "tp"]:
                    print(f"Scale {var}...")
                    da_dict[var] = xr.where(da_dict[var] < 1.e-05, 0., da_dict[var] * 1000.)                    
            
    ds_new = xr.Dataset(da_dict)
    
    return ds_new

def load_cerra_monthly(variables: dict, year_month, latitude_vals=None, longitude_vals=None, cerra_datadir=CERRA_MONTHLY_PATH):
    """
    Load data from monthly CERRA files. Note that all variables are available in a single monthly file (unlike the ERA5 data files)
    :param variables: dict, variables to load with corresponding model levels, example: {"t": [106, 101]}
    :param year_month: datetime, year and month of data
    :param latitude_vals: list, latitude values to filter 
    :param longitude_vals: list, longitude values to filter
    :param cerra_datadir: str, path to CERRA data
    :return: xr.Dataset, dataset containing the requested variables where model level data is stored in separate variables, example: "t_ml106", "t_ml101"
    """

    fname = get_cerra_monthly_path(year_month, cerra_datadir)

    # open data-file
    ds_cerra = xr.open_dataset(fname, engine="cfgrib", backend_kwargs={"indexpath": ""})

    # filter spatially (if wanted)
    if latitude_vals is not None and longitude_vals is not None:
        ds_cerra = filter_by_lat_lon(ds_cerra, lon_range=longitude_vals, lat_range=latitude_vals)
        

    # retrieve variables of interest while handling level-dimension 
    # (e.g. {"t": [106, 101]} results into variables named "t_ml106" and "t_ml101", respectively)
    da_dict = {}
    for var, mls in variables.items():
        for ml in mls:
            da_dict[f"{var}_ml{ml}"] = ds[var].sel({"hybrid": ml}).squeeze().drop_vars("hybrid")

    ds_new = xr.Dataset(da_dict)
    ds_cerra.close()

    return ds_new

def load_imerg_monthly(variables: List[str], year_month, latitude_vals=None, longitude_vals=None, imerg_datadir=IMERG_MONTHLY_PATH):
    """
    Load data from monthly IMERG files. Note that these files only provide precipitation data.
    :param year_month: datetime, year and month of data
    :param latitude_vals: list, latitude values to filter
    :param longitude_vals: list, longitude values to filter
    :param imerg_datadir: str, path to IMERG data
    :return: xr.Dataset, dataset containing the requested IMERG data
    """
    fname = get_imerg_monthly_path(year_month, imerg_datadir)
    logger.info(f"Read file {fname}...")
    print(f"Read file {fname}...")
    ds_imerg = xr.open_dataset(fname)
    da_imerg = ds_imerg[variables]

    if latitude_vals is not None and longitude_vals is not None:
        da_imerg = filter_by_lat_lon(da_imerg, lon_range=longitude_vals, lat_range=latitude_vals)

    ds_imerg.close()
    return da_imerg


def load_era5_from_ds(variables: List[str], date, hour, ds_era5, log_precip=False, norm=False,
                      normalisation_strategy: Dict = None,
                      latitude_vals=None, longitude_vals=None, var_name_lookup=None,
                      constants_path=CONSTANTS_PATH):
    """
    Function to fetch a sample from a xr.Dataset providing monthly ERA5 data,
    designed to match the structure of the load_fcst function, so they can be interchanged.
    :param variables: List[str], variables to load (obsolete, but retained for compatibility)
    :param date: datetime, date of data
    :param hour: int, hour of data
    :param ds_era5: xr.Dataset, dataset containing ERA5 data
    :param log_precip: Boolean, whether to take logs of the data (Obsolete for this case as using config)
    :param norm: Boolean, whether to normalise (obsolete in this case as using config)
    :param latitude_vals: list, latitude values to filter
    :param longitude_vals: list, longitude values to filter
    :param var_name_lookup: dict, lookup table for retrieving longmanes variables (obsolete, but retained for compatibility)
    :param constants_path: str, path to constants
    :return: np.ndarray, array of sample data
    """
    date_now = date.replace(hour=hour)
    ds_now = ds_era5.sel({"time": date_now})
    logger.info(f"Set ERA5 sample data for {date_now.strftime('%Y-%m-%d %H:00')} UTC...")
    print(f"Set ERA5 sample data for {date_now.strftime('%Y-%m-%d %H:00')} UTC...")
    
    lat_name, lon_name = infer_lat_lon_names(ds_now)

    if norm:
        logger.debug("Normalise data for sample {date_now.strftime('%Y-%m-%d %H:00')} UTC." + 
                     "Consider to use normalized monthly data as input for the load_era5_from_ds-method.")
        print("Normalise data for sample {date_now.strftime('%Y-%m-%d %H:00')} UTC." + 
                     "Consider to use normalized monthly data as input for the load_era5_from_ds-method.")

        norm_stats = get_norm_stats(variables, NORMALISATION_YEAR, data_dir=ERA5_PATH, loader_monthly=load_era5_monthly, dataset_name="era5",
                                    latitude_vals=latitude_vals, longitude_vals=longitude_vals, output_dir=constants_path)
        
        ds_now = normalise_data(ds_now, variables, norm_stats, normalisation_strategy)

    # convert xr.Dataset to xr.DataArray
    da_now = ds_now.to_array().squeeze() 

    da_now = da_now.transpose(lat_name, lon_name, "variable")

    # return as numpy array
    return da_now.values


def load_cerra_from_ds(variables: List[str], date: datetime, hour: int, ds_cerra: xr.Dataset, 
                       log_precip: bool=False, norm: bool=False, latitude_vals=None, longitude_vals=None,
                       var_name_lookup=None, constants_path=CONSTANTS_PATH):
    """
    Function to fetch a sample from a xr.Dataset providing monthly CERRA data,
    designed to match the structure of the load_fcst function, so they can be interchanged.
    :param variables: List[str], variables to load (obsolete, but retained for compatibility)
    :param date: datetime, date of data
    :param hour: int, hour of data
    :param ds_era5: xr.Dataset, dataset containing CERRA data
    :param log_precip: Boolean, whether to take logs of the data (Obsolete for this case as using config)
    :param norm: Boolean, whether to normalise (obsolete in this case as using config)
    :param latitude_vals: list, latitude values to filter
    :param longitude_vals: list, longitude values to filter
    :param var_name_lookup: dict, lookup table for retrieving longmanes variables (obsolete, but retained for compatibility)
    :param constants_path: str, path to constants
    :return: np.ndarray, array of sample data
    """
    date_now = date.replace(hour=hour)
    ds_now = ds_cerra.sel({"time": date_now})
    logger.info(f"Set CERRA sample data for {date_now.strftime('%Y-%m-%d %H:00')} UTC...")
    
    lat_name, lon_name = infer_lat_lon_names(ds_now)

    if norm:
        logger.debug("Normalise data for sample {date_now.strftime('%Y-%m-%d %H:00')} UTC." + 
                     "Consider to use normalized monthly data as input for the load_cerra_from_ds-method.")

        norm_stats = get_norm_stats(variables, NORMALISATION_YEAR, data_dir=CERRA_MONTHLY_PATH, loader_monthly=load_cerra_monthly, dataset_name="cerra",
                                    latitude_vals=latitude_vals, longitude_vals=longitude_vals, output_dir=constants_path)
        
        ds_now = normalise_data(ds_now, variables, norm_stats, NORMALISATION_STRATEGY)

    # convert xr.Dataset to xr.DataArray
    da_now = ds_now.to_array().squeeze() 

    da_now = da_now.transpose(lat_name, lon_name, "variable")

    # return as numpy array
    return da_now.values

def load_imerg_from_ds(date: datetime, hour: int, ds_imerg: xr.Dataset, latitude_vals: list=None, longitude_vals: list=None, 
                       normalisation_type: str=None, constants_path=CONSTANTS_PATH):
    """
    Function to fetch a sample from a xr.Dataset providing monthly IMERG data,
    designed to match the structure of the load_observational_data function, so they can be interchanged.
    :param date: datetime, date of data
    :param hour: int, hour of data
    :param ds_imerg: xr.Dataset, dataset containing IMERG data
    :param latitude_vals: list, latitude values to filter
    :param longitude_vals: list, longitude values to filter
    :param normalisation_type: str, type of normalisation
    :param constants_path: str, path to constants
    """
    date_now = date.replace(hour=hour)
    varname = 'precipitation'
    da_now = ds_imerg[varname].sel({"time": date_now})
    logger.info(f"Set IMERG sample data for {date_now.strftime('%Y-%m-%d %H:00')} UTC...")
    print(f"Set IMERG sample data for {date_now.strftime('%Y-%m-%d %H:00')} UTC...")
    
    lat_name, lon_name = infer_lat_lon_names(da_now)

    if normalisation_type is not None:
        logger.debug(f"Normalise data for sample {date_now.strftime('%Y-%m-%d %H:00')} UTC." + 
                     "Consider to use normalized monthly data as input for the load_imerg_from_ds-method.")
        print(f"Normalise data for sample {date_now.strftime('%Y-%m-%d %H:00')} UTC." + 
                     "Consider to use normalized monthly data as input for the load_imerg_from_ds-method.")

        norm_stats = get_norm_stats([varname], NORMALISATION_YEAR, data_dir=IMERG_MONTHLY_PATH, loader_monthly=load_imerg_monthly, dataset_name="imerg",
                                    latitude_vals=latitude_vals, longitude_vals=longitude_vals, output_dir=constants_path)
        
        da_now = normalise_data(da_now, [varname], norm_stats, normalisation_type)

    # return as numpy array
    return da_now.values  


def get_norm_stats(variables: List[str], norm_year: datetime, data_dir: Union[Path, str], loader_monthly, dataset_name: str,
                   latitude_vals: Tuple[float, float], longitude_vals: Tuple[float, float], output_dir: Union[Path, str], 
                   use_cached: bool = True, mean_dims: List[str] = None):
    
    def calculate_statistics(data: Union[xr.DataArray, xr.Dataset], dims: List[str] = None) -> Dict[str, float]:
        return {
            'min': data.min(dim=dims),
            'max': data.max(dim=dims),
            'mean': data.mean(dim=dims),
            'std': data.std(dim=dims)
        }
    
    fp = Path(f"{output_dir}/{dataset_name}_norm_{norm_year}.pkl")
    
    if use_cached and fp.is_file():
        logger.info(f'Loading stats from file {fp}...')
        print(f'Loading stats from file {fp}...')

        with open(fp, 'rb') as f:
            norms = pickle.load(f)
    else:
        # get list of months for normalization year
        logger.info(f"Calculate stats for year {norm_year}...") 
        print(f"Calculate stats for year {norm_year}...") 
        ds_m = []
        all_m = list(pd.date_range(start=f'{norm_year}-01-01', end=f'{norm_year+1}-01-01' , freq='M'))
        
        for m in all_m[0:3]:
            print("Reduced number of months in deriving stats!!!")
            logger.debug(f"Process data for {m.strftime('%Y-%m')}...")
            ds_m.append(loader_monthly(variables, m, latitude_vals, longitude_vals, data_dir))
            
        logger.debug(f"Concatenate {len(ds_m)} datasets...")
        ds_all = xr.concat(ds_m, dim="time")
        
        logger.debug("Calculate statistics...")
        norms = calculate_statistics(ds_all, mean_dims)
        
        if not fp.is_file():
            logger.info(f"Save statistics derived from year {norm_year} to file '{fp}'...")
            print(f"Save statistics derived from year {norm_year} to file '{fp}'...")
            with open(fp, 'wb') as f:
                pickle.dump(norms, f, pickle.HIGHEST_PROTOCOL)
        else:
            logger.info(f"Normalisation file '{fp}' already exists. Derived normalisation data remains unsafed.")
            
    return norms

def normalise_data(ds, var_suffices, stat_dict, norm_strategy):
    """
    Normalise variables in dataset.
    Note that log and sqrt normalisation are followed by standardisation.
    :param ds: xr.Dataset, dataset containing data
    :param var_suffices: dict, dict containing suffices of variable names
    :param stat_dict: dict, dict of values required for normalisation
    :param norm_strategy: dict, dict containing normalisation strategy for each variable
    :return: xr.Dataset, dataset with normalized data
    """
    data_vars = list(ds.data_vars)
    nvars = len(data_vars)
    varcount = 0
    
    # read suffices of variable names which should correspond to variable names in the normalization_strategy-dictionary
    if isinstance(norm_strategy, dict):
        vars = var_suffices.keys()

        # get unique list of required normalization techniwues applied to data
        norm = list(set([norm_strategy[var].get("normalisation") for var in vars]))
        logger.info(f"Normalize {nvars} variables with {len(norm)} normalization techniques...")
    else:
        assert isinstance(norm_strategy, str)
        
        norm = [norm_strategy]
        
        logger.debug(f"Apply {norm_strategy} to the following variables: {', '.join(data_vars)}")
        print(f"Apply {norm_strategy} to the following variables: {', '.join(data_vars)}")

    # apply each identified normalization to the variables
    for n in norm:
        # get variables that should be normalized with current technique n
        if isinstance(norm_strategy, dict):
            var_norm = [var for var, info in norm_strategy.items() if info.get("normalisation") == n]
            vars2norm = []
        
            for suffix in var_norm:
                vars2norm += [var for var in data_vars if var == suffix or var.startswith(f"{suffix}_")]
                
            logger.debug(f"Apply {n} to the following variables: {', '.join(vars2norm)}")
            print(f"Apply {n} to the following variables: {', '.join(vars2norm)}")
        else:
            vars2norm = data_vars        

        # dataset will be updated in place
        if n == 'standardise':
            ds.update((ds[vars2norm] - stat_dict['mean'][vars2norm]) / stat_dict['std'][vars2norm])

        elif n == 'minmax':
            ds.update((ds[vars2norm] - stat_dict['min'][vars2norm]) / (stat_dict['max'][vars2norm] - stat_dict['min'][vars2norm]))

        elif n == 'log':
            print(ds[vars2norm])
            ds.update(log_plus_1(ds[vars2norm]))
            # additional standardisation
            # This is not done in original paper, cf. Section 2.1 of https://doi.org/10.1029/2022MS003120
            # ds.update((ds[vars_now] - stat_dict['mean'][vars_now]) / stat_dict['std'][vars_now])
        elif n == 'max':
            ds.update(ds[vars2norm] / stat_dict['max'][vars2norm])
            
        elif n == 'sqrt':
            ds.update(np.sqrt(ds[vars2norm]))
            # additional standardisation
            # This is not in the original code
            # ds.update((ds[vars_now] - stat_dict['mean'][vars_now]) / stat_dict['std'][vars_now])
        else:
            raise ValueError(f"Unrecognised normalisation type {n} for variable(-s) {', '.join(vars2norm)}")

        varcount += len(vars2norm)

    # snaity check that all variables have been normalized
    assert varcount == nvars, f"Not all variables have been normalized ({nvars-varcount} are missing)."
    
    return ds

def denormalise_data(ds: xr.Dataset, var_suffices: Dict, stat_dict: Dict, norm_strategy: Dict):
    """
    Denormalise variables in dataset.
    Note that standardisation has been applied after log and sqrt normalisation (cf. normalise_data-method).
    :param ds: xr.Dataset, dataset containing data
    :param var_suffices: dict, dict containing suffices of variable names
    :param stat_dict: dict, dict of values required for normalisation
    :param norm_strategy: dict, dict containing normalisation strategy for each variable
    :return: xr.Dataset, dataset with denormalized data
    """
    # read suffices of variable names which should correspond to variable names in the normalization_strategy-dictionary
    vars = var_suffices.keys()

    # get unique list of required normalization techniwues applied to data
    norm = list(set([norm_strategy[var].get("normalisation") for var in vars]))
    data_vars = list(ds.data_vars)
    nvars = len(data_vars)
    varcount = 0

    logger.info(f"Denormalize {nvars} variables according to {len(norm)} normalization techniques...")

    # apply each identified denormalization to the variables
    for n in norm:
        # get variables that should be normalized with current technique n
        var_suffix_now = [var for var, info in norm_strategy.items() if info.get("normalisation") == n]
        vars_now = []
        for suffix in var_suffix_now:
            vars_now += [var for var in data_vars if var.startswith(f"{suffix}_")]
        logger.debug(f"Invert {n} normalisation to the following variables: {', '.join(vars_now)}")

        if n == 'standardise':
            ds.update(ds[vars_now] * stat_dict['std'][vars_now] + stat_dict['mean'][vars_now])

        elif n == 'minmax':
            ds.update(ds[vars_now] * (stat_dict['max'][vars_now] - stat_dict['min'][vars_now]) + stat_dict['min'][vars_now])

        elif n == 'log':
            # invert standardisation
            # ds[vars_now] * stat_dict['std'][vars_now] + stat_dict['mean'][vars_now]
            # invert log-transformation
            ds.update(np.exp(ds[vars_now]) - 1)
        elif n == 'max':
            ds.update(ds[vars_now] * stat_dict['max'][vars_now])
        elif n == 'sqrt':
            # invert standardisation
            # ds.update(ds[vars_now] * stat_dict['std'][vars_now] + stat_dict['mean'][vars_now])
            # invert sqrt-transformation
            ds.update(ds[vars_now] ** 2)
        else:
            raise ValueError(f"Unrecognised normalisation type {n} for variable(-s) {', '.join(vars_now)}")

        varcount += len(vars_now)

    # snaity check that all variables have been denormalized
    assert varcount == nvars, f"Not all variables have been denormalized ({nvars-varcount} are missing)."
    
    return ds

### Functions that work with the ERA5 data in University of Bristol


def get_era5_filepath_prefix(variable, era_data_dir=ERA5_PATH,
                             var_name_lookup=VAR_LOOKUP_ERA5):
    filepath_attrs = var_name_lookup[variable]

    if filepath_attrs.get('subfolder'):
        return os.path.join(era_data_dir, '{folder}/{subfolder}/ERA5_{subfolder}_{suffix}'.format(**filepath_attrs))
    else:
        return os.path.join(era_data_dir, '{folder}/ERA5_{folder}_{suffix}'.format(**filepath_attrs))


def get_era5_path(variable, year, month, era_data_dir=ERA5_PATH, var_name_lookup=VAR_LOOKUP_ERA5):
    suffix = get_era5_filepath_prefix(variable=variable, era_data_dir=era_data_dir, var_name_lookup=var_name_lookup)
    
    if month == '*':
        return f'{suffix}_{year}{month}.nc'
    else:
        return f'{suffix}_{year}{int(month):02d}.nc'



def load_era5_month_raw(variable, year, month, latitude_vals=None, longitude_vals=None,
                        era_data_dir=ERA5_PATH, interpolate=True):
    
    ds = xr.load_dataset(get_era5_path(variable, year=year, month=month, era_data_dir=era_data_dir))

    # Multiplication with float32 leads to some discrepancies
    ds[variable] = ds[variable].astype(np.float64)
       
    interpolation_method = 'conservative' if variable == 'tp' else 'bilinear'
    
    # only works if longitude and latitude vals specified together
    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method=interpolation_method)
        else:
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, lat_range=latitude_vals)

    # Make sure dataset is consistent with others
    ds = make_dataset_consistent(ds)
    
    if variable == 'tp':
        # Convert to mm
        ds['tp'] = ds['tp'] * 1000
    
    return ds


def load_era5_day_raw(variable: str, year: int, month: int, 
                      day: int, latitude_vals: list=None, 
                      longitude_vals: list=None,
                      era_data_dir: str=ERA5_PATH, interpolate: bool=True):

    month_ds = load_era5_month_raw(variable, year, month, latitude_vals=latitude_vals,
                                    longitude_vals=longitude_vals,
                                    era_data_dir=era_data_dir, interpolate=interpolate)

    day_ds = month_ds.sel(time=f'{year}-{int(month):02d}-{int(day):02d}')
    
    return day_ds


def get_era5_stats(variable: str, longitude_vals: list, latitude_vals: list, 
                   year: int=NORMALISATION_YEAR, output_dir: int=None,
                   era_data_dir: int=ERA5_PATH, use_cached: bool=False):
    
    min_lat = int(min(latitude_vals))
    max_lat = int(max(latitude_vals))
    min_lon = int(min(longitude_vals))
    max_lon = int(max(longitude_vals))
        
    # Filepath is spcific to make sure we don't use the wrong normalisation stats
    fp = f'{output_dir}/ERA_norm_{variable}_{year}_lat{min_lat}-{max_lat}lon{min_lon}-{max_lon}.pkl'

    if use_cached and os.path.isfile(fp):
        logger.debug('Loading stats from cache')

        with open(fp, 'rb') as f:
            stats = pickle.load(f)

    else:

        all_dates = list(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='M'))
        all_dates = [item.date() for item in all_dates]

        datasets = []

        for date in all_dates:
            year = date.year
            month = date.month
            tmp_ds = load_era5_month_raw(variable, year, month, latitude_vals=latitude_vals,
                                         longitude_vals=longitude_vals, 
                                         era_data_dir=era_data_dir,
                                         interpolate=False)

            datasets.append(tmp_ds)
        concat_ds = xr.concat(datasets, dim='time')

        stats = {'min': np.abs(concat_ds[variable]).min().values,
                'max': np.abs(concat_ds[variable]).max().values,
                'mean': concat_ds[variable].mean().values,
                'std': concat_ds[variable].std().values}

        if output_dir:
            with open(fp, 'wb') as f:
                pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    return stats


def load_era5(ifield, date, hour=0, fcst_dir=ERA5_PATH, log_precip=False, norm=False,
              latitude_vals=None, longitude_vals=None, var_name_lookup=VAR_LOOKUP_ERA5,
              constants_path=CONSTANTS_PATH):
    """

    Function to fetch ERA5 data, designed to match the structure of the load_fcst function, so they can be interchanged

    Args:
        latitude_vals:
        constants_path:
        ifield: field to load
        date: str, date in form YYYYMMDD
        hour: int or list, hour or hours to fetch from (Obsolete in this case as only daily data)
        log_precip: Boolean, whether to take logs of the data (Obsolete for this case as using config)
        norm: Boolean, whether to normalise (obsolete in this case as using config)
        fcst_dir:

    Returns:

    """
    dt = datetime.strptime(date, '%Y%m%d')

    ds = load_era5_day_raw(variable=ifield, year=dt.year, month=dt.month, day=dt.day,
                        latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                        era_data_dir=fcst_dir)


    stats_dict = get_era5_stats(ifield, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                                use_cached=True, era_data_dir=fcst_dir,
                                output_dir=constants_path)

    if norm:
        # Normalisation here      
        ds = preprocess(ifield, ds, stats_dict=stats_dict, var_name_lookup=var_name_lookup)
    
    # Only one time index should be returned since this is daily data
    y = np.array(ds[ifield][0, :, :])
    ds.close()

    if ifield == 'tp' and log_precip == True:
        y = normalise_precipitation(y, normalisation_type='log')

    ds.close()

    return y

def get_imerg_filepaths(year: int, month: int, day: int, 
                        hour: int, imerg_data_dir: str=IMERG_PATH, 
                        file_ending: str='.nc'):
    
    dt_start = datetime(year, month, day, hour, 0, 0) 
        
    fp1 = os.path.join(imerg_data_dir, '3B-HHR.MS.MRG.3IMERG.' + dt_start.strftime('%Y%m%d-S%H0000-E%H2959') + f'.{2*hour * 30:04d}.V06B{file_ending}')
    fp2 = os.path.join(imerg_data_dir, '3B-HHR.MS.MRG.3IMERG.' + dt_start.strftime('%Y%m%d-S%H3000-E%H5959') + f'.{(2*hour + 1) * 30:04d}.V06B{file_ending}')
    
    return [fp1, fp2]
    
def load_imerg_raw(year: int, month: int, day: int, 
                   hour:int, latitude_vals: list=None, 
                   longitude_vals: list=None,
                   imerg_data_dir: str=IMERG_PATH, file_ending: str='.nc'):

    if isinstance(latitude_vals, list):
        latitude_vals = np.array(latitude_vals)
        
    if isinstance(longitude_vals, list):
        longitude_vals = np.array(longitude_vals)
        
    if not isinstance(latitude_vals[0], np.float32):
        latitude_vals = latitude_vals.astype(np.float32)
        longitude_vals = longitude_vals.astype(np.float32)
    
    fps = get_imerg_filepaths(year, month, day, hour, imerg_data_dir=imerg_data_dir, file_ending=file_ending)
    
    datasets = []
    
    for fp in fps:
        if file_ending.lower() == '.nc':
            datasets.append(xr.load_dataset(fp))
        elif file_ending.lower() == '.hdf5':
            datasets.append(load_hdf5_file(fp, 'Grid'))
        else:
            raise IOError(f'File formats {file_ending} not supported')

    ds = xr.concat(datasets, dim='time').mean('time')
    
    if ds.lon.values.max() < max(longitude_vals) or ds.lon.values.min() > min(longitude_vals):
        raise ValueError('Longitude range outside of data range')
    
    if ds.lat.values.max() < max(latitude_vals) or ds.lat.values.min() > min(latitude_vals):
        raise ValueError('Latitude range outside of data range')

    # Note we use method nearest; the iMERG data isn't interpolated to be on
    # the same grid as the input forecast necessarily (they don't need to match exactly)
    # Add small shift otherwise it may be non-deterministic as to which side it chooses
    if longitude_vals is not None:
        ds = ds.sel(lon=longitude_vals + 1e-6, method='nearest')

    if latitude_vals is not None:
        ds = ds.sel(lat=latitude_vals + 1e-6, method='nearest')
        
    # Make sure dataset is consistent with others
    ds = make_dataset_consistent(ds)
    ds = ds.transpose('lat', 'lon', 'latv', 'lonv')

    return ds


def load_imerg(date: datetime, hour: int=18, data_dir: str=IMERG_PATH,
               latitude_vals: list=None, longitude_vals: list=None,
               normalisation_type: str=None):
    """

     Function to fetch iMERG data, designed to match the structure of the load_radar function, so they can be
     interchanged

     Args:
         date: str, date in form YYYYMMDD
         hour: int or list, hour or hours to fetch from (Obsolete in this case as only daily data)
         log_precip: Boolean, whether to take logs of the data (Obsolete for this case as using config)
         radar_dir: str, directory where imerg data is stored

     Returns:

     """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y%m%d')
        
    ds = load_imerg_raw(year=date.year, month=date.month, day=date.day, hour=hour,
                        latitude_vals=latitude_vals, longitude_vals=longitude_vals, imerg_data_dir=data_dir)

    # Take mean since data may be half hourly
    precip = ds['precipitationCal'].values
    ds.close()
    
    if normalisation_type is not None:
        precip = normalise_precipitation(precip, normalisation_type=normalisation_type)

    return precip  


if __name__ == '__main__':

    # Batch job for loading / saving only the relevant data

    parser = ArgumentParser(description='Subset data to desired lat/lon range')
    parser.add_argument('--fcst-data-source', choices=['ifs', 'era5'], type=str,
                        help='Source of forecast data')
    parser.add_argument('--obs-data-source', choices=['nimrod', 'imerg'], type=str,
                        help='Source of observational (ground truth) data')
    parser.add_argument('--years', default=2018, nargs='+', type=int,
                        help='Year(s) to process (space separated)')
    parser.add_argument('--months', default=list(range(1, 13)), nargs='+', type=int,
                        help='Year(s) to process (space separated)')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--min-latitude', type=int, required=True)
    parser.add_argument('--max-latitude', type=int, required=True)
    parser.add_argument('--min-longitude', type=int, required=True)
    parser.add_argument('--max-longitude', type=int, required=True)
    parser.add_argument('--input-obs-folder', type=str, default=None)

    args = parser.parse_args()

    print(args)
    
    years = [args.years] if isinstance(args.years, int) else args.years

    latitude_vals = range(args.min_latitude -1, args.max_latitude + 1)
    longitude_vals = range(args.min_longitude -1, args.max_longitude + 1)

    if not os.path.isdir(args.output_dir):
        raise IOError('Output directory does not exist! Please create it or specify a different one')

    imerg_folders = ['IMERG', 'half_hourly', 'final']
    era5_output_dir = os.path.join(args.output_dir, 'ERA5')
    imerg_output_dir = os.path.join(args.output_dir, '/'.join(imerg_folders))

    if not os.path.isdir(era5_output_dir):
        os.mkdir(era5_output_dir)

    for _, properties in VAR_LOOKUP_ERA5.items():

        var_folder = os.path.join(era5_output_dir, properties['folder'])
        if not os.path.isdir(var_folder):
            os.mkdir(var_folder)

        subfolder = properties.get('subfolder')

        if subfolder:
            subfolder = os.path.join(var_folder, subfolder)
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)
    
    dir = args.output_dir
    for folder in imerg_folders:
        current_dir = os.path.join(dir, folder)
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)
        dir = os.path.join(dir, folder)

    all_imerg_dates = []
    for year in years:
        for month in args.months:
            for day in range(1, monthrange(year, month)[1]+1):
                for hour in range(0, 24):
                    all_imerg_dates.append((year, month, day, hour))
    
    print('starting observational data gathering')
    for date_item in tqdm(all_imerg_dates, total=len(all_imerg_dates), position=0, leave=True):

        year, month, day, hour = date_item

        fps = glob(os.path.join(args.input_obs_folder, 
                                f'3B-HHR.MS.MRG.3IMERG.{year}{month:02d}{day:02d}-S{hour:02d}*'))
        
        if len(fps) == 0:
            print(f'No files found for {year}{month:02d}{day:02d}-S{hour:02d}')
        
        for fp in fps:
            ds =load_hdf5_file(fp)
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, 
                                   lat_range=latitude_vals)
            path_suffix = fp.split('/')[-1]
            imerg_path = os.path.join(imerg_output_dir, path_suffix).replace('.HDF5', '.nc')
            ds.to_netcdf(imerg_path)
            ds.close()

    # print("starting forcast data gathering")
    # all_era5_dates = []
    # for year in years:
    #     for month in args.months:
    #         all_era5_dates.append((year, month))
            
    # for date_item in tqdm(all_era5_dates, total=len(all_era5_dates)): 
    #     year, month = date_item              
    #     for variable in VAR_LOOKUP_ERA5:
    #         input_era5_path = get_era5_path(variable, year, month)
    #         era5_ds = xr.load_dataset(input_era5_path)
    #         era5_ds = filter_by_lat_lon(era5_ds, 
    #                                     lat_range=latitude_vals, lon_range=longitude_vals)
    #         prefix = get_era5_filepath_prefix(variable=variable, 
    #                                           era_data_dir=era5_output_dir)
    #         output_era5_path = f'{prefix}_{year}{month:02d}.nc'

    #         era5_ds.to_netcdf(output_era5_path)
    #         era5_ds.close()
