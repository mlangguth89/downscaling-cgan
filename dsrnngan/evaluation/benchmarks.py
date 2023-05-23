import logging
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
from typing import Iterable, Union
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression, HuberRegressor
from numba import jit

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)


def nn_interp_model(data, upsampling_factor):
    return np.repeat(np.repeat(data, upsampling_factor, axis=-1), upsampling_factor, axis=-2)


def zeros_model(data, upsampling_factor):
    return nn_interp_model(np.zeros(data.shape), upsampling_factor)

@jit(nopython=True)
def empirical_quantile_map(obs_train: np.ndarray, model_train: np.ndarray, s: np.ndarray,
                           quantiles: Union[int, ArrayLike]=10) -> np.ndarray:
    """
    Empirical quantile mapping for bias correction

    Args:
        obs_train (np.ndarray): Observational training data to construct the quantiles
        model_train (np.ndarray): Model training data to construct the quantiles
        s (np.ndarray): 1D series to correct
        quantiles (Union[int, ArrayLike], optional): Either the number of quantiles to use, or the locations of quantiles. Defaults to 10.
        extrapolate (str, optional): Type of extrapolation to use. Defaults to None.

    Returns:
        np.ndarray: quantile mapped data
    """
    if isinstance(quantiles, int):
        quantiles = np.linspace(0, 1., quantiles)
    else:
        quantiles = np.array(quantiles)
        if 1.0 not in quantiles:
            # We need to have the maximum value in order to deal with extreme values
            np.append(quantiles, 1.0)
    
    q_obs = np.quantile(obs_train[np.isfinite(obs_train)], quantiles)
    q_model = np.quantile(model_train[np.isfinite(model_train)], quantiles)
    
    model_corrected = np.interp(s, q_model, q_obs, right=np.nan)
    
    # if extrapolate is None:
    #     model_corrected[s > np.max(q_model)] = q_obs[-1]
    #     model_corrected[s < np.min(q_model)] = q_obs[0]
        
    # elif extrapolate == 'constant':
    extreme_inds = np.argwhere(s >= np.max(q_model))
                    
    if len(extreme_inds) > 0:
        uplift = q_obs[-1] - q_model[-1]
        model_corrected[extreme_inds] = s[extreme_inds] + uplift
    # else:
    #     raise ValueError(f'Unrecognised value for extrapolate: {extrapolate}')
    # return model_corrected


def quantile_map_grid(array_to_correct: np.ndarray, fcst_train_data: np.ndarray, 
                      obs_train_data: np.ndarray, quantiles: Union[int, ArrayLike], neighbourhood_size: int=0,
                      ) -> np.ndarray:
    """Quantile map data that is on a grid

    Args:
        array_to_correct (np.ndarray): Data to be quantile mapped. Should have dimension (n_samples, width, height)
        fcst_train_data (np.ndarray): Training data for the forecast, to calculate quantiles
        obs_train_data (np.ndarray): Training data for the observations, to calculate quantiles
        quantiles (Union[int, ArrayLike], optional): Either the number of quantiles to use, or the locations of quantiles. Defaults to 10.
        neighbourhood_size (int, optional): Radius of pixels around the grid cell in which to calculate the quantiles. Defaults to 0.
        extrapolate (str, optional): Type of extrapolation for values that lie outside the training range. Defaults to 'constant'.

    Returns:
        np.ndarray: quantile mapped array
    """
    (_, width, height) = array_to_correct.shape
      
    fcst_corrected = np.empty(array_to_correct.shape)
    fcst_corrected[:,:,:] = np.nan
    
    for w in tqdm(range(width), file=sys.stdout):
        for h in range(height):
            w_range = np.arange(max(w - neighbourhood_size,0), min(w + neighbourhood_size + 1, width), 1)
            h_range = np.arange(max(h - neighbourhood_size,0), min(h + neighbourhood_size + 1, height), 1)
            result = empirical_quantile_map(obs_train=obs_train_data[:,w_range,:][:,:,h_range], 
                                                        model_train=fcst_train_data[:,w_range,:][:,:,h_range], 
                                                        s=array_to_correct[:,w,h],
                                                        quantiles=quantiles)
            fcst_corrected[:,w,h] = result
            
    return fcst_corrected

class QuantileMapper():
    
    def __init__(self, month_ranges: list, 
                 latitude_range: Iterable, longitude_range: Iterable, quantile_locs: list,
                 num_lat_lon_chunks: int=2) -> None:
        
        self.month_ranges = month_ranges
        self.latitude_range = [np.round(item, 2) for item in sorted(latitude_range)]
        self.longitude_range = [np.round(item, 2) for item in sorted(longitude_range)]
        self.num_lat_lon_chunks = num_lat_lon_chunks
        
        if 1.0 not in quantile_locs:
            # We need to have the maximum value in order to deal with extreme values
            quantile_locs.append(1.0)
            
        self.quantile_locs = quantile_locs
        self.quantile_latitude_groupings = None
        self.quantile_longitude_groupings = None
        self.quantile_date_groupings = None
        self.quantiles_by_area = None

    def get_quantile_areas(self, training_dates, training_hours=None):
        
        if isinstance(training_dates, np.ndarray):
            training_dates = list(training_dates.copy())
        
        if training_hours is not None:
            date_hour_list = list(zip(training_dates, training_hours))
        else:
            date_hour_list = list(zip(training_dates,[0]*len(training_dates)))

        date_chunks =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list if item[0].month in month_range] for month_range in self.month_ranges}
        date_indexes =  {k : [date_hour_list.index(item) for item in chunk] for k, chunk in date_chunks.items()}

        assert len(list(chain.from_iterable(date_chunks.values()))) == len(training_dates)
        
        lat_chunk_size = int(len(self.latitude_range)/ self.num_lat_lon_chunks)
        lat_range_chunks = [self.latitude_range[n*lat_chunk_size:(n+1)*lat_chunk_size] for n in range(self.num_lat_lon_chunks)]
        lat_range_chunks[-1] = lat_range_chunks[-1] + self.latitude_range[self.num_lat_lon_chunks*lat_chunk_size:]

        lon_chunk_size = int(len(self.longitude_range)/ self.num_lat_lon_chunks)
        lon_range_chunks = [self.longitude_range[n*lon_chunk_size:(n+1)*lon_chunk_size] for n in range(self.num_lat_lon_chunks)]
        lon_range_chunks[-1] = lon_range_chunks[-1] + self.longitude_range[self.num_lat_lon_chunks*lon_chunk_size:]

        self.quantile_date_groupings = {}
        for t_name, d in date_indexes.items():
            
            self.quantile_date_groupings[f't{t_name}'] = d
            
        self.quantile_latitude_groupings = {}
        for n, lat_chunk in enumerate(lat_range_chunks):
            
            lat_rng = [lat_chunk[0], lat_chunk[-1]]
                
            lat_index_range = [self.latitude_range.index(lat_rng[0]), self.latitude_range.index(lat_rng[1])]
            
            self.quantile_latitude_groupings[n] = {'lat_index_range': lat_index_range,
                                                   'lat_range_mean': np.mean(lat_index_range)}
            
        self.quantile_longitude_groupings = {}
        for m, lon_chunk in enumerate(lon_range_chunks):
                
            lon_rng = [lon_chunk[0], lon_chunk[-1]]
            
            lon_index_range = [self.longitude_range.index(lon_rng[0]), self.longitude_range.index(lon_rng[1])]
            
            self.quantile_longitude_groupings[m] = {'lon_index_range': lon_index_range,
                                                    'lon_range_mean': np.mean(lon_index_range)}
            
        return self.quantile_date_groupings, self.quantile_latitude_groupings, self.quantile_longitude_groupings


    def train(self, fcst_data, obs_data, training_dates, training_hours):
        
        self.get_quantile_areas(training_dates=training_dates, training_hours=training_hours)
  
        self.quantiles_by_area = {}
        for time_period, date_indexes in self.quantile_date_groupings.items():
            self.quantiles_by_area[time_period] = {}
            
            for n, lat_grouping in self.quantile_latitude_groupings.items():
                for m, lon_grouping in self.quantile_longitude_groupings.items():
                    
                    area_id = f'lat{n}_lon{m}'

                    lat_index_range = lat_grouping['lat_index_range']
                    lon_index_range = lon_grouping['lon_index_range']
                    
                    fcst = fcst_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
                    obs = obs_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
                    
                    obs_quantiles = np.quantile(obs.flatten(), self.quantile_locs)
                    fcst_quantiles = np.quantile(fcst.flatten(), self.quantile_locs)

                    self.quantiles_by_area[time_period][area_id] = {'fcst_quantiles': fcst_quantiles, 
                                                                    'obs_quantiles': obs_quantiles}

        return self.quantiles_by_area
    
    @staticmethod
    def get_weighted_quantiles(lat_index: int, lon_index: int, 
                               area_centres: dict, quantiles_for_time_period: dict):
        
        # TODO: make this work in time as well as space
        coordinate_array = np.array([lat_index, lon_index])
                    
        # Calculate distance via exp(-|a-b|_1)
        exp_weighting = {k: np.exp(- np.linalg.norm(v - coordinate_array, ord=1)) for k, v in area_centres.items()}
        normalising_factor = np.sum(list(exp_weighting.values()))
        area_weights = {k: v / normalising_factor for k, v in exp_weighting.items()}
        
        # Calculate weighted average of quantiles to use
        obs_quantiles = np.sum(np.stack([v['obs_quantiles'] * area_weights[k] for k, v in quantiles_for_time_period.items()], axis=0), axis=0)
        fcst_quantiles = np.sum(np.stack([v['fcst_quantiles'] * area_weights[k] for k, v in quantiles_for_time_period.items()], axis=0), axis=0)
        
        return obs_quantiles, fcst_quantiles
        
        
    def get_quantile_mapped_forecast(self, fcst: np.ndarray, dates: Iterable, hours: Iterable=None):

        # Find indexes of dates in test set relative to the date chunks
        
        fcst = fcst.copy()

        if hours is not None:
            date_hour_list = list(set(zip(dates,hours)))
        else:
            date_hour_list = list(set(zip(dates,[0]*len(dates))))

        test_date_chunks =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list if item[0].month in month_range] for month_range in self.month_ranges}
        test_date_indexes = {k : [date_hour_list.index(item) for item in chunk] for k, chunk in test_date_chunks.items()}
        
        (_, lat_dim, lon_dim) = fcst.shape
        
        fcst_corrected = np.empty(fcst.shape)
        fcst_corrected[:,:,:] = np.nan
        area_centres = {f'lat{n}_lon{m}': np.array([self.quantile_latitude_groupings[n]['lat_range_mean'], self.quantile_longitude_groupings[m]['lon_range_mean']]) for n in range(self.num_lat_lon_chunks) for m in range(self.num_lat_lon_chunks)}

        for date_index_name, d_ix in test_date_indexes.items():
            
            quantiles_for_time_period = self.quantiles_by_area[f't{date_index_name}']

            for lat_index in tqdm(range(lat_dim)):
                for lon_index in range(lon_dim):
                    
                    obs_quantiles, fcst_quantiles = self.get_weighted_quantiles(lat_index, lon_index, area_centres, quantiles_for_time_period)
             
                    tmp_fcst_array = fcst[d_ix, lat_index, lon_index] 
                    tmp_fcst_array = np.interp(tmp_fcst_array, fcst_quantiles, obs_quantiles)
                                
                    # Deal with zeros; assign random bin
                    ifs_zero_quantiles = [n for n, q in enumerate(fcst_quantiles) if q == 0.0]
                    if ifs_zero_quantiles:
                        zero_inds = np.argwhere(tmp_fcst_array == 0.0)
                        
                        if len(zero_inds) > 0:
                            tmp_fcst_array[zero_inds] = np.array(obs_quantiles)[np.random.choice(ifs_zero_quantiles, size=zero_inds.shape)]
                    
                    # Deal with values outside the training range
                    max_training_forecast_val = np.max(fcst_quantiles)
                    max_obs_forecast_val = np.max(obs_quantiles)
                    extreme_inds = np.argwhere(fcst[d_ix, lat_index, lon_index]  >= max_training_forecast_val)
                    
                    if len(extreme_inds) > 0:
                        uplift = max_obs_forecast_val - max_training_forecast_val
                        tmp_fcst_array[extreme_inds] = tmp_fcst_array[extreme_inds] + uplift
                    
                    fcst_corrected[d_ix,lat_index,lon_index] = tmp_fcst_array

        return fcst_corrected


def get_exponential_tail_params(data, percentile_threshold=0.9999):
    
    threshold_val = np.quantile(data, percentile_threshold)
    
    vals_over_threshold = data[np.where(data >= threshold_val)]
    vals, bins = np.histogram(vals_over_threshold - threshold_val, bins=100, density=False)
    
    bin_centres = np.array([0.5*(bins[n] + bins[n+1]) for n in range(len(bins)-1)]).reshape(-1, 1)

    # find first element that is 1; cut off fitting at this point
    first1 = np.where(vals <= 1)[0][0]

    vals = vals[:first1]
    bin_centres = bin_centres[:first1]

    reg = HuberRegressor().fit(bin_centres, np.log(vals))
    
    return reg.coef_, reg.intercept_, (bin_centres, vals)                