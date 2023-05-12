import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
from typing import Iterable, Union
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression, HuberRegressor

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)


def nn_interp_model(data, upsampling_factor):
    return np.repeat(np.repeat(data, upsampling_factor, axis=-1), upsampling_factor, axis=-2)


def zeros_model(data, upsampling_factor):
    return nn_interp_model(np.zeros(data.shape), upsampling_factor)

def empirical_quantile_map(obs_train: np.ndarray, model_train: np.ndarray, s: np.ndarray,
                           quantiles: Union[int, ArrayLike]=10, extrapolate: str=None) -> np.ndarray:
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
    
    if extrapolate is None:
        model_corrected[s > np.max(q_model)] = q_obs[-1]
        model_corrected[s < np.min(q_model)] = q_obs[0]
        
    elif extrapolate == 'constant':
        extreme_inds = np.argwhere(s >= np.max(q_model))
                        
        if len(extreme_inds) > 0:
            uplift = q_obs[-1] - q_model[-1]
            model_corrected[extreme_inds] = s[extreme_inds] + uplift
    else:
        raise ValueError(f'Unrecognised value for extrapolate: {extrapolate}')
    return model_corrected


def quantile_map_grid(array_to_correct: np.ndarray, fcst_train_data: np.ndarray, 
                      obs_train_data: np.ndarray, quantiles: Union[int, ArrayLike], neighbourhood_size: int=0,
                      extrapolate: str='constant') -> np.ndarray:
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
    
    for w in range(width):
        for h in range(height):
            w_range = np.arange(max(w - neighbourhood_size,0), min(w + neighbourhood_size + 1, width), 1)
            h_range = np.arange(max(h - neighbourhood_size,0), min(h + neighbourhood_size + 1, height), 1)
            result = empirical_quantile_map(obs_train=obs_train_data[:,w_range,:][:,:,h_range], 
                                                        model_train=fcst_train_data[:,w_range,:][:,:,h_range], 
                                                        s=array_to_correct[:,w,h],
                                                        quantiles=quantiles, extrapolate=extrapolate)
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
        self.quantile_areas = None
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

        self.quantile_areas = {}
        for t_name, d in date_indexes.items():
            for n, lat_chunk in enumerate(lat_range_chunks):
                for m, lon_chunk in enumerate(lon_range_chunks):
                    
                    lat_rng = [lat_chunk[0], lat_chunk[-1]]
                    lon_rng = [lon_chunk[0], lon_chunk[-1]]
                    
                    lat_index_range = [self.latitude_range.index(lat_rng[0]), self.latitude_range.index(lat_rng[1])]
                    lon_index_range = [self.longitude_range.index(lon_rng[0]), self.longitude_range.index(lon_rng[1])]
                    
                    self.quantile_areas[f't{t_name}_lat{n}_lon{m}'] = {'lat_range': lat_rng, 'lon_range': lon_rng,
                                                            'lat_index_range': lat_index_range,
                                                            'lon_index_range': lon_index_range,
                                                            'date_indexes': d}
        return self.quantile_areas


    def train(self, fcst_data, obs_data, training_dates, training_hours):
        
        self.get_quantile_areas(training_dates=training_dates, training_hours=training_hours)
  
        self.quantiles_by_area = {}
        for k, q in self.quantile_areas.items():
            lat_index_range = q['lat_index_range']
            lon_index_range = q['lon_index_range']
            date_indexes = q['date_indexes']
            
            fcst = fcst_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
            obs = obs_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
            
            obs_quantiles = np.quantile(obs.flatten(), self.quantile_locs)
            fcst_quantiles = np.quantile(fcst.flatten(), self.quantile_locs)

            self.quantiles_by_area[k] = {'fcst_quantiles': list(zip(self.quantile_locs, fcst_quantiles)), 
                                    'obs_quantiles': list(zip(self.quantile_locs, obs_quantiles))}

        return self.quantiles_by_area

    def get_quantile_mapped_forecast(self, fcst: np.ndarray, dates: Iterable, hours: Iterable=None):

        # Find indexes of dates in test set relative to the date chunks
        
        fcst = fcst.copy()

        if hours is not None:
            date_hour_list = list(zip(dates,hours))
        else:
            date_hour_list = list(zip(dates,[0]*len(dates)))

        test_date_chunks =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list if item[0].month in month_range] for month_range in self.month_ranges}
        test_date_indexes = {k : [date_hour_list.index(item) for item in chunk] for k, chunk in test_date_chunks.items()}
        
        (_, lat_dim, lon_dim) = fcst.shape
        
        fcst_corrected = np.empty(fcst.shape)
        fcst_corrected[:,:,:] = np.nan

        for date_index_name, d_ix in test_date_indexes.items():
            for lat_index in range(lat_dim):
                for lon_index in range(lon_dim):
                    
                    area_name = [k for k, v in self.quantile_areas.items() if lat_index in range(v['lat_index_range'][0], v['lat_index_range'][1]+1) and lon_index in range(v['lon_index_range'][0], v['lon_index_range'][1]+1)]
                    area_name = [a for a in area_name if a.startswith(f't{date_index_name}')]
                    
                    # If lat and lon ranges are mismatched, then it can't find the area name
                    # For now just exclude these pixels until fixed properly
                
                    if area_name:
                                    
                        if len(area_name) > 1:
                            raise ValueError('Too many quadrants, something is wrong')
                        else:
                            area_name = area_name[0]
                            
                        tmp_fcst_array = fcst[d_ix, lat_index, lon_index] 
                        
                        obs_quantiles = [item[1] for item in self.quantiles_by_area[area_name]['obs_quantiles']]
                        fcst_quantiles = [item[1] for item in self.quantiles_by_area[area_name]['fcst_quantiles']]

                        quantile_locs = [item[0] for item in self.quantiles_by_area[area_name]['obs_quantiles']]
                        assert set(quantile_locs) == set([item[0] for item in self.quantiles_by_area[area_name]['fcst_quantiles']])

                        fcst_corrected[d_ix,lat_index,lon_index] = np.interp(tmp_fcst_array, fcst_quantiles, obs_quantiles)
                                    
                        # Deal with zeros; assign random bin
                        ifs_zero_quantiles = [n for n, q in enumerate(fcst_quantiles) if q == 0.0]
                        if ifs_zero_quantiles:
                            zero_inds = np.argwhere(tmp_fcst_array == 0.0)
                            fcst_corrected[zero_inds, lat_index, lon_index ] = np.array(obs_quantiles)[np.random.choice(ifs_zero_quantiles, size=zero_inds.shape)]
                        
                        # Deal with values outside the training range
                        max_training_forecast_val = np.max(fcst_quantiles)
                        max_obs_forecast_val = np.max(obs_quantiles)
                        extreme_inds = np.argwhere(tmp_fcst_array >= max_training_forecast_val)
                        
                        if len(extreme_inds) > 0:
                            uplift = max_obs_forecast_val - max_training_forecast_val
                            fcst_corrected[extreme_inds, lat_index, lon_index] = tmp_fcst_array[extreme_inds] + uplift
                        
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