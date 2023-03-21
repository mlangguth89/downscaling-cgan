import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
from sklearn.linear_model import LinearRegression, HuberRegressor


logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)


def nn_interp_model(data, upsampling_factor):
    return np.repeat(np.repeat(data, upsampling_factor, axis=-1), upsampling_factor, axis=-2)


def zeros_model(data, upsampling_factor):
    return nn_interp_model(np.zeros(data.shape), upsampling_factor)

def get_quantile_areas(dates, month_ranges, latitude_range, longitude_range, hours=None, num_lat_lon_chunks=2):

    if isinstance(dates, np.ndarray):

        dates = list(dates.copy())
    
    if hours is not None:
        date_hour_list = list(zip(dates,hours))
    else:
        date_hour_list = list(zip(dates,[0]*len(dates)))

    lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
    lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]

    date_chunks =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list if item[0].month in month_range] for month_range in month_ranges}
    date_indexes =  {k : [date_hour_list.index(item) for item in chunk] for k, chunk in date_chunks.items()}

    assert len(list(chain.from_iterable(date_chunks.values()))) == len(dates)

    
    lat_chunk_size = int(len(lat_range_list)/num_lat_lon_chunks)
    lat_range_chunks = [lat_range_list[n*lat_chunk_size:(n+1)*lat_chunk_size] for n in range(num_lat_lon_chunks)]
    lat_range_chunks[-1] = lat_range_chunks[-1] + lat_range_list[num_lat_lon_chunks*lat_chunk_size:]

    lon_chunk_size = int(len(lon_range_list)/num_lat_lon_chunks)
    lon_range_chunks = [lon_range_list[n*lon_chunk_size:(n+1)*lon_chunk_size] for n in range(num_lat_lon_chunks)]
    lon_range_chunks[-1] = lon_range_chunks[-1] + lon_range_list[num_lat_lon_chunks*lon_chunk_size:]

    quantile_areas = {}
    for t_name, d in date_indexes.items():
        for n, lat_chunk in enumerate(lat_range_chunks):
            for m, lon_chunk in enumerate(lon_range_chunks):
                
                lat_rng = [lat_chunk[0], lat_chunk[-1]]
                lon_rng = [lon_chunk[0], lon_chunk[-1]]
                
                lat_index_range = [lat_range_list.index(lat_rng[0]), lat_range_list.index(lat_rng[1])]
                lon_index_range = [lon_range_list.index(lon_rng[0]), lon_range_list.index(lon_rng[1])]
                
                quantile_areas[f't{t_name}_lat{n}_lon{m}'] = {'lat_range': lat_rng, 'lon_range': lon_rng,
                                                        'lat_index_range': lat_index_range,
                                                        'lon_index_range': lon_index_range,
                                                        'date_indexes': d}
    return quantile_areas


def get_quantiles_by_area(quantile_areas, fcst_data, obs_data, quantile_locs, quantile_threshold=None):
    
    quantiles_by_area = {}
    for k, q in quantile_areas.items():
        lat_index_range = q['lat_index_range']
        lon_index_range = q['lon_index_range']
        date_indexes = q['date_indexes']
        
        fcst = fcst_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
        obs = obs_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
        
        obs_quantiles = np.quantile(obs.flatten(), quantile_locs)
        fcst_quantiles = np.quantile(fcst.flatten(), quantile_locs)

        quantiles_by_area[k] = {'fcst_quantiles': list(zip(quantile_locs, fcst_quantiles)), 
                                'obs_quantiles': list(zip(quantile_locs, obs_quantiles))}

    return quantiles_by_area

def get_quantile_mapped_forecast(fcst, dates, month_ranges, quantile_areas, quantiles_by_area, hours=None, 
                                 quantile_threshold=0.99999):
    # Find indexes of dates in test set relative to the date chunks
    
    fcst = fcst.copy()

    if hours is not None:
        date_hour_list = list(zip(dates,hours))
    else:
        date_hour_list = list(zip(dates,[0]*len(dates)))

    test_date_chunks =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list if item[0].month in month_range] for month_range in month_ranges}
    test_date_indexes = {k : [date_hour_list.index(item) for item in chunk] for k, chunk in test_date_chunks.items()}
    
    (_, lat_dim, lon_dim) = fcst.shape
    
    fcst_corrected = np.empty(fcst.shape)
    fcst_corrected[:,:,:] = np.nan

    for date_index_name, d_ix in test_date_indexes.items():
        for lat_index in range(lat_dim):
            for lon_index in range(lon_dim):
                
                area_name = [k for k, v in quantile_areas.items() if lat_index in range(v['lat_index_range'][0], v['lat_index_range'][1]+1) and lon_index in range(v['lon_index_range'][0], v['lon_index_range'][1]+1)]
                area_name = [a for a in area_name if a.startswith(f't{date_index_name}')]
                
                # If lat and lon ranges are mismatched, then it can't find the area name
                # For now just exclude these pixels until fixed properly
            
                if area_name:
                                
                    if len(area_name) > 1:
                        raise ValueError('Too many quadrants, something is wrong')
                    else:
                        area_name = area_name[0]
                        
                    tmp_fcst_array = fcst[d_ix, lat_index, lon_index] 
                    
                    
                    imerg_quantiles = [item[1] for item in quantiles_by_area[area_name]['obs_quantiles']]
                    ifs_quantiles = [item[1] for item in quantiles_by_area[area_name]['fcst_quantiles']]

                    quantile_locs = [item[0] for item in quantiles_by_area[area_name]['obs_quantiles']]
                    assert set(quantile_locs) == set([item[0] for item in quantiles_by_area[area_name]['fcst_quantiles']])

                    fcst_corrected[d_ix,lat_index,lon_index] = np.interp(tmp_fcst_array, ifs_quantiles, imerg_quantiles)
                                
                    # Deal with zeros; assign random bin
                    ifs_zero_quantiles = [n for n, q in enumerate(ifs_quantiles) if q == 0.0]
                    
                    zero_inds = np.argwhere(tmp_fcst_array == 0.0)
                    fcst_corrected[zero_inds, lat_index, lon_index ] = np.array(imerg_quantiles)[np.random.choice(ifs_zero_quantiles, size=zero_inds.shape)]
                    
                    # Find nearest quantile to the threshold
                    quantile_threshold_ix = np.abs(np.array(quantile_locs) - quantile_threshold).argmin()
                    ifs_quantile_threshold = ifs_quantiles[quantile_threshold_ix]
                    imerg_quantile_threshold = imerg_quantiles[quantile_threshold_ix]
                    extreme_inds = np.argwhere(tmp_fcst_array >= ifs_quantile_threshold)

                    uplift = imerg_quantile_threshold - ifs_quantile_threshold

                    fcst_corrected[extreme_inds, lat_index, lon_index ] = tmp_fcst_array[extreme_inds] + uplift
    
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