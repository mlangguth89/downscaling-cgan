import numpy as np

def nn_interp_model(data, upsampling_factor):
    return np.repeat(np.repeat(data, upsampling_factor, axis=-1), upsampling_factor, axis=-2)


def zeros_model(data, upsampling_factor):
    return nn_interp_model(np.zeros(data.shape), upsampling_factor)


def get_quantiles_by_area(quantile_areas, fcst_data, obs_data, quantile_boundaries):
    quantiles_by_area = {}
    for k, q in quantile_areas.items():
        lat_index_range = q['lat_index_range']
        lon_index_range = q['lon_index_range']
        date_indexes = q['date_indexes']
        
        fcst = fcst_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
        obs = obs_data[date_indexes, lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
        
        obs_quantiles = np.quantile(obs.flatten(), quantile_boundaries)
        fcst_quantiles = np.quantile(fcst.flatten(), quantile_boundaries)
        
        fcst_bin_edges = list(fcst_quantiles)
        fcst_bin_edges[-1] = 5000 # Extreme value to catch all values into at least one bin
        obs_bin_edges = list(obs_quantiles)

        imerg_bin_centres = np.array([0.5*(obs_bin_edges[n+1] + obs_bin_edges[n]) for n in range(len(obs_bin_edges)-1)])      
            
        quantiles_by_area[k] = {'ifs_bin_edges': fcst_bin_edges, 'imerg_bin_centres': imerg_bin_centres, 'imerg_bin_edges': obs_bin_edges,
                                'ifs_quantiles': fcst_quantiles}

    return quantiles_by_area

def get_quantile_mapped_forecast(fcst, dates, hours, month_ranges, quantile_areas, quantiles_by_area):
    # Find indexes of dates in test set relative to the date chunks

    date_hour_list = list(zip(dates,hours))

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
                               
                if len(area_name) > 1:
                    raise ValueError('Too many quadrants, something is wrong')
                else:
                    area_name = area_name[0]
            
                tmp_fcst_array = fcst[d_ix, lat_index, lon_index] 
                
                imerg_bin_centres = quantiles_by_area[area_name]['imerg_bin_centres']
                ifs_bin_edges = quantiles_by_area[area_name]['ifs_bin_edges']
                    
                inds = np.digitize(tmp_fcst_array, ifs_bin_edges) - 1
                
                assert inds.size == len(d_ix)

                fcst_corrected[d_ix,lat_index,lon_index] = imerg_bin_centres[inds]
                
                # Deal with zeros; assign random bin
                ifs_zero_bin_edges = [n for n, be in enumerate(ifs_bin_edges) if be ==0.0]
                
                zero_inds = np.argwhere(tmp_fcst_array == 0.0)
                fcst_corrected[zero_inds, lat_index, lon_index ] = imerg_bin_centres[np.random.choice(ifs_zero_bin_edges, size=zero_inds.shape)]

    
    return fcst_corrected
                