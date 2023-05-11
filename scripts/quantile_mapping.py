"""
Script to quantile map the forecasts

Note that this requires data to be prepared from the training set, and so is currently
not in a fit state to be used without a proper pipeline being set up    
"""

import pickle
import os, sys
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.data.data import denormalise, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE
from dsrnngan.evaluation.benchmarks import get_quantile_areas, get_quantiles_by_area, get_quantile_mapped_forecast
from dsrnngan.utils.utils import load_yaml_file, get_best_model_number
from dsrnngan.data.data import denormalise



###########################
# Load model data
###########################

model_type = 'cropped_4000'

log_folders = {'basic': '/user/work/uz22147/logs/cgan/d9b8e8059631e76f/n1000_201806-201905_e50',
               'full_image': '/user/work/uz22147/logs/cgan/43ae7be47e9a182e_full_image/n1000_201806-201905_e50',
               'cropped': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n2000_201806-201905_e20',
               'cropped_4000': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n4000_201806-201905_e10',
               'reweighted': '/user/work/uz22147/logs/cgan/de5750a9ef3bed6d/n3000_201806-201905_e10',
               'cropped_v2': '/user/work/uz22147/logs/cgan/f6998afe16c9f955/n4000_201806-201905_e10'}
# Get best model
log_folder = log_folders[model_type]

model_number = get_best_model_number(log_folder=log_folder)

log_folder = log_folders[model_type]
with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
truth_array = arrays['truth']
samples_gen_array = arrays['samples_gen']
fcst_array = arrays['fcst_array']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
training_dates = [d[0] for d in arrays['dates']]
training_hours = [h[0] for h in arrays['hours']]

assert len(set(list(zip(training_dates, training_hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
(n_samples, width, height, ensemble_size) = samples_gen_array.shape

###########################

###########################

# Get lat/lon range from log folder
base_folder = '/'.join(log_folder.split('/')[:-1])
config = load_yaml_file(os.path.join(base_folder, 'setup_params.yaml'))

# Locations
min_latitude = config['DATA']['min_latitude']
max_latitude = config['DATA']['max_latitude']
latitude_step_size = config['DATA']['latitude_step_size']
min_longitude = config['DATA']['min_longitude']
max_longitude = config['DATA']['max_longitude']
longitude_step_size = config['DATA']['longitude_step_size']
latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)

lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]

special_areas = {'lake_victoria': {'lat_range': [-3.05,1.05], 'lon_range': [31.05, 35.05]},
                 'nairobi': {'lat_range': [-1.55,-1.05], 'lon_range': [36.55, 37.05]},
                 'mombasa (coastal)': {'lat_range': [-4.15,-3.95], 'lon_range': [39.55, 39.85]},
                #  'addis ababa': {'lat_range': [8.8, 9.1], 'lon_range': [38.5, 38.9]},
                 'bale_mountains': {'lat_range': [6.65, 7.05], 'lon_range': [39.35, 40.25]},
                #  'Butembo / virunga (DRC)': {'lat_range': [-15.05, 0.55], 'lon_range': [29.05, 29.85]},
                 'Kampala': {'lat_range': [.05, 0.65], 'lon_range': [32.15, 32.95]},
                 'Nzoia basin': {'lat_range': [-0.35, 1.55], 'lon_range': [34.55, 36.55]}}

for k, v in special_areas.items():
    special_areas[k]['lat_index_range'] = [lat_range_list.index(v['lat_range'][0]), lat_range_list.index(v['lat_range'][1])]
    special_areas[k]['lon_index_range'] = [lon_range_list.index(v['lon_range'][0]), lon_range_list.index(v['lon_range'][1])]

# Quantiles
step_size = 0.001
range_dict = {0: {'start': 0.1, 'stop': 1, 'interval': 0.1, 'marker': '+', 'marker_size': 32},
              1: {'start': 1, 'stop': 10, 'interval': 1, 'marker': '+', 'marker_size': 256},
              2: {'start': 10, 'stop': 80, 'interval':10, 'marker': '+', 'marker_size': 512},
              3: {'start': 80, 'stop': 99.1, 'interval': 1, 'marker': '+', 'marker_size': 256},
              4: {'start': 99.1, 'stop': 99.91, 'interval': 0.1, 'marker': '+', 'marker_size': 128},
              5: {'start': 99.9, 'stop': 99.99, 'interval': 0.01, 'marker': '+', 'marker_size': 32 },
              6: {'start': 99.99, 'stop': 99.999, 'interval': 0.001, 'marker': '+', 'marker_size': 10},
              7: {'start': 99.999, 'stop': 99.9999, 'interval': 0.0001, 'marker': '+', 'marker_size': 10},
              8: {'start': 99.9999, 'stop': 99.99999, 'interval': 0.00001, 'marker': '+', 'marker_size': 10}}
                  
percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
percentiles=np.concatenate(percentiles_list)
quantile_locs = [np.round(item / 100.0, 6) for item in percentiles]

###########################

###########################

fps = glob('/user/work/uz22147/quantile_training_data/*_744.pkl')

imerg_train_data = []
ifs_train_data = []
training_dates = []
training_hours = []

for fp in fps:
    with open(fp, 'rb') as ifh:
        training_data = pickle.load(ifh)
        
    imerg_train_data.append(denormalise(training_data['obs']))
    ifs_train_data.append(denormalise(training_data['fcst_array']))

    training_dates += [item[0] for item in training_data['dates']]
    training_hours += [item[0] for item in training_data['hours']]

imerg_train_data = np.concatenate(imerg_train_data, axis=0)
ifs_train_data = np.concatenate(ifs_train_data, axis=0)


month_ranges = [[n for n in range(1,13)]]
quantile_threshold = 0.999

# identify best threshold and train on all the data
quantile_areas = get_quantile_areas(list(training_dates), month_ranges, latitude_range, longitude_range, hours=training_hours, num_lat_lon_chunks=1)
quantiles_by_area = get_quantiles_by_area(quantile_areas, fcst_data=ifs_train_data, obs_data=imerg_train_data, 
                                          quantile_locs=quantile_locs)

fcst_corrected = get_quantile_mapped_forecast(fcst=fcst_array, dates=training_dates, 
                                              hours=training_hours, month_ranges=month_ranges, 
                                              quantile_areas=quantile_areas, 
                                              quantiles_by_area=quantiles_by_area)

# Same for cgan

# NOTE:This requires data collection for the model 

cgan_training_sample_dict = {'cropped': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n10000_201603-201802_e1',
                             'cropped_4000': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n10000_201603-201802_e1'}
# model_number = 288000
model_number = get_best_model_number(log_folder=cgan_training_sample_dict[model_type])
with open(os.path.join(cgan_training_sample_dict[model_type], f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
imerg_training_data = arrays['truth']
cgan_training_data = arrays['samples_gen'][:,:,:,0]
training_dates = [d[0] for d in arrays['dates']]
training_hours = [h[0] for h in arrays['hours']]

# identify best threshold and train on all the data
quantile_locs = [np.round(item / 100.0, 10) for item in percentiles] + [1.0]

# identify best threshold and train on all the data
quantile_areas = get_quantile_areas(list(training_dates), month_ranges, latitude_range, longitude_range, 
                                    hours=training_hours, 
                                    num_lat_lon_chunks=1)
quantiles_by_area = get_quantiles_by_area(quantile_areas, fcst_data=cgan_training_data, 
                                          obs_data=imerg_training_data, 
                                          quantile_locs=quantile_locs)


cgan_corrected = get_quantile_mapped_forecast(fcst=samples_gen_array[:,:,:,0].copy(), dates=training_dates, 
                                              hours=training_hours, month_ranges=month_ranges, 
                                              quantile_areas=quantile_areas, 
                                              quantiles_by_area=quantiles_by_area)

with open(os.path.join(log_folder, f'qmapped_{model_type}_{model_number}.pkl'), 'wb+') as ofh:
    pickle.dump({'fcst_corrected': fcst_corrected,
                 'cgan_corrected': cgan_corrected}, ofh)