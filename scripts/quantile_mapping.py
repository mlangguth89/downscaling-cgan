"""
Script to quantile map the forecasts

Note that this requires data to be prepared from the training set, and so is currently
not in a fit state to be used without a proper pipeline being set up 

lbatch -t 10 -m 100 --queue short --conda-env base -a ID --array-range 1 10 --cmd python -m scripts.quantile_mapping --num-lat-lon-chunks ARRAY_ID --model-type cropped_4000 --output-folder plots/quantile_map_plots
"""

import pickle
import os, sys
import copy
import numpy as np
from pathlib import Path
from glob import glob
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
sys.path.insert(1, str(HOME / 'dsrnngan'))

from dsrnngan.data.data import denormalise, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE
from dsrnngan.evaluation.benchmarks import QuantileMapper
from dsrnngan.utils.utils import load_yaml_file, get_best_model_number
from dsrnngan.data.data import denormalise
from dsrnngan.evaluation.plots import plot_quantiles, quantile_locs

parser = ArgumentParser(description='Script for quantile mapping.')

parser.add_argument('--num-lat-lon-chunks', type=int, help='Number of chunks to split up spatial data into along each axis', default=1)
parser.add_argument('--model-type', type=str, help='Choice of model type')
parser.add_argument('--output-folder', type=str, help='Folder to save plots in')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--save-data', action='store_true', help='save data')
parser.add_argument('--plot', action='store_true', help='Make plots')
args = parser.parse_args()


###########################
# Load model data
###########################

print('Lading model data', flush=True)
log_folders = {'basic': '/user/work/uz22147/logs/cgan/d9b8e8059631e76f/n1000_201806-201905_e50',
               'full_image': '/user/work/uz22147/logs/cgan/43ae7be47e9a182e_full_image/n1000_201806-201905_e50',
               'cropped': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n2000_201806-201905_e20',
               'cropped_4000': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n4000_201806-201905_e10',
               'reweighted': '/user/work/uz22147/logs/cgan/de5750a9ef3bed6d/n3000_201806-201905_e10',
               'cropped_v2': '/user/work/uz22147/logs/cgan/f6998afe16c9f955/n4000_201806-201905_e10'}
# Get best model
log_folder = log_folders[args.model_type]

model_number = get_best_model_number(log_folder=log_folder)

log_folder = log_folders[args.model_type]
with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

if args.debug:
    n_samples = 100
else:
    n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth'][:n_samples, :, :]
samples_gen_array = arrays['samples_gen'][:n_samples, :,:,:]
fcst_array = arrays['fcst_array'][:n_samples, :,: ]
persisted_fcst_array = arrays['persisted_fcst'][:n_samples, :,: ]
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)[:n_samples, :,:]
dates = [d[0] for d in arrays['dates']][:n_samples]
hours = [h[0] for h in arrays['hours']][:n_samples]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
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

quantile_data_dicts = {'test': {
                    'GAN': {'data': samples_gen_array[:, :, :, 0], 'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'data': truth_array, 'color': 'k'},
                    'Fcst': {'data': fcst_array, 'color': 'r', 'marker': '+', 'alpha': 1},
                    'Fcst + qmap': {'data': None, 'color': 'r', 'marker': 'o', 'alpha': 0.7},
                    'GAN + qmap': {'data': None, 'color': 'b', 'marker': 'o', 'alpha': 0.7}
                    },
                    'train_fcst': {'Fcst': {'data': None, 'color': 'r', 'marker': '+', 'alpha': 1},
                                   'Obs (IMERG)': {'data': None, 'color': 'k'},
                                   'Fcst + qmap': {'data': None, 'color': 'r', 'marker': 'o', 'alpha': 0.7}},
                    'train_gan': {'GAN': {'data': None, 'color': 'r', 'marker': '+', 'alpha': 1},
                                  'Obs (IMERG)': {'data': None, 'color': 'k'},
                                  'GAN + qmap': {'data': None, 'color': 'b', 'marker': 'o', 'alpha': 0.7}}}

###########################
print('## Quantile mapping for IFS', flush=True)
###########################
fps = glob('/user/work/uz22147/quantile_training_data/*_744.pkl')

imerg_train_data = []
ifs_train_data = []
training_dates = []
training_hours = []

if args.debug:
    fps = fps[:3]

for fp in tqdm(fps, file=sys.stdout):
    with open(fp, 'rb') as ifh:
        training_data = pickle.load(ifh)
        
    imerg_train_data.append(denormalise(training_data['obs']))
    ifs_train_data.append(denormalise(training_data['fcst_array']))

    training_dates += [item[0] for item in training_data['dates']]
    training_hours += [item[0] for item in training_data['hours']]

imerg_train_data = np.concatenate(imerg_train_data, axis=0)
ifs_train_data = np.concatenate(ifs_train_data, axis=0)

quantile_data_dicts['train_fcst']['Fcst']['data'] = ifs_train_data
quantile_data_dicts['train_fcst']['Obs (IMERG)']['data'] = imerg_train_data

month_ranges = [[n for n in range(1,13)]]

qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, longitude_range=longitude_range, quantile_locs=quantile_locs,
                         num_lat_lon_chunks=args.num_lat_lon_chunks)
qmapper.train(fcst_data=ifs_train_data, obs_data=imerg_train_data, training_dates=training_dates, training_hours=training_hours)
quantile_data_dicts['test']['Fcst + qmap']['data'] = qmapper.get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, hours=hours)

# Evaluate on training set
quantile_data_dicts['train_fcst']['Fcst + qmap']['data'] = qmapper.get_quantile_mapped_forecast(fcst=ifs_train_data, dates=training_dates, hours=training_hours)


###########################
print('## Quantile mapping for GAN', flush=True)
###########################

# NOTE:This requires data collection for the model 

cgan_training_sample_dict = {'cropped': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n10000_201603-201802_e1',
                             'cropped_4000': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n10000_201603-201802_e1'}
# model_number = 288000
model_number = get_best_model_number(log_folder=cgan_training_sample_dict[args.model_type])
with open(os.path.join(cgan_training_sample_dict[args.model_type], f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
if args.debug:
    n_samples = 100
else:
    n_samples = arrays['truth'].shape[0]
    
imerg_training_data = arrays['truth'][:n_samples,:,:]
cgan_training_data = arrays['samples_gen'][:n_samples,:,:,0]
training_dates = [d[0] for d in arrays['dates']][:n_samples]
training_hours = [h[0] for h in arrays['hours']][:n_samples]

quantile_data_dicts['train_gan']['GAN']['data'] = cgan_training_data
quantile_data_dicts['train_gan']['Obs (IMERG)']['data'] = imerg_training_data

qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, longitude_range=longitude_range, quantile_locs=quantile_locs,
                         num_lat_lon_chunks=args.num_lat_lon_chunks)
qmapper.train(fcst_data=cgan_training_data, obs_data=imerg_training_data, training_dates=training_dates, training_hours=training_hours)
quantile_data_dicts['test']['GAN + qmap']['data'] = qmapper.get_quantile_mapped_forecast(fcst=samples_gen_array[:,:,:,0], dates=dates, hours=hours)

# Evaluate on training set
quantile_data_dicts['train_gan']['GAN + qmap']['data'] = qmapper.get_quantile_mapped_forecast(fcst=cgan_training_data, dates=training_dates, hours=training_hours)

###########################
print('### Saving data ', flush=True)
###########################

if args.save_data:
    with open(f'quantile_data_dicts_{args.num_lat_lon_chunks}.pkl', 'wb+') as ofh:
        pickle.dump(quantile_data_dicts, ofh)


###########################
print('## Creating plots', flush=True)
###########################

if args.plot:

    for data_type, quantile_data_dict in tqdm(quantile_data_dicts.items(), file=sys.stdout):
        # Overall q-q plot
        
        fcst_key = 'GAN' if 'GAN' in quantile_data_dict else 'Fcst'
        
        plot_quantiles(quantile_data_dict, 
                    save_path=os.path.join(args.output_folder, f'qq_plot_{data_type}_total_n{args.num_lat_lon_chunks}_total.pdf'))

        # Q-Q plot for areas
        fig, ax = plt.subplots(max(2, len(special_areas)),1, figsize=(10, len(special_areas)*10))
        fig.tight_layout(pad=4)
        for n, (area, area_range) in enumerate(special_areas.items()):


            lat_range = area_range['lat_index_range']
            lon_range = area_range['lon_index_range']
            
            local_quantile_data_dict = {}
            for k, v in quantile_data_dict.items():
                local_quantile_data_dict[k] = copy.deepcopy(v)
                local_quantile_data_dict[k]['data'] = local_quantile_data_dict[k]['data'][:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
            
            plot_quantiles(local_quantile_data_dict, ax=ax[n])
            ax[n].set_title(area)
        
        plt.savefig(os.path.join(args.output_folder, f'qq_plot_{data_type}_area_n{args.num_lat_lon_chunks}_total.pdf'), format='pdf')