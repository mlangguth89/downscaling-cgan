import pickle
import os, sys
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
from dsrnngan.utils.utils import bootstrap_metric_function, special_areas
from dsrnngan.utils import read_config
from dsrnngan.evaluation.scoring import fss
###########################
# Load model data
###########################

parser = ArgumentParser(description='Script for quantile mapping.')
parser.add_argument('--log-folder', type=str, help='model log folder', required=True)
parser.add_argument('--model-number', type=str, help='model number', required=True)
parser.add_argument('--n-bootstrap-samples', type=int, default=1000,
                    help='Number of bootstrap samples to use.')
parser.add_argument('--plot', action='store_true', help='Make plots')
parser.add_argument('--area', type=str, default='all', choices=list(special_areas.keys()), 
help="Area to run analysis on. Defaults to 'All' which performs analysis over the whole domain")
args = parser.parse_args()

# Get lat/lon range from log folder
base_folder = '/'.join(args.log_folder.split('/')[:-1])
data_config = read_config.read_data_config(config_folder=base_folder)
model_config = read_config.read_model_config(config_folder=base_folder)

# Locations
latitude_range, longitude_range=read_config.get_lat_lon_range_from_config(data_config=data_config)
lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]

special_areas['all']['lat_range'] = [min(latitude_range), max(latitude_range)]
special_areas['all']['lon_range'] =  [min(longitude_range), max(longitude_range)]

for k, v in special_areas.items():
    lat_vals = [lt for lt in lat_range_list if v['lat_range'][0] <= lt <= v['lat_range'][1]]
    lon_vals = [ln for ln in lon_range_list if v['lon_range'][0] <= ln <= v['lon_range'][1]]
    
    if lat_vals and lon_vals:
 
        special_areas[k]['lat_index_range'] = [lat_range_list.index(lat_vals[0]), lat_range_list.index(lat_vals[-1])]
        special_areas[k]['lon_index_range'] = [lon_range_list.index(lon_vals[0]), lon_range_list.index(lon_vals[-1])]

area = args.area
lat_range_index = special_areas[area]['lat_index_range']
lon_range_index = special_areas[area]['lon_index_range']
latitude_range=np.arange(special_areas[area]['lat_range'][0], special_areas[area]['lat_range'][-1] + data_config.latitude_step_size, data_config.latitude_step_size)
longitude_range=np.arange(special_areas[area]['lon_range'][0], special_areas[area]['lon_range'][-1] + data_config.longitude_step_size, data_config.longitude_step_size)

lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]



with open(os.path.join(args.log_folder, f'arrays-{args.model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth'][:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]
samples_gen_array = arrays['samples_gen'][:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_array = arrays['fcst_array'][:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1 ]
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)[:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]

dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
if len(samples_gen_array.shape) == 3:
    samples_gen_array = np.expand_dims(samples_gen_array, axis=-1)
(n_samples, width, height, ensemble_size) = samples_gen_array.shape


# Open quantile mapped forecasts
with open(os.path.join(args.log_folder, f'fcst_qmap_3.pkl'), 'rb') as ifh:
    fcst_corrected = pickle.load(ifh)

with open(os.path.join(args.log_folder, f'cgan_qmap_2.pkl'), 'rb') as ifh:
    cgan_corrected = pickle.load(ifh)

cgan_corrected = cgan_corrected[:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_corrected = fcst_corrected[:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]
    
###########################

###########################


quantile_thresholds = [0.9, 0.99, 0.999, 0.9999, 0.99999]
hourly_thresholds = [np.quantile(truth_array, q) for q in quantile_thresholds]
window_sizes = [1, 10, 20, 40, 60,80, 100, 150,200,300,400,500]
bootstrap_results_cgan_dict = {}
bootstrap_results_ifs_dict = {}

for n, thr in enumerate(hourly_thresholds):
    
    bootstrap_results_cgan_dict[quantile_thresholds[n]] = {}
    bootstrap_results_ifs_dict[quantile_thresholds[n]] = {}
    
    for ws in window_sizes:
        calculate_fss = lambda obs, fcst: fss(obs_array=obs, fcst_array=fcst, scale=ws, thr=thr, mode='constant')

        bootstrap_results_cgan_dict[quantile_thresholds[n]][ws] = bootstrap_metric_function(metric_func=calculate_fss, obs_array=truth_array, fcst_array=cgan_corrected[:,:,:,0], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        bootstrap_results_ifs_dict[quantile_thresholds[n]][ws] = bootstrap_metric_function(metric_func=calculate_fss, obs_array=truth_array, fcst_array=fcst_corrected, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)

# Save results 
with open(os.path.join(args.log_folder, f'bootstrap_fss_results_n{args.n_bootstrap_samples}_{args.area}.pkl'), 'wb+') as ofh:
    pickle.dump({'cgan': bootstrap_results_cgan_dict, 
                 'fcst': bootstrap_results_ifs_dict}, ofh)
