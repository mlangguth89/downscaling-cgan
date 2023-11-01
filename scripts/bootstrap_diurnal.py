import pickle
import os, sys
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
from dsrnngan.utils.utils import bootstrap_metric_function
from dsrnngan.utils import read_config
from dsrnngan.evaluation.scoring import fss
###########################
# Load model data
###########################

parser = ArgumentParser(description='Script for bootstrapping diurnal cucle.')
parser.add_argument('--log-folder', type=str, help='model log folder', required=True)
parser.add_argument('--model-number', type=str, help='model number', required=True)
parser.add_argument('--n-bootstrap-samples', type=int, default=1000,
                    help='Number of bootstrap samples to use.')
parser.add_argument('--plot', action='store_true', help='Make plots')
args = parser.parse_args()


with open(os.path.join(args.log_folder, f'arrays-{args.model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth']
fcst_array = arrays['fcst_array']
samples_gen_array = arrays['samples_gen']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
if len(samples_gen_array.shape) == 3:
    samples_gen_array = np.expand_dims(samples_gen_array, axis=-1)
(n_samples, width, height, ensemble_size) = samples_gen_array.shape
del fcst_array

# Open quantile mapped forecasts
with open(os.path.join(args.log_folder, f'fcst_qmap_15.pkl'), 'rb') as ifh:
    fcst_corrected = pickle.load(ifh)

with open(os.path.join(args.log_folder, f'cgan_qmap_1.pkl'), 'rb') as ifh:
    cgan_corrected = pickle.load(ifh)
    
    
###########################

###########################

# Get lat/lon range from log folder
base_folder = '/'.join(args.log_folder.split('/')[:-1])
data_config = read_config.read_data_config(config_folder=base_folder)
model_config = read_config.read_model_config(config_folder=base_folder)


# Locations
latitude_range, longitude_range=read_config.get_lat_lon_range_from_config(data_config=data_config)

lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]

metric_types = {'quantile_999': lambda obs_array,fcst_array: np.quantile(fcst_array, 0.999),
                    'quantile_9999': lambda obs_array,fcst_array: np.quantile(fcst_array, 0.9999),
                    'quantile_99999': lambda obs_array,fcst_array: np.quantile(fcst_array, 0.99999),
                'median': lambda obs_array,fcst_array: np.quantile(fcst_array, 0.5),
                'mean': lambda obs_array,fcst_array: np.mean(fcst_array)}

bootstrap_results_cgan_dict = {}
bootstrap_results_ifs_dict = {}
bootstrap_results_obs_dict = {}

bin_width = 3
hour_bin_edges = np.arange(0, 24, bin_width)

digitized_hours = np.digitize(hours, bins=hour_bin_edges)

for hr in range(24):
    digitized_hour = np.digitize(hr, bins=hour_bin_edges)
    
    bootstrap_results_cgan_dict[digitized_hour] = {}
    bootstrap_results_ifs_dict[digitized_hour] = {}
    bootstrap_results_obs_dict[digitized_hour] = {}

    hour_indexes = np.where(np.array(digitized_hours) == digitized_hour)[0]

    obs_for_hour = truth_array[hour_indexes,...]
    fcst_for_hour = fcst_corrected[hour_indexes,...]
    samples_for_hour = cgan_corrected[hour_indexes,:,:,0]
    
    for metric_type, metric_fn in metric_types.items():

        bootstrap_results_cgan_dict[digitized_hour][metric_type] = bootstrap_metric_function(metric_func=metric_fn, obs_array=samples_for_hour, fcst_array=samples_for_hour, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        bootstrap_results_ifs_dict[digitized_hour][metric_type] = bootstrap_metric_function(metric_func=metric_fn, obs_array=fcst_for_hour, fcst_array=fcst_for_hour, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        bootstrap_results_obs_dict[digitized_hour][metric_type] = bootstrap_metric_function(metric_func=metric_fn, obs_array=obs_for_hour, fcst_array=obs_for_hour, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)

# Save results 
with open(os.path.join(args.log_folder, f'bootstrap_diurnal_results_n{args.n_bootstrap_samples}.pkl'), 'wb+') as ofh:
    pickle.dump({'cgan': bootstrap_results_cgan_dict, 
                 'fcst': bootstrap_results_ifs_dict,
                 'obs': bootstrap_results_obs_dict}, ofh)
