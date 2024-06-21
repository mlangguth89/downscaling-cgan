import pickle
import os, sys
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
from dsrnngan.utils.utils import bootstrap_metric_function
from dsrnngan.utils import read_config
from dsrnngan.evaluation.scoring import equitable_threat_score, hit_rate, false_alarm_rate
###########################
# Load model data
###########################

parser = ArgumentParser(description='Script for bootstrapping equitable threat score.')
parser.add_argument('--log-folder', type=str, help='model log folder', required=True)
parser.add_argument('--model-number', type=str, help='model number', required=True)
parser.add_argument('--n-bootstrap-samples', type=int, default=1000,
                    help='Number of bootstrap samples to use.')
parser.add_argument('--plot', action='store_true', help='Make plots')
parser.add_argument('--debug', action='store_true', help='Debug mode')
args = parser.parse_args()


with open(os.path.join(args.log_folder, f'arrays-{args.model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

if args.debug:
    n_samples = 100
else:
    n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth'][:n_samples, ::]
samples_gen_array = arrays['samples_gen'][:n_samples, ::]
fcst_array = arrays['fcst_array'][:n_samples, ::]
persisted_fcst_array = arrays['persisted_fcst'][:n_samples, ::]
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)[:n_samples, ::]
dates = [d[0] for d in arrays['dates']][:n_samples]
hours = [h[0] for h in arrays['hours']][:n_samples]

del arrays

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
if len(samples_gen_array.shape) == 3:
    samples_gen_array = np.expand_dims(samples_gen_array, axis=-1)
(n_samples, width, height, ensemble_size) = samples_gen_array.shape


# Open quantile mapped forecasts
with open(os.path.join(args.log_folder, f'fcst_qmap_3.pkl'), 'rb') as ifh:
    fcst_corrected = pickle.load(ifh)[:n_samples, ::]

with open(os.path.join(args.log_folder, f'cgan_qmap_2.pkl'), 'rb') as ifh:
    cgan_corrected = pickle.load(ifh)[:n_samples, ::]
    
    
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

bootstrap_results_cgan_dict = {}
bootstrap_results_ifs_dict = {}

hourly_thresholds = [0.1, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50]

metric_lookup = {'far': false_alarm_rate, 'hitrate': hit_rate, 'ets': equitable_threat_score}

for metric_name, metric in metric_lookup.items():
    for n, threshold in enumerate(hourly_thresholds):

        bootstrap_results_cgan_dict[n] = {}
        bootstrap_results_ifs_dict[n] = {}

        metric_fn = lambda y_true, y_pred: metric(y_true, y_pred, threshold=threshold)

        bootstrap_results_cgan_dict[n] = bootstrap_metric_function(metric_func=metric_fn, obs_array=truth_array, fcst_array=cgan_corrected[...,0], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        bootstrap_results_ifs_dict[n] = bootstrap_metric_function(metric_func=metric_fn, obs_array=truth_array, fcst_array=fcst_corrected, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)

    # Save results 
    with open(os.path.join(args.log_folder, f'bootstrap_{metric_name}_results_n{args.n_bootstrap_samples}.pkl'), 'wb+') as ofh:
        pickle.dump({'cgan': bootstrap_results_cgan_dict, 
                    'fcst': bootstrap_results_ifs_dict,
                    'thresholds': hourly_thresholds}, ofh)
