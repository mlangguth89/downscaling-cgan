import pickle
import os, sys
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
from matplotlib import pyplot as plt


HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
from dsrnngan.utils.utils import load_yaml_file, get_best_model_number
from dsrnngan.utils.utils import bootstrap_summary_statistic
###########################
# Load model data
###########################

# Get best model
log_folder = '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n4000_201806-201905_e10'

model_number = get_best_model_number(log_folder=log_folder)

with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth']
samples_gen_array = arrays['samples_gen']
fcst_array = arrays['fcst_array']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
(n_samples, width, height, ensemble_size) = samples_gen_array.shape


# Open quantile mapped forecasts
with open(os.path.join(log_folder, f'fcst_qmap_270.pkl'), 'rb') as ifh:
    fcst_corrected = pickle.load(ifh)

with open(os.path.join(log_folder, f'cgan_qmap_15.pkl'), 'rb') as ifh:
    cgan_corrected = pickle.load(ifh)
    
    
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

special_areas = {'Lake Victoria': {'lat_range': [-3.05,1.05], 'lon_range': [31.05, 35.05]},
                 'Nairobi': {'lat_range': [-1.55,-1.05], 'lon_range': [36.55, 37.05]},
                 'Mombasa': {'lat_range': [-4.15,-3.95], 'lon_range': [39.55, 39.85]},
                #  'addis ababa': {'lat_range': [8.8, 9.1], 'lon_range': [38.5, 38.9]},
                 'Bale Mountains': {'lat_range': [6.65, 7.05], 'lon_range': [39.35, 40.25]},
                #  'Butembo / virunga (DRC)': {'lat_range': [-15.05, 0.55], 'lon_range': [29.05, 29.85]},
                 'Kampala': {'lat_range': [.05, 0.65], 'lon_range': [32.15, 32.95]},
                 'Nzoia basin': {'lat_range': [-0.35, 1.55], 'lon_range': [34.55, 36.55]}}

for k, v in special_areas.items():
    special_areas[k]['lat_index_range'] = [lat_range_list.index(v['lat_range'][0]), lat_range_list.index(v['lat_range'][1])]
    special_areas[k]['lon_index_range'] = [lon_range_list.index(v['lon_range'][0]), lon_range_list.index(v['lon_range'][1])]
    
    
    
# calculate standard deviation and mean of
quantile_locations = [np.round(1 - 10**(-n),n+1) for n in range(3, 9)]
def calculate_quantiles(input_array, quantile_locations=quantile_locations):
    
    return np.quantile(input_array, quantile_locations)
fcst_quantiles = calculate_quantiles(fcst_corrected)
cgan_quantiles = calculate_quantiles(cgan_corrected)
obs_quantiles = calculate_quantiles(truth_array)


n_bootstrap_samples = 1000

bootstrap_results_dict_obs = bootstrap_summary_statistic(calculate_quantiles, truth_array[0::2, ...], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)
bootstrap_results_dict_fcst_qmap = bootstrap_summary_statistic(calculate_quantiles, fcst_corrected[0::2, ...], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)
bootstrap_results_dict_cgan_qmap = bootstrap_summary_statistic(calculate_quantiles, cgan_corrected[0::2, ...], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)


fig, ax = plt.subplots(1,1)

ax.errorbar(obs_quantiles, fcst_quantiles, yerr=2*bootstrap_results_dict_fcst_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
ax.plot(obs_quantiles, obs_quantiles, '--')
ax.set_xlabel('Model (mm/hr)')

ax.set_ylabel('Observations (mm/hr)')
plt.savefig('quantile_bootstrap_intervals_fcst_total.pdf', format='pdf')

fig, ax = plt.subplots(1,1)

ax.errorbar(obs_quantiles, cgan_quantiles, yerr=2*bootstrap_results_dict_cgan_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
ax.plot(obs_quantiles, obs_quantiles, '--')
ax.set_xlabel('Model (mm/hr)')

ax.set_ylabel('Observations (mm/hr)')
plt.savefig('quantile_bootstrap_intervals_cgan_total.pdf', format='pdf')



# Same for all the areas
quantile_locations =  [np.round(1 - 10**(-n),n+1) for n in range(1, 7)]
n_cols = 2

fig, ax = plt.subplots(int(np.round(len(special_areas)/2)),2, figsize=(2*3, int(np.round(len(special_areas)/2))*3))
fig.tight_layout(pad=4)
for n, (area, area_range) in enumerate(special_areas.items()):
    row = int(n/2)
    column = n %2
    print(area)
    lat_range = area_range['lat_index_range']
    lon_range = area_range['lon_index_range']
    
    cgan_quantiles = calculate_quantiles(cgan_corrected[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)
    obs_quantiles = calculate_quantiles(truth_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)

    bootstrap_results_dict_obs = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), truth_array[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)
    bootstrap_results_dict_cgan_qmap = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), cgan_corrected[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)
    
    ax[row, column].errorbar(obs_quantiles, cgan_quantiles, yerr=2*bootstrap_results_dict_cgan_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
    ax[row, column].plot(obs_quantiles, obs_quantiles, '--')
    ax[row, column].set_xlabel('Observations (mm/hr)')
#     ax[row, column].set_xticks(obs_quantiles)
#     ax[row, column].set_xticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])

    ax[row, column].set_ylabel('Model (mm/hr)')

#     ax[row, column].set_yticks(cgan_quantiles)
#     ax[row, column].set_yticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])
    ax[row, column].set_title(area)
fig.tight_layout(pad=2.0)
plt.savefig('quantile_bootstrap_intervals_cgan_area.pdf', format='pdf')


# Same for all the areas
quantile_locations =  [np.round(1 - 10**(-n),n+1) for n in range(1, 7)]
n_cols = 2

fig, ax = plt.subplots(int(np.round(len(special_areas)/2)),2, figsize=(2*3, int(np.round(len(special_areas)/2))*3))
fig.tight_layout(pad=4)
for n, (area, area_range) in enumerate(special_areas.items()):
    row = int(n/2)
    column = n %2
    print(area)
    lat_range = area_range['lat_index_range']
    lon_range = area_range['lon_index_range']
    
    fcst_quantiles = calculate_quantiles(fcst_corrected[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)
    obs_quantiles = calculate_quantiles(truth_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)

    bootstrap_results_dict_obs = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), truth_array[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)
    bootstrap_results_dict_fcst_qmap = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), fcst_corrected[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=n_bootstrap_samples, time_resample=True)
    
    ax[row, column].errorbar(obs_quantiles, fcst_quantiles, yerr=2*bootstrap_results_dict_fcst_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
    ax[row, column].plot(obs_quantiles, obs_quantiles, '--')
    ax[row, column].set_xlabel('Observations (mm/hr)')
#     ax[row, column].set_xticks(obs_quantiles)
#     ax[row, column].set_xticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])

    ax[row, column].set_ylabel('Model (mm/hr)')

#     ax[row, column].set_yticks(cgan_quantiles)
#     ax[row, column].set_yticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])
    ax[row, column].set_title(area)
fig.tight_layout(pad=2.0)
plt.savefig('quantile_bootstrap_intervals_fcst_area.pdf', format='pdf')