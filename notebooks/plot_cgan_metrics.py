# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pickle
import os, sys
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec
from metpy import plots as metpy_plots
from matplotlib.colors import ListedColormap, BoundaryNorm
from string import ascii_lowercase

from zoneinfo import ZoneInfo
from datetime import datetime, timezone

HOME = Path(os.getcwd()).parents[0]
sys.path.insert(1, str(HOME))

os.environ['ESMFMKFILE'] = "/Users/bobbyantonio/anaconda3/envs/data_analysis/lib/esmf.mk"

# %%
from dsrnngan.evaluation.plots import plot_precipitation, plot_contourf, range_dict, quantile_locs, plot_quantiles
from dsrnngan.data.data import denormalise, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE
from dsrnngan.data import data
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.evaluation.rapsd import plot_spectrum1d, rapsd
from dsrnngan.evaluation.thresholded_ranks import findthresh
from dsrnngan.evaluation import scoring
from dsrnngan.utils.utils import get_best_model_number, load_yaml_file, special_areas, get_area_range
from dsrnngan.evaluation.benchmarks import QuantileMapper
from dsrnngan.utils import read_config

BASE_MODEL_PATH = '/Users/bobbyantonio/Documents/cGAN/cgan_model_data'


# %%
model_type = 'final-nologs-full'
area = 'all'

log_folders = {
               'final-nologs': {'log_folder': '7c4126e641f81ae0_medium-cl100-final-nologs/n4000_202010-202109_45682_e20', 'model_number': 217600},
               'final-nologs-full': {'log_folder': '7c4126e641f81ae0_medium-cl100-final-nologs/n8640_202010-202109_45682_e1', 'model_number': 217600},
               'final-nologs-mam2018': {'log_folder': 'n2088_201803-201805_f37bd_e1', 'model_number': 217600},
}

format_lookup = {'cGAN': {'color': 'b', 'marker': '+', 'alpha': 1, 'linestyle': '--'},
                'IMERG': {'color': 'k'},
                'IFS': {'color': 'r', 'marker': '+', 'alpha': 1, 'linestyle': '--'},
                'IFS-qm': {'color': 'r', 'marker': 'd', 'alpha': 0.7, 'linestyle': '--'},
                'cGAN-qm': {'color': 'b', 'marker': 'd', 'alpha': 0.7,'linestyle': '--'}}

# %%
# Get best model
if isinstance(log_folders[model_type], str):
    log_folder = os.path.join(BASE_MODEL_PATH, log_folders[model_type])

    # model_number = get_best_model_number(log_folder=log_folder)
    model_number=268800
else:
    
    model_number = log_folders[model_type]['model_number']
    log_folder = os.path.join(BASE_MODEL_PATH, log_folders[model_type]['log_folder'])

try:
    with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
        arrays = pickle.load(ifh)

    with open(os.path.join(log_folder, 'fcst_qmap_3.pkl'), 'rb') as ifh:
        fcst_corrected = pickle.load(ifh)
    with open(os.path.join(log_folder, 'cgan_qmap_2.pkl'), 'rb') as ifh:
        cgan_corrected = pickle.load(ifh)    
        
except FileNotFoundError:
    arrays = {}
    # Local data version
    with open(os.path.join(log_folder, f'truth-array-{model_number}.pkl'), 'rb') as ifh:
        arrays['truth'] = pickle.load(ifh)
    with open(os.path.join(log_folder, 'fcst_array.pkl'), 'rb') as ifh:
        arrays['fcst_array'] = pickle.load(ifh)
    with open(os.path.join(log_folder, f'samples_gen_0.pkl'), 'rb') as ifh:
        arrays['samples_gen'] = pickle.load(ifh)
        arrays['samples_gen'] = np.expand_dims(arrays['samples_gen'], axis=-1)
        
    with open(os.path.join(log_folder, f'dates.pkl'), 'rb') as ifh:
        arrays['dates'] = pickle.load(ifh)
    with open(os.path.join(log_folder, f'hours.pkl'), 'rb') as ifh:
        arrays['hours'] = pickle.load(ifh)

    with open(os.path.join(log_folder, 'fcst_qmap_3.pkl'), 'rb') as ifh:
        fcst_corrected = pickle.load(ifh)
    with open(os.path.join(log_folder, 'cgan_qmap_2_small.pkl'), 'rb') as ifh:
        cgan_corrected = pickle.load(ifh)
        cgan_corrected = np.expand_dims(cgan_corrected, axis=-1) 
         
folder_suffix = log_folder.split('/')[-1]



# %%
# Get lat/lon range from log folder
base_folder = '/'.join(log_folder.split('/')[:-1])
try:
    config = load_yaml_file(os.path.join(base_folder, 'setup_params.yaml'))
    model_config, data_config = read_config.get_config_objects(config)
except FileNotFoundError:
    data_config = read_config.read_data_config(config_folder=base_folder)

latitude_range, lat_range_index, longitude_range, lon_range_index = get_area_range(data_config, area=area)

# %%

n_samples =  arrays['truth'].shape[0]
# n_samples = 500
truth_array = arrays['truth'][:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]
samples_gen_array = arrays['samples_gen'][:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_array = arrays['fcst_array'][:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1 ]
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)[:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]
dates = [d[0] for d in arrays['dates']][:n_samples]
hours = [h[0] for h in arrays['hours']][:n_samples]

eat_datetimes = [datetime(d.year, d.month, d.day, hours[n]).replace(tzinfo=timezone.utc).astimezone(ZoneInfo('Africa/Nairobi')) for n,d in enumerate(dates)]
dates = [datetime(d.year, d.month, d.day) for d in eat_datetimes]
hours = [d.hour for d in eat_datetimes]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
(n_samples, width, height, ensemble_size) = samples_gen_array.shape


cgan_corrected = cgan_corrected[:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_corrected = fcst_corrected[:n_samples, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]


# %%
del arrays

# %% [markdown]
# ## Calculate frequency of exceedance for different thresholds

# %%
thr = 50


with open(os.path.join('/user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs/n18000_201603-202009_6f02b_e1', f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
folder_suffix = log_folder.split('/')[-1]
plot_dir = os.path.join('/user/home/uz22147/repos/downscaling-cgan/plots', folder_suffix)
full_year_truth_array = arrays['truth'][:, :, :]
del arrays


# %%
# Complicated version
complicated_daily_return_period = []
thresholds = range(1,51)
for thr in tqdm(thresholds):
    mask = np.sum(full_year_truth_array > thr, axis=0) > 0
    complicated_daily_return_period.append((full_year_truth_array.shape[0] / np.sum(full_year_truth_array > thr , axis=0)[mask].mean()) / 24)

# %%
# more straightforward version; number of times that at least one pixel exceeds threshold.
daily_return_period = []
thresholds = range(1,80)
for thr in tqdm(thresholds):
    daily_return_period.append( (full_year_truth_array.shape[0] / (np.sum(np.sum(full_year_truth_array > thr, axis=1), axis=1) > 0).sum()) / 24)

# %%
annotation_quantiles = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
annotation_quantile_vals = [np.quantile(full_year_truth_array, q ) for q in annotation_quantiles]

# %%
fig, ax = plt.subplots(1,1, figsize=(5,5))

ax.plot(thresholds, daily_return_period, linewidth=3)

max_val = max(daily_return_period)
for n, q_val in tqdm(enumerate(annotation_quantile_vals[2:])):
    ax.vlines(q_val, 0, max_val, linestyles='--')
    q_val_text = f'{np.round(float(annotation_quantiles[n+2])*100, 12)}th'.replace('.0th', 'th')
    ax.text(1.01*q_val,  max_val, q_val_text)

ax.set_ylabel('Average return period (days)')
ax.set_xlabel('Rainfall threshold (mm/hr)')
plt.savefig(os.path.join(plot_dir, 'return_periods_train_{area}.pdf'), format='pdf', bbox_inches ='tight')

# %%
os.path.join(plot_dir, f'return_periods_train_{area}.pdf')

# %% [markdown]
# ### Locations of high rainfall

# %%
from dsrnngan.evaluation.plots import get_geoaxes, border_feature, disputed_border_feature, lake_feature

for n, thr in enumerate(annotation_quantile_vals):
    fig, ax = get_geoaxes(1,1)
    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
                
    im = ax.imshow(np.sum(truth_array > thr, axis=0),
            interpolation='nearest',
            origin='lower',
            extent=[min(longitude_range), max(longitude_range), 
                    min(latitude_range), max(latitude_range)],
            transform=ccrs.PlateCarree(),
            alpha=0.8, cmap='Reds')
    ax.add_feature(border_feature)
    ax.add_feature(disputed_border_feature)
    ax.add_feature(lake_feature, alpha=0.4)
    print(annotation_quantiles[n])

# %% [markdown]
# ### Summary of data

# %%
# Summary statistics
array_dict = {'IFS': fcst_array,'IFS-qm': fcst_corrected,'iMERG': truth_array, 'cGAN': samples_gen_array, 'cGAN-qm': cgan_corrected}
for k, v in array_dict.items():
    
    print('*'*10)
    print(k)
    print('Mean = ', v.mean())
    print('Max = ', v.max())
    print('Min = ', v.min())

# %%
cgan_corrected = cgan_corrected[:n_samples,...]
fcst_corrected = fcst_corrected[:n_samples,...]

# %%
import xarray as xr

all_imerg_data = []
all_ifs_data = []

for year in tqdm(range(2003, 2018)):
    for month in range(1,13):

        imerg_ds = xr.open_dataarray(f'/user/home/uz22147/repos/rainfall_data/daily_imerg_rainfall_{month}_{year}.nc')

        imerg_data = imerg_ds.sel(lat=latitude_range, method='nearest').sel(lon=longitude_range, method='nearest').values

        for t in range(imerg_data.shape[0]):
            
            snapshot = imerg_data[t, :, :]
            
            all_imerg_data.append(snapshot)
            
all_imerg_data = np.stack(all_imerg_data, axis = 0)

daily_historical_avg = np.mean(all_imerg_data, axis=0)
daily_historical_std = np.std(all_imerg_data, axis=0)

hourly_historical_avg = daily_historical_avg / 24
hourly_historical_std = daily_historical_std / 24

# %%
from dsrnngan.data.data import DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE, all_ifs_fields
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
from datetime import datetime, timedelta


# Find dry and rainy days in sampled dataset
means = [(n, truth_array[n,:,:].mean()) for n in range(n_samples)]
sorted_means = sorted(means, key=lambda x: x[1])

n_extreme_hours = 50
# wet_day_indexes = [item[0] for item in sorted_means[-n_extreme_hours:]]
# dry_day_indexes = [item[0] for item in sorted_means[:n_extreme_hours]]

tp_index = all_ifs_fields.index('tp')

# plot configurations
levels = [0, 0.1, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150] # in units of log10
precip_cmap = ListedColormap(metpy_plots.ctables.colortables["precipitation"][:len(levels)-1], 'precipitation')
precip_norm = BoundaryNorm(levels, precip_cmap.N)

plt.rcParams.update({'font.size': 11})
spacing = 10
units = "Rain rate [mm h$^{-1}$]"
precip_levels=np.arange(0, 1.5, 0.15)

# indexes = np.random.choice(wet_day_indexes,10)
# indexes = np.random.choice(range(n_samples), 10)
# , datetime(2021,5,10,17,0,0)
all_datetimes = [date + timedelta(hours=hours[n]) for n, date in enumerate(dates)]
datetimes_to_plot = [datetime(2020,10,21,21,0,0), datetime(2020,12,17,5,0,0),datetime(2021,5,8,8,0,0), datetime(2021,2,19,17,0,0)]
indexes = [n for n, item in enumerate(all_datetimes) if item in datetimes_to_plot]


# + dry_day_indexes[:2]
num_cols = 5
num_samples = len(indexes)
num_rows = num_samples
# , figsize = (num_rows*4, num_cols*4)),
# gs = gridspec.GridSpec(spacing*num_rows+1, spacing*num_cols, wspace=0.5, hspace=2)
rows = [[f'cgan_sample_{n}', f'cgan_mean_{n}', f'imerg_{n}', f'ifs_{n}', f'ifs_qmap_{n}', 'cbar'] for n in range(num_rows)]

fig = plt.figure(constrained_layout=True, figsize=(2.5*num_cols, 3*num_rows))
gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                       width_ratios=[1]*(num_cols - 1) + [0.05],
                       height_ratios=[1]*(num_rows) + [0.05],
                       wspace=0.005)                      
for n in tqdm(range(len(indexes))):

    ix = indexes[n]
    img_gens = samples_gen_array[ix, :,:,:]
    truth = truth_array[ix,:,:]
    fcst = fcst_array[ix,:,:]
    fcst_corr = fcst_corrected[ix, :, :]
    date = dates[ix]
    hour = hours[ix]
    avg_img_gens = img_gens.mean(axis=-1)
    date_str = date.strftime('%d-%m-%Y') + f' {hour:02d}:00:00'
    
    # cGAN
    data_lookup = {'cgan_sample': {'data': img_gens[:,:,0], 'title': 'cGAN-qm'},
                   'cgan_mean': {'data': avg_img_gens, 'title': f'cGAN-qm sample average'},
                   'imerg' : {'title': f"IMERG: {date_str}", 'data': truth},
                   'ifs': {'data': fcst, 'title': 'IFS-qm'},
                #    'ifs_qmap': {'data': fcst_corr, 'title': 'IFS + qmap'}
                # rr, 'title': 'IFS qmap'}
                   }
    for col, (k, val) in enumerate(data_lookup.items()):
   
        ax = fig.add_subplot(gs[n, col], projection = ccrs.PlateCarree())
        im = plot_precipitation(ax, data=val['data'], title=val['title'], longitude_range=longitude_range, latitude_range=latitude_range)


# precip_cmap = ListedColormap(metpy_plots.ctables.colortables["precipitation"][:len(precip_levels)-1], 'precipitation')
# precip_norm = BoundaryNorm(precip_levels, precip_cmap.N)
cbar_ax = fig.add_subplot(gs[-1, :])
# cb = mpl.colorbar.Colorbar(axd['cbar'], cmap=precip_cmap, norm=precip_norm)
# cbar_ax = fig.add_axes([0.95, 0.0, 0.02, 0.95])
cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10,
                  )

# cb.ax.set_yticks(precip_levels)
# cb.ax.set_yticklabels(precip_levels)
# cb.ax.tick_params(axis='both', which='major')
cb.ax.set_xlabel("Precipitation (mm / hr)", loc='center')


plt.savefig(os.path.join(plot_dir, f'cGAN_samples_IFS_{model_type}_{model_number}_{area}.pdf'), format='pdf')

# %%
os.path.join(plot_dir, f'cGAN_samples_IFS_{model_type}_{model_number}_{area}.pdf')

# %% [markdown]
# # Rainfall scatter plot

# %%
# Rainfall averaged over whole domain
from scipy.stats import pearsonr

metric_types = {'quantile_999': lambda x: np.quantile(x, 0.999),
                'quantile_9999': lambda x: np.quantile(x, 0.9999),
                'quantile_99999': lambda x: np.quantile(x, 0.99999),
            'median': lambda x: np.quantile(x, 0.5),
            'mean': lambda x: np.mean(x)}
metric_name = 'mean'

scatter_results = {}
for metric_name, metric_fn in tqdm(metric_types.items()):
    if metric_name == 'mean':

        scatter_results[metric_name] = {}

        scatter_results[metric_name]['cGAN-qm'] = [metric_fn(cgan_corrected[n,:,:,0]) for n in range(cgan_corrected.shape[0])]
        scatter_results[metric_name]['IMERG'] = [metric_fn(truth_array[n,:,:]) for n in range(truth_array.shape[0])]
        scatter_results[metric_name]['IFS-qm'] = [metric_fn(fcst_corrected[n,:,:]) for n in range(fcst_corrected.shape[0])]
        
        print('cGAN Correlation: ', pearsonr(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['cGAN-qm']))
        print('IFS Correlation: ', pearsonr(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['IFS-qm']))

# %%
plt.rcParams.update({'font.size': 12})

for metric_name in ['mean']:
    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].scatter(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['cGAN-qm'] , marker='+', label='cGAN-qm')
    ax[0].set_title('cGAN-qm')
    ax[1].scatter(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['IFS-qm'], marker='+',label='IFS-qm')
    max_val = np.max(scatter_results[metric_name]['cGAN-qm']  + scatter_results[metric_name]['IMERG'] + scatter_results[metric_name]['IFS-qm'])
    ax[1].set_title('IFS-qm')
    
    
    for n, a in enumerate(ax):
        a.plot(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['IMERG'], 'k--')
        a.set_xlabel('IMERG observations (mm/hr)')
        a.set_ylabel('Forecasts (mm/hr)')
        a.set_xlim([0, max_val])
        a.set_ylim([0, max_val])
        
    print(metric_name)
    plt.savefig(os.path.join(plot_dir, f'scatter_{metric_name}_{model_type}_{model_number}_{area}.pdf'), format='pdf')

# %%
print(os.path.join(plot_dir, f'scatter_{metric_name}_{model_type}_{model_number}_{area}.pdf'))

# %%
from scipy.stats import binned_statistic

binned_cgan_mean, bin_edges, _ = binned_statistic(truth_array.flatten(), cgan_corrected[...,0].flatten(), statistic='mean', bins=100, range=None)
binned_fcst_mean, bin_edges, _ = binned_statistic(truth_array.flatten(), fcst_corrected.flatten(), statistic='mean', bins=100, range=None)
binned_obs_mean, bin_edges, _ = binned_statistic(truth_array.flatten(), truth_array.flatten(), statistic='mean', bins=100, range=None)

# %%
h = np.histogram(truth_array, 10)
bin_edges = h[1]
bin_edges[0] = 0
bin_centres = [0.5*(bin_edges[n] + bin_edges[n+1]) for n in range(len(bin_edges) - 1)]

mean_data = {}
data_dict = {'IFS': fcst_corrected,
             'cGAN': cgan_corrected}

for name, d in data_dict.items():
    
    mean_data[name] = []
    for n in range(len(bin_edges) - 1):
        valid_ix = np.logical_and(truth_array <=bin_edges[n+1], truth_array >= bin_edges[n])
        tmp_fcst_data = d[valid_ix]
        assert tmp_fcst_data.size > 0
        mean_data[name].append(tmp_fcst_data.mean())
        
fig, ax = plt.subplots(1,1)

for name, d in mean_data.items():
    ax.scatter(bin_centres, d, label=name)
ax.set_xlabel('Observations (mm/hr)')
ax.set_ylabel('Model (mm/hr)')
ax.legend()

# %% [markdown]
# # Reliability diagnostics

# %% [markdown]
# ### Spread error relationship

# %%
se_cgan_array_above_thr = []
for en in range(cgan_corrected.shape[-1]):
    se_cgan_array_above_thr.append(cgan_corrected[:n_samples,:,:,en][ensmean_array[:n_samples,:,:] > thr])
se_cgan_array_above_thr = np.stack(se_cgan_array_above_thr, axis=0)
se_obs_array_above_thr = truth_array[:n_samples,:,:][ensmean_array[:n_samples,:,:] > thr]


# %%
from dsrnngan.evaluation.scoring import get_spread_error_data

binned_mse, binned_spread = get_spread_error_data(n_samples=4000, observation_array=se_cgan_array_above_thr, ensemble_array=se_cgan_array_above_thr, 
                                                n_bins=100)
if np.isnan(binned_spread[0]):
    binned_spread = binned_spread[1:]
    binned_mse = binned_mse[1:]


# %%
from dsrnngan.evaluation.scoring import get_spread_error_data

binned_mse, binned_spread = get_spread_error_data(n_samples=4000, observation_array=truth_array, ensemble_array=samples_gen_array, 
                                                n_bins=100)
if np.isnan(binned_spread[0]):
    binned_spread = binned_spread[1:]
    binned_mse = binned_mse[1:]


# %%
ensemble_data_dir = '/user/home/uz22147/repos/downscaling-cgan/plots/n1000_202010-202109_45682_e100'
with open(os.path.join(ensemble_data_dir,'spread_error_cl100-medium-nologs_217600.pkl'), 'rb') as ifh:
    se_data = pickle.load(ifh)
binned_spread = se_data['binned_spread']
binned_mse = se_data['binned_mse']

# %%
fig, ax = plt.subplots(1,1, figsize=(5,5))
plt.rcParams.update({'font.size': 12})
ax.plot(np.sqrt(binned_spread), np.sqrt(binned_mse), marker='+', markersize=15)
ax.plot(np.linspace(0,max(binned_spread),4), 
        np.linspace(0,max(binned_spread),4), 'k--')
ax.set_ylabel('RMSE (mm/hr)')
ax.set_xlabel('RMS spread (mm/hr)')
ax.set_xlim([0,4])
ax.set_ylim([0,4])
ax.set_xticks(np.arange(0,4,1))
ax.set_yticks(np.arange(0,4,1))
plt.savefig(os.path.join(ensemble_data_dir, f'spread_error_{model_type}_{model_number}_{area}.pdf'), format='pdf')

# %%
os.path.join(ensemble_data_dir, f'spread_error_{model_type}_{model_number}_{area}.pdf')

# %% [markdown]
# ## Rank histogram

# %%
from pysteps.verification.ensscores import rankhist

# %%
n_samples = 1000

rank_cgan_array = []
for en in range(cgan_corrected.shape[-1]):
    rank_cgan_array.append(cgan_corrected[:n_samples,:,:,en][ensmean_array[:n_samples,:,:] < 0.1])
rank_cgan_array = np.stack(rank_cgan_array, axis=0)
rank_cgan_array = np.expand_dims(rank_cgan_array, -1)

rank_obs_array = np.expand_dims(truth_array[:n_samples,:,:][ensmean_array[:n_samples,:,:] < 0.1], axis=-1)

# %%
ranks = rankhist(X_f=rank_cgan_array, X_o=rank_obs_array, normalize=True)

# %%
ranks_ensemble_size = 100
with open(f'/user/home/uz22147/repos/downscaling-cgan/plots/n1000_202010-202109_45682_e{ranks_ensemble_size}/ranks_cl100-medium-nologs_217600.pkl', 'rb') as ifh:
    ranks = pickle.load(ifh)
    
with open(f'/user/home/uz22147/repos/downscaling-cgan/plots/n1000_202010-202109_45682_e{ranks_ensemble_size}/ranks_above_thr_cl100-medium-nologs_217600.pkl', 'rb') as ifh:
    ranks_above_thr = pickle.load(ifh)
    
with open(f'/user/home/uz22147/repos/downscaling-cgan/plots/n1000_202010-202109_45682_e{ranks_ensemble_size}/ranks_below_thr_cl100-medium-nologs_217600.pkl', 'rb') as ifh:
    ranks_below_thr = pickle.load(ifh)

# %%
from string import ascii_lowercase
ranks_dict = {'rank_hist': ranks,
              'rank_hist_above_thr': ranks_above_thr,
              'rank_hist_below_thr': ranks_below_thr}

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1,3, figsize=(12,4))
fig.tight_layout(pad=3)
for n, (name, rks) in enumerate(ranks_dict.items()):
       
       ax[n].bar(np.linspace(0,1,ranks_ensemble_size+1), rks, width=1/ranks_ensemble_size, 
              color='cadetblue', edgecolor='cadetblue')
       # ax.set_ylim([0,0.02])
       # ax.set_xlim([0-0.5/ranks_ensemble_size,1+0.5/ranks_ensemble_size])

       ax[n].hlines(1/ranks_ensemble_size, 0-0.5/ranks_ensemble_size,1+0.5/ranks_ensemble_size, linestyles='dashed', colors=['k'])
       ax[n].set_xlabel('Normalised rank')
       if n ==0:
              ax[n].set_ylabel('Normalised frequency')
       ax[n].set_yticks(np.arange(0,0.0305,0.005))
       ax[n].set_title(f'({ascii_lowercase[n]})')
fp = os.path.join('/user/home/uz22147/repos/downscaling-cgan/plots/n1000_202010-202109_45682_e100/',f'rank_hist_{model_type}_{model_number}_{area}.pdf')
plt.savefig(fp, format='pdf', bbox_inches='tight')

# %%
print(fp)

# %% [markdown]
# # RAPSD

# %%
from dsrnngan.evaluation.evaluation import calculate_ralsd_rmse

# %%
ralsd_rmse_all = []
for ix in tqdm(range(truth_array.shape[0])):
    ralsd_rmse = calculate_ralsd_rmse(truth_array[ix:ix+1, ...], samples_gen_array[ix:ix+1,..., 0])
    ralsd_rmse_all.append(ralsd_rmse.flatten())

# %%
from dsrnngan.evaluation.scoring import mse
ralsd_rmse_all = []
for ix in tqdm(range(truth_array.shape[0])):
    fft_freq_truth = rapsd(truth_array[ix,:,:], fft_method=np.fft)
    fft_freq_pred = rapsd(samples_gen_array[ix,:,:,0], fft_method=np.fft)
    
    ralsd_rmse_all.append(mse(fft_freq_truth, fft_freq_pred))

# %%
from dsrnngan.evaluation.rapsd import rapsd

rapsd_data_dict = {
                    'cGAN-qm': {'data': cgan_corrected[:, :, :, 0], 'color': 'b', 'linestyle': '-'},
                    'IMERG' : {'data': truth_array, 'color': 'k', 'linestyle': '-'},
                    'IFS-qm': {'data': fcst_corrected, 'color': 'r', 'linestyle': '-'},
                    }

rapsd_results = {}
for k, v in rapsd_data_dict.items():
        rapsd_results[k] = []
        for n in tqdm(range(n_samples)):
        
                fft_freq_pred = rapsd(v['data'][n,:,:], fft_method=np.fft)
                rapsd_results[k].append(fft_freq_pred)

        rapsd_results[k] = np.mean(np.stack(rapsd_results[k], axis=-1), axis=-1)

# %%
from dsrnngan.evaluation.rapsd import rapsd

rapsd_data_dict = {
                    'cGAN': {'data': samples_gen_array[:, :, :, 0], 'color': 'b', 'linestyle': '-'},
                    'IMERG' : {'data': truth_array, 'color': 'k', 'linestyle': '-'},
                    'IFS': {'data': fcst_array, 'color': 'r', 'linestyle': '-'},
                    }

rapsd_results = {}
for k, v in rapsd_data_dict.items():
        rapsd_results[k] = []
        for n in tqdm(range(n_samples)):
        
                fft_freq_pred = rapsd(v['data'][n,:,:], fft_method=np.fft)
                rapsd_results[k].append(fft_freq_pred)

        rapsd_results[k] = np.mean(np.stack(rapsd_results[k], axis=-1), axis=-1)

# %%
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(1,1, figsize=(8,8))
# np.arange(1,136) / (135*11),

for k, v in rapsd_results.items():
    ax.plot(np.arange(1,136) / (135*11), v , label=k, color=format_lookup[k]['color'], linestyle=format_lookup[k].get('linestyle', '-'), linewidth=3)
    # ax.plot(wavelengths, v , label=k, color=rapsd_data_dict[k]['color'], linestyle=rapsd_data_dict[k]['linestyle'], linewidth=3)
    
plt.xscale('log')
plt.yscale('log')
ax.set_ylabel('Power Spectral Density')
ax.set_xlabel('Wavenumber (1/km)')
ax.legend(frameon=False)

plt.savefig(os.path.join(plot_dir, f'rapsd_{model_type}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')

# %%
os.path.join(plot_dir, f'rapsd_{model_type}_{model_number}_{area}.pdf')

# %% [markdown]
# ## Quantiles

# %%
from dsrnngan.evaluation.plots import range_dict, get_quantile_data
quantile_data_dict = {
                    'cGAN': samples_gen_array[:, :, :, 0],
                    'cGAN-qm': cgan_corrected[:, :, :, 0],
                    'IMERG': truth_array,
                    'IFS': fcst_array,
                    'IFS-qm': fcst_corrected,
                    }
range_dict_truncated = range_dict.copy()
range_dict_truncated = {}


# %%

quantile_results, quantile_boundaries, intervals = get_quantile_data(quantile_data_dict,save_path=None,range_dict=range_dict)
with open(os.path.join(plot_dir,f'quantiles_total_{model_type}_{model_number}_{area}.pkl'), 'wb+') as ifh:
    pickle.dump(quantile_results, ifh)

# %%
with open(os.path.join(plot_dir,f'quantiles_total_{model_type}_{model_number}_{area}.pkl'), 'rb') as ifh:
    quantile_results= pickle.load(ifh)
    

quantile_results_truncated = {}
for k, v in quantile_results.items():
    quantile_results_truncated[k] = {}

    for ix, qs in v.items():
        if ix <8:
            quantile_results_truncated[k][ix] = qs


# %%
from itertools import chain

quantile_results_unravelled = {}
for name, data_dict in quantile_results_truncated.items():
    quantile_results_unravelled[name]=list(chain.from_iterable([v for v in quantile_results_truncated[name].values()]))

# %%
# try:
# Get bootstrap results
with open(os.path.join(log_folder, f'bootstrap_quantile_results_n1000_{area}.pkl'), 'rb') as ifh:
    bstrap_results = pickle.load(ifh)
    
bootstrap_results_dict_fcst_qmap = bstrap_results['fcst']
bootstrap_results_dict_obs = bstrap_results['obs']
bootstrap_results_dict_cgan_qmap = bstrap_results['cgan']

# calculate standard deviation and mean of
quantile_locations = [np.round(1 - 10**(-n),n+1) for n in range(3, 9)]
def calculate_quantiles(input_array, quantile_locations=quantile_locations):
    return np.quantile(input_array, quantile_locations)
fcst_quantiles = calculate_quantiles(fcst_corrected)
cgan_quantiles = calculate_quantiles(cgan_corrected[:,:,:,0])
obs_quantiles = calculate_quantiles(truth_array)
# except:
#     bootstrap_results_dict_fcst_qmap = None
#     bootstrap_results_dict_obs = None
#     bootstrap_results_dict_cgan_qmap = None

# %%
plot_hist = False
plt.rcParams.update({'font.size': 11})

if plot_hist:
   fig, ax = plt.subplots(1,2, figsize=(10,5))
else:
   fig, ax = plt.subplots(1,1, figsize=(5,5))
   ax = [ax]
max_truth_val = max(quantile_results_unravelled['IMERG'])
max_quantile = 0.9999999
quantile_annotation_dict = {str(np.round(q, 11)): np.quantile(quantile_data_dict['IMERG'], q) for q in [1 - 10**(-n) for n in range(3, 6)] if q <=max_quantile}

for data_name, results in quantile_results_unravelled.items():
   if data_name != 'IMERG':
      s = ax[0].scatter(quantile_results_unravelled['IMERG'], 
                                       results, 
                                       c=format_lookup[data_name]['color'],
                                       marker=format_lookup[data_name]['marker'], 
                                       label=data_name,
                                       s=25)
if bootstrap_results_dict_fcst_qmap is not None:
   ax[0].errorbar(obs_quantiles[:-2], fcst_quantiles[:-2], yerr=2*bootstrap_results_dict_fcst_qmap['std'][:-2], 
               xerr=2*bootstrap_results_dict_obs['std'][:-2], 
               capsize=2, ls='none', ecolor='r')
   ax[0].errorbar(obs_quantiles[:-2], cgan_quantiles[:-2], 
               yerr=2*bootstrap_results_dict_cgan_qmap['std'][:-2], 
               xerr=2*bootstrap_results_dict_obs['std'][:-2],
               capsize=2, ls='none', ecolor='b')         

ax[0].plot(np.linspace(0, max_truth_val, 100), np.linspace(0, max_truth_val, 100), 'k--')
ax[0].set_xlabel('Observations (mm/hr)')
ax[0].set_ylabel('Model (mm/hr)')
ax[0].legend(loc='lower right', frameon=False)

# find largest value
max_val = 0
for v in quantile_results_unravelled.values():
   if max(v) > max_val:
      max_val = max(v)
                
for k, v in quantile_annotation_dict.items():
   ax[0].vlines(v, 0, max_val, linestyles='--')
   ax[0].text(0.95*v,  max_val + 2, f'{np.round(float(k)*100, 12)}th')

if plot_hist:
   ###############################################################
   ### Histogram   
   from itertools import chain

   (q_99pt9, q_99pt99) = np.quantile(truth_array, [0.999, 0.9999])

   # plt.rcParams.update({'font.size': 20})

   bin_boundaries=np.arange(0,300,4)

   data_dict = {'IMERG': {'data': truth_array, 'histtype': 'stepfilled', 'alpha':0.6, 'facecolor': 'grey'}, 
               'IFS': {'data': fcst_array, 'histtype': 'step'},
               'IFS-qm': {'data': fcst_corrected, 'histtype': 'step',  'linestyle': '--'},
               'cGAN': {'data': samples_gen_array[:,:,:,0], 'histtype': 'step',},
                  'cGAN-qm': {'data': cgan_corrected[...,0], 'histtype': 'step', 'linestyle': '--'}}
   rainfall_amounts = {}

   for n, (name, d) in enumerate(data_dict.items()):
      
      ax[1].hist(d['data'].flatten(), bins=bin_boundaries, histtype=d['histtype'], label=name, alpha=d.get('alpha'),
                  facecolor=d.get('facecolor'), edgecolor=format_lookup[name]['color'], linestyle= d.get('linestyle'), linewidth=1.5)
      
      
   ax[1].set_yscale('log')
   ax[1].legend(frameon=False)
   ax[1].set_xlabel('Rainfall (mm/hr)')
   ax[1].set_ylabel('Frequency of occurence')
   ax[1].set_xlim([0,150])
   # ax.vlines(q_99pt9, 0, 10**8, linestyles='--')
   ax[1].vlines(q_99pt99, 0, 10**8, linestyles='--', linewidth=2)
   # ax.text(q_99pt9 - 5, 10**8 - 20, '$99.9^{th}$')
   ax[1].text(q_99pt99 +1 , 10**8, '$99.99^{th}$')
   # ax.set_ylabel('Total amount of rainfall in bin (mm/hr)')

   ax[0].set_title('(a)')
   ax[1].set_title('(b)')

plt.savefig(os.path.join(plot_dir, f'q-q_hist_{model_type}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')

# %%
print(os.path.join(plot_dir, f'q-q_hist_{model_type}_{model_number}_{area}.pdf'))

# %%
from itertools import chain

(q_99pt9, q_99pt99) = np.quantile(truth_array, [0.999, 0.9999])

plt.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(1,1, figsize=(12,10))
fig.tight_layout(pad=4)
bin_boundaries=np.arange(0,300,4)

data_dict = {'IMERG': {'data': truth_array, 'histtype': 'stepfilled', 'alpha':0.6, 'facecolor': 'grey'}, 
             'IFS': {'data': fcst_array, 'histtype': 'step', 'edgecolor': 'red'},
             'IFS-qm': {'data': fcst_corrected, 'histtype': 'step', 'edgecolor': 'red', 'linestyle': '--'},
             'cGAN': {'data': samples_gen_array[:,:,:,0], 'histtype': 'step', 'edgecolor': 'blue'},
                'cGAN-qm': {'data': cgan_corrected[...,0], 'histtype': 'step', 'edgecolor': 'blue', 'linestyle': '--'}}
rainfall_amounts = {}

edge_colours = ["blue", "green", "red", 'orange']
for n, (name, d) in enumerate(data_dict.items()):
    
    axs.hist(d['data'].flatten(), bins=bin_boundaries, histtype=d['histtype'], label=name, alpha=d.get('alpha'),
                facecolor=d.get('facecolor'), edgecolor=d.get('edgecolor'), linestyle= d.get('linestyle'), linewidth=3)
    
    
axs.set_yscale('log')
axs.legend()
axs.set_xlabel('Rainfall (mm/hr)')
axs.set_ylabel('Frequency of occurence')
axs.set_xlim([0,150])
# ax.vlines(q_99pt9, 0, 10**8, linestyles='--')
axs.vlines(q_99pt99, 0, 10**8, linestyles='--', linewidth=3)
# ax.text(q_99pt9 - 5, 10**8 - 20, '$99.9^{th}$')
axs.text(q_99pt99 + 7 , 10**7, '$99.99^{th}$')
# ax.set_ylabel('Total amount of rainfall in bin (mm/hr)')
plt.savefig(os.path.join(plot_dir, f'histograms_{model_type}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')


# %%
os.path.join(plot_dir, f'quantiles_total_{model_type}_{model_number}_{area}.pdf')

# %%

# %%
# Quantiles for different areas
import copy


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

# %% [markdown]
# ## Long term average and std for IMERG, plus quantile mapping
#

# %% [markdown]
# ## Bias and RMSE

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colorbar, colors, gridspec


# %%
# RMSE
rmse_dict = {'cgan': np.sqrt(np.mean(np.square(truth_array - samples_gen_array[:,:,:,0]), axis=0)),
            'cgan_qmap' : np.sqrt(np.mean(np.square(truth_array - cgan_corrected[:n_samples,:,:,0]), axis=0)),
            'fcst' : np.sqrt(np.mean(np.square(truth_array - fcst_array), axis=0)),
            'fcst_qmap' : np.sqrt(np.mean(np.square(truth_array - fcst_corrected[:n_samples,:,:]), axis=0))}

# RMSE
fig, ax = plt.subplots(len(rmse_dict.keys())+1,2, 
                       subplot_kw={'projection' : ccrs.PlateCarree()},
                       figsize=(12,16))


max_level = 10
value_range = list(np.arange(0, max_level, max_level / 50))

value_range_2 = list(np.arange(0, 5, 5 / 50))

for n, (k, v) in enumerate(rmse_dict.items()):
    
    im = plot_contourf(ax[n,0], v, title=k, value_range=value_range, lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n,0])
    ax[n,0].set_title(k)
    im2 = plot_contourf(ax[n,1], v / hourly_historical_std, title=k + ' / truth_std', value_range=value_range_2 , lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im2, ax=ax[n,1])

im = plot_contourf(ax[n+1,0], hourly_historical_std, title='truth_std', lat_range=latitude_range, lon_range=longitude_range)
plt.colorbar(im, ax=ax[n+1,0])

# plt.savefig(f'cgan_metrics/rmse_{model_type}_{model_number}_{area}.pdf', format='pdf')


# %%

# edge_len =5
# rainy_pixels =truth_array[:,edge_len:-edge_len, edge_len:-edge_len] > 1
# bias_dict = {'cGAN': np.mean(cgan_corrected[:,edge_len:-edge_len, edge_len:-edge_len,0][rainy_pixels] - truth_array[:,edge_len:-edge_len, edge_len:-edge_len][rainy_pixels], axis=0),
#             'IFS' : np.mean(fcst_corrected[:,edge_len:-edge_len, edge_len:-edge_len][rainy_pixels] - truth_array[:,edge_len:-edge_len, edge_len:-edge_len][rainy_pixels], axis=0)}


# %% [markdown]
# ### Bias

# %%
# overall average biases
bias_dict = {'cGAN-qm': np.mean(cgan_corrected[...,0] - truth_array, axis=0),
            'IFS-qm' : np.mean(fcst_corrected - truth_array, axis=0)}

bias_std_dict = {'cGAN-qm': np.std(cgan_corrected[...,0], axis=0) - np.std(truth_array,axis=0),
            'IFS-qm' : np.std(fcst_corrected, axis=0) - np.std(truth_array,axis=0)}

# %%
bias_summary_stats = {}
for k in bias_dict.keys():
    
    bias_summary_stats['rms_bias'] = np.sqrt(np.power(bias_dict[k],2).mean())
    bias_summary_stats['rms_std_bias'] = np.sqrt(np.power(bias_std_dict[k],2).mean())
print(bias_summary_stats)

# %%
lat_range=latitude_range
lon_range=longitude_range[:-1]
plt.rcParams.update({'font.size': 12})

num_rows = 2
num_cols =2

fig = plt.figure(constrained_layout=True, figsize=(1.5*2.5*num_cols, 1.5*3*num_rows))
gs = gridspec.GridSpec(num_rows + 2, num_cols, figure=fig, 
                       height_ratios=[1, 0.05, 1, 0.05],
                       wspace=0.005)   
bias_dict = {'cGAN-qm': np.mean(cgan_corrected[...,0] - truth_array, axis=0),
            'IFS-qm' : np.mean(fcst_corrected - truth_array, axis=0)}
val_range = list(np.arange(-0.75, 0.755, 0.05))

max_bias_val = max([v.max() for v in bias_dict.values()])
min_bias_val = min([v.min() for v in bias_dict.values()])

for col, (k,v) in enumerate(bias_dict.items()):

    ax = fig.add_subplot(gs[0, col], projection = ccrs.PlateCarree())

    value_range = list(np.arange(-0.5, 0.5, 0.01))

    #remove edges
    edge_len =5

    title = f'{k} ({np.sqrt(np.power(bias_dict[k],2).mean()):0.2f})'

    im = plot_contourf(v, 
                       title=title, 
                       ax=ax,
                       cmap='BrBG', 
                       value_range=val_range, 
                       lat_range=latitude_range,
                       lon_range=longitude_range # There is a bug somewhere in how the arrays are produced
                    #    lat_range=latitude_range[edge_len:-edge_len], 
                    #    lon_range=longitude_range[edge_len:-edge_len]
                       )
    lon_ticks = np.arange(25, 50,10)
    lat_ticks = np.arange(-10, 15,10)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
    ax.set_yticklabels([f"{np.abs(lt)}{'N' if lt >=0 else 'S'}" for lt in lat_ticks])
cbar_ax = fig.add_subplot(gs[1, :])
cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10,
                  )

cb.ax.set_xlabel("Average bias (mm/hr)", loc='center')

# Standard deviation bias
bias_std_dict = {'cGAN-qm': np.std(cgan_corrected[...,0], axis=0) - np.std(truth_array,axis=0),
            'IFS-qm' : np.std(fcst_corrected, axis=0) - np.std(truth_array,axis=0)}
for col, (k,v) in enumerate(bias_std_dict.items()):

    ax = fig.add_subplot(gs[2, col], projection = ccrs.PlateCarree())

    # value_range = list(np.arange(-0.5, 0.5, 0.01))
    value_range = None

    title = f'{k} ({np.sqrt(np.power(bias_std_dict[k],2).mean()):0.2f})'

    #remove edges
    edge_len =5
    im = plot_contourf(v, 
                       title=title, 
                       ax=ax,
                       cmap='BrBG', 
                       value_range=val_range, 
                       lat_range=latitude_range,
                       lon_range=longitude_range

                       )
    lon_ticks = np.arange(25, 50,10)
    lat_ticks = np.arange(-10, 15,10)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
    ax.set_yticklabels([f"{np.abs(lt)}{'N' if lt >=0 else 'S'}" for lt in lat_ticks])
    
cbar_ax = fig.add_subplot(gs[-1, :])
cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10,
                  )

cb.ax.set_xlabel("Standard Deviation Bias (mm/hr)", loc='center')

plot_filename = os.path.join(plot_dir, f'bias_{model_type}{model_number}_{area}.pdf')
plt.savefig(plot_filename, format='pdf', bbox_inches='tight')

# %%
print(os.path.join(plot_dir, f'bias_{model_type}{model_number}_{area}.pdf'))

# %% [markdown]
# # Fractions skill score

# %%
# bootstrapping results
with open(os.path.join(log_folder, f'bootstrap_fss_results_n100_{area}.pkl'), 'rb') as ifh:
    fss_bstrap_results = pickle.load(ifh)

# %%
fss_bstrap_results.keys()

# %%
quantile_locs = list(fss_bstrap_results['cgan'].keys())
window_sizes = list(fss_bstrap_results['cgan'][0.9].keys())

# %%
#evaluate percentage of rainy pixels based on threshold
daily_thresholds = range(1,40)
num_rainy_pixels = []
for thr in daily_thresholds:
    num_rainy_pixels.append((truth_array >= thr/24.0).astype(np.single).mean())
plt.plot(daily_thresholds, num_rainy_pixels)

# %%
from dsrnngan.evaluation.evaluation import get_fss_scores

scales = [2*w+1 for w in window_sizes]
n_samples = 4000
fss_data_dict = {
                    'cGAN': cgan_corrected[:n_samples, :, :, 0],
                    'IFS': fcst_corrected[:n_samples, :, :],
                    'IFS_raw': fcst_array[:n_samples, :, :]}

# get quantiles
hourly_thresholds = np.quantile(truth_array, quantile_locs)
fss_results = get_fss_scores(truth_array, fss_data_dict, hourly_thresholds, scales, n_samples)


# %%
# Save results 
with open(f'fss_{model_type}_{model_number}_{area}.pkl', 'wb+') as ofh:
    pickle.dump(fss_results, ofh)

# %%
with open(os.path.join(plot_dir, 'fss_cl100-medium-nologs_217600.pkl'), 'rb') as ifh:
    fss_results = pickle.load(ifh)

# %%
from string import ascii_lowercase

plt.rcParams.update({'font.size': 11})


linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1,10))]
n=0
# fig, ax = plt.subplots(2,3, figsize = (10,6))
# fig.tight_layout(pad=3)

num_cols = 3
num_rows = 2
fig = plt.figure(constrained_layout=True, figsize=(4*num_cols, 4*num_rows))
gs = gridspec.GridSpec(num_rows, 2*num_cols, figure=fig, 
                wspace=0.005)  
    

fss_format_lookup = {'fcst': {'label': 'IFS-qm', 'color': 'r'}, 
                     'cgan':  {'label': 'cGAN-qm', 'color': 'b'},
                     'cgan_raw': {'label': 'cGAN', 'color': 'b', 'linestyle': '--'},
                     'fcst_raw': {'label': 'IFS', 'color': 'r', 'linestyle': '--'}}


for n, thr in enumerate(fss_bstrap_results['cgan'].keys()):
    
    col = n%3
    row = int(n/3)
    
    if row == 1:
        ax = fig.add_subplot(gs[row, 2*col+1:2*col+3])
    else:
        ax = fig.add_subplot(gs[row, 2*col:2*col+2])
    q_val_text = f'{np.round(float(thr)*100, 12)}th'.replace('.0th', 'th')

    
    for m, (name) in enumerate(fss_bstrap_results.keys()):
        label = fss_format_lookup[name]['label']
            

        bstrap_data = fss_bstrap_results[name][thr]
        window_sizes = list(bstrap_data.keys())
        window_sizes_km = 11* np.array(window_sizes)
        mean_fss_values = [item['mean'] for item in bstrap_data.values()]
        std_fss_values = np.array([item['std'] for item in bstrap_data.values()])

        # ax.plot(window_sizes, fss_values, label=label, color='k', linestyle=linestyles[m])
        # ax.errorbar(window_sizes_km, mean_fss_values, yerr=2*std_fss_values, 
        #                 color='k', ls=linestyles[m], capsize=2, label=label)
        ax.plot(window_sizes_km, mean_fss_values, 'o', 
                   color=fss_format_lookup[name]['color'], 
                   label=fss_format_lookup[name]['label'], 
                   markersize=5,
                   linestyle=fss_format_lookup[name].get('linestyle', '-'))
        
        if name in ['cgan', 'fcst']:
            ax.fill_between(window_sizes_km, y1=mean_fss_values-2*std_fss_values, y2=mean_fss_values+2*std_fss_values,
                                color=fss_format_lookup[name].get('color', 'black'), alpha=0.2)
    
    # ax.plot(window_sizes_km, mean_fss_values, '-o', 
    #                color=fss_format_lookup[name]['color'], 
    #                label=fss_format_lookup[name]['label'], 
    #                markersize=5)
    ax.set_title(f'({ascii_lowercase[n]}) {q_val_text}')

    # ax.hlines(0.5, 0, 1 + 11*max(fss_bstrap_results[name][thr].keys()), linestyles='dashed', colors=['r'])
    ax.set_ylim(0,1)
    ax.set_xlabel('Neighbourhood width (km)')
    ax.set_ylabel('FSS')
    ax.set_xlim(0, 1 + 11*max(fss_bstrap_results[name][thr].keys()))
    if thr in [0.999, 0.99999, 0.9999]:
        ax.legend(loc='upper right', frameon=False, ncol=2)
    else:
        ax.legend(loc='lower right', frameon=False, ncol=2)
    # else:
        # ax.legend(loc='upper left')
plot_filename = os.path.join(plot_dir, f'fss_{model_type}_{model_number}_{area}.pdf')
print(plot_filename)
plt.savefig(plot_filename, format='pdf', bbox_inches='tight')

# %%
plot_filename

# %%
plt.rcParams.update({'font.size': 11})
# fig.tight_layout(pad=4.0)

linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1,10))]
n=0
for n, thr in enumerate(fss_results['quantile_thresholds']):
        
    fig, ax = plt.subplots(1, 1, figsize = (3,3))
    
    q_val_text = f'{np.round(float(thr)*100, 12)}th'.replace('.0th', 'th')

    
    for m, (name, scores) in enumerate(fss_results['results']['scores'].items()):
        label = 'cGAN-qm' if name == 'cgan' else 'IFS-qm'
        ax.plot(fss_results['results']['window_sizes'], scores[n], label=label, color='k', linestyle=linestyles[m])
        

    ax.set_title(q_val_text)

    ax.hlines(0.5, 0, max(fss_results['results']['window_sizes']), linestyles='dashed', colors=['r'])
    ax.set_ylim(0,1)
    ax.set_xlabel('Neighbourhood size (grid cell units)')
    ax.set_ylabel('FSS')
    ax.set_xlim(0, max(fss_results['results']['window_sizes']))
    if thr == 0.99999:
        ax.legend(loc='upper right')
    else:
        ax.legend(loc='lower right')
    # else:
        # ax.legend(loc='upper left')
        
    # plt.savefig(os.path.join(plot_dir, f'fss_q{q_val_text}_{model_type}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')

# %%
# FSS for regions
window_sizes = list(range(1,11)) + [20, 40, 60, 80, 100] + [150, 200]
n_samples = 10


# get quantiles
quantile_locs = [0.5, 0.985, 0.999, 0.9999]
fss_area_results = {}
for n, (area, area_range) in enumerate(special_areas.items()):
    
    lat_range_ends = area_range['lat_range']
    lon_range_ends = area_range['lon_range']
    lat_range_index = area_range['lat_index_range']
    lon_range_index = area_range['lon_index_range']
    lat_range = np.arange(lat_range_ends[0], lat_range_ends[-1]+0.0001, 0.1)
    lon_range = np.arange(lon_range_ends[0], lon_range_ends[-1]+0.0001, 0.1)
    
    area_truth_array = truth_array[:,lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1]]
    fss_data_dict = {
                    'cgan': samples_gen_array[:n_samples, lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1], 0],
                    'ifs': fcst_array[:n_samples,lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1]],
                    'fcst_qmap': fcst_corrected[:n_samples, lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1]]}  
    fss_area_results[area] = get_fss_scores(area_truth_array, fss_data_dict, quantile_locs, window_sizes, n_samples)
    
    plot_fss_scores(fss_results=fss_area_results[area], output_folder='plots', output_suffix=f'{area}_{model_type}_{model_number}')


# %% [markdown]
# ### FSS at grid cells

# %%
from dsrnngan.evaluation.scoring import get_metric_by_hour, get_metric_by_grid_cell, fss

threshold = 5
neighbourhood_size = 100
bin_width=3
area = 'All'
area_range = special_areas[area]
lat_range_ends = area_range['lat_range']
lon_range_ends = area_range['lon_range']
lat_range_index = [area_range['lat_index_range'][0], area_range['lat_index_range'][-1] +1]
lon_range_index = [area_range['lon_index_range'][0], area_range['lon_index_range'][-1] +1]
lat_range = np.arange(lat_range_ends[0], lat_range_ends[-1]+0.001, 0.1)
lon_range = np.arange(lon_range_ends[0], lon_range_ends[-1]+0.001, 0.1)

area_truth_array = truth_array[:,lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1]]

fss_fn = lambda y_true, y_pred : fss(obs_array=np.expand_dims(np.expand_dims(y_true, axis=0), axis=-1), fcst_array=np.expand_dims(np.expand_dims(y_pred, axis=0), axis=-1), scale=neighbourhood_size, thr=threshold)
metric_fn = lambda y_true, y_pred: get_metric_by_grid_cell(metric_fn=fss_fn, y_true=y_true, y_pred=y_pred)
fcst_fss_by_hour, hour_bin_edges = get_metric_by_hour(metric_fn=metric_fn, obs_array=truth_array[:,lat_range_index[0]:lat_range_index[-1]+1, lon_range_index[0]:lon_range_index[-1]+1], 
                                                      fcst_array=fcst_corrected[:,lat_range_index[0]:lat_range_index[-1]+1, lon_range_index[0]:lon_range_index[-1]+1], hours=hours, bin_width=bin_width)
gan_fss_by_hour, _ = get_metric_by_hour(metric_fn=metric_fn, obs_array=truth_array[:,lat_range_index[0]:lat_range_index[-1]+1, 
                                                                                   lon_range_index[0]:lon_range_index[-1]+1], 
                                        fcst_array=cgan_corrected[:,lat_range_index[0]:lat_range_index[-1]+1, lon_range_index[0]:lon_range_index[-1]+1,0], hours=hours, bin_width=bin_width)

# %%
from dsrnngan.evaluation.plots import plot_contourf
from scipy.ndimage import uniform_filter
n_rows = 2
n_cols = int(np.ceil(len(fcst_fss_by_hour)/n_rows))

fig = plt.figure(constrained_layout=True, figsize=(2.5*n_cols, 3*n_rows))
gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, 
                    width_ratios=[1]*(n_cols ),
                    height_ratios=[1]*(n_rows) + [0.05],
                wspace=0.005)  
    
digitized_hours = np.digitize(hours, bins=hour_bin_edges)

for n, hour_ix in enumerate(fcst_fss_by_hour):
    row = int(n / n_cols)
    column = n % n_cols
    ax = fig.add_subplot(gs[row, column], projection = ccrs.PlateCarree())

    hour_indexes = np.where(np.array(digitized_hours) == hour_ix)[0]

    if n == len(hour_bin_edges)-1:
        hr_end = 24
    else:
        hr_end = hour_bin_edges[n+1]-1
    bias = gan_fss_by_hour[hour_ix] -fcst_fss_by_hour[hour_ix]
    smoothed_bias = uniform_filter(np.nan_to_num(bias,0), size=21, mode='reflect')
    
    im = plot_contourf(bias, f"{hour_bin_edges[n]:02d}-{hr_end:02d}h EAT", ax, value_range=np.arange(-0.1,0.11,0.01), 
                lon_range=lon_range, 
                lat_range=lat_range,
                    cmap='RdBu', extend='both', add_borders=False)


fig.suptitle(f'(neighbourhood_size = {neighbourhood_size}, threshold={threshold})')
cbar_ax = fig.add_subplot(gs[-1, :])
cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10,
                )
cb.ax.set_xlabel(f"FSS(GAN) - FSS(IFS)", 
                 loc='center')
# plt.savefig(f'plots/fss_by_hour_thr{threshold}_{area}_{model_type}_{model_number}_{area}.pdf', format='pdf')

# %%
from dsrnngan.evaluation.scoring import get_filtered_array


data_dict = {'imerg' : truth_array, 'cgan': samples_gen_array[:, :, :, 0], 'ifs': fcst_array
             }
mode = 'constant'

for thr_index in range(len(hourly_thresholds)):

    thr = hourly_thresholds[thr_index] # 0 = median

    arrays_filtered = {}

    for size in window_sizes:

        arrays_filtered[size] = {}

        for k, d in tqdm(data_dict.items()):  
            arrays_filtered[size][k] = []
            for n in range(truth_array.shape[0]):
                # Convert to binary fields with the given intensity threshold
                I = (d >= thr).astype(np.single)

                # Compute fractions of pixels above the threshold within a square
                # neighboring area by applying a 2D moving average to the binary fields        
                arrays_filtered[size][k].append(get_filtered_array(int_array=I, mode=mode, size=size))

        for k in arrays_filtered[size]:
            arrays_filtered[size][k] = np.stack(arrays_filtered[size][k])

with open(f'fss_grid_{model_type}_{model_number}_thr{thr_index}.pkl', 'wb+') as ofh:
    pickle.dump(arrays_filtered, ofh)    


# %%
# plot MSE of fractions
n = 5
size = window_sizes[n]
bias = {}
for k in data_dict:
    
    if k == 'imerg':
        pass
    
    else:
        arr = arrays_filtered[size][k]
        bias[k] = np.mean(arr - arrays_filtered[size]['imerg'], axis=0)

fig, ax = plt.subplots(len(bias), 1, figsize=(15,15), subplot_kw={'projection' : ccrs.PlateCarree()})

for n, (k, v) in enumerate(bias.items()):
    im = plot_contourf(ax[n], v, title=k, lat_range=latitude_range, value_range=np.arange(-1, 1, 0.01), lon_range=longitude_range,cmap='RdBu')
    plt.colorbar(im, ax=ax[n])

# %% [markdown]
# ## Diurnal cycle

# %%
with open(os.path.join(plot_dir, '/user/home/uz22147/repos/downscaling-cgan/plots/n4000_202010-202109_45682_e20/diurnal_cycle_cl100-medium-nologs_217600.pkl'), 'rb') as ifh:
    diurnal_results = pickle.load(ifh)

with open(os.path.join(log_folder, 'bootstrap_diurnal_results_n100.pkl'), 'rb') as ifh:
    raw_diurnal_bstrap_results = pickle.load(ifh)

raw_diurnal_bstrap_results = {'cGAN': raw_diurnal_bstrap_results['cgan'], 'IFS': raw_diurnal_bstrap_results['fcst'], 'Obs (IMERG)': raw_diurnal_bstrap_results['obs'] }

# %%
lsm = data.load_land_sea_mask(
                       latitude_vals=latitude_range, 
                       longitude_vals=longitude_range)
# lake victoria lsm
lv_latitude_range, lv_lat_range_index, lv_longitude_range, lv_lon_range_index = get_area_range(data_config, area='lake_victoria')
lv_lsm = lsm.copy()
lv_lsm = 1 - lv_lsm
lv_lsm[:lv_lat_range_index[0],:] = 0
lv_lsm[lv_lat_range_index[1]:,:] = 0
lv_lsm[:,:lv_lon_range_index[0]] = 0
lv_lsm[:,lv_lon_range_index[1]:] = 0

lv_area = np.ones(lsm.shape)
lv_area[:lv_lat_range_index[0],:] = 0
lv_area[lv_lat_range_index[1]:,:] = 0
lv_area[:,:lv_lon_range_index[0]] = 0
lv_area[:,lv_lon_range_index[1]:] = 0

lsm_ocean = lsm.copy()
lsm_ocean[lv_lsm>0] = 1
lsm_ocean[:, :140] = 1

# %%
rearranged_diurnal_bstrap_results = {}
methods = list(raw_diurnal_bstrap_results['cGAN'][1].keys())
indexes =  list(raw_diurnal_bstrap_results['cGAN'].keys())

for method in methods:
    rearranged_diurnal_bstrap_results[method] = {}
    for data_type in raw_diurnal_bstrap_results:
        rearranged_diurnal_bstrap_results[method][data_type] = {}
        for ix in indexes:
            rearranged_diurnal_bstrap_results[method][data_type][ix] =  raw_diurnal_bstrap_results[data_type][ix][method]

diurnal_bstrap_results = rearranged_diurnal_bstrap_results

# %%
diurnal_bstrap_results.keys()

# %%
from string import ascii_lowercase

format_lkp = {'cGAN': {'color': 'b', 'label': 'cGAN-qm'}, 'IFS': {'color': 'r', 'label': 'IFS-qm'}, 'Obs (IMERG)': {'label': 'IMERG'}}
plt.rcParams.update({'font.size': 15})

x_vals = list(diurnal_bstrap_results['mean']['IFS'].keys())
bin_width = 24 / len(x_vals)
hour_bin_edges = np.arange(0, 24, bin_width)
fig, ax = plt.subplots(1,3, figsize=(15,5))
fig.tight_layout(pad=1)
name_lookup = {'mean': 'Mean', 'quantile_999': '99.9th percentile', 'quantile_9999': '99.99th percentile', 'quantile_99999': '99.999th percentile' }

for n, metric_name in enumerate(['mean', 'quantile_999', 'quantile_9999']):
    diurnal_dict = diurnal_bstrap_results[metric_name]
    

    for name, arr in diurnal_dict.items():
        label=format_lkp[name]['label']

        mean_vals = [item['mean'] for item in arr.values()]
        std_error_vals = np.array([item['std'] for item in arr.values()])
        ax[n].plot(x_vals, mean_vals, '-o', label=label, color=format_lkp[name].get('color', 'black'))
        ax[n].fill_between(x_vals, y1=mean_vals-2*std_error_vals, y2=mean_vals+2*std_error_vals,
                        color=format_lkp[name].get('color', 'black'), alpha=0.2)
        
    max_val = np.max(diurnal_dict['IFS'])
    ax[n].set_xticks(np.array(x_vals) - .5)
    ax[n].set_xticklabels([int(hr) for hr in hour_bin_edges])
    ax[n].set_xlabel('Hour')
    ax[n].set_ylabel('Average mm/hr')
    ax[n].set_ylim([0, None])
    ax[n].legend(loc='lower right')
    ax[n].set_title(f'({ascii_lowercase[n]}) ' + name_lookup[metric_name])

plt.savefig(os.path.join(plot_dir,f'diurnal_cycle_{model_type}_{model_number}.pdf'), bbox_inches='tight')

# %%
print(os.path.join(plot_dir,f'diurnal_cycle_{model_type}_{model_number}.pdf'))

# %%
from string import ascii_lowercase

format_lkp = {'cGAN': {'color': 'b', 'label': 'cGAN-qm'}, 'IFS': {'color': 'r', 'label': 'IFS-qm'}, 'Obs (IMERG)': {'label': 'IMERG'}}
plt.rcParams.update({'font.size': 15})

x_vals = list(diurnal_bstrap_results['mean']['IFS'].keys())
bin_width = 24 / len(x_vals)
hour_bin_edges = np.arange(0, 24, bin_width)
fig, ax = plt.subplots(nrows=1,ncols=4, figsize=(20,5))
fig.tight_layout(pad=1)
name_lookup = {'mean': 'Mean', 
               'quantile_999': '99.9th percentile', 
               'quantile_9999': '99.99th percentile', 
               'quantile_99999': '99.999th percentile',
               'mean_land': 'Mean Land', 
               'mean_ocean': 'Mean Ocean', 
               'mean_lv_lake': 'Mean LV (Lake Only)', 
               'mean_lv_area': 'Mean LV (Region)'}

for n, metric_name in enumerate(['mean_land', 'mean_ocean', 'mean_lv_lake', 'mean_lv_area']):
    diurnal_dict = diurnal_bstrap_results[metric_name]
    

    for name, arr in diurnal_dict.items():
        label=format_lkp[name]['label']

        mean_vals = [item['mean'] for item in arr.values()]
        std_error_vals = np.array([item['std'] for item in arr.values()])
        ax[n].plot(x_vals, mean_vals, '-o', label=label, color=format_lkp[name].get('color', 'black'))
        ax[n].fill_between(x_vals, y1=mean_vals-2*std_error_vals, y2=mean_vals+2*std_error_vals,
                        color=format_lkp[name].get('color', 'black'), alpha=0.2)
        
    max_val = np.max(diurnal_dict['IFS'])
    ax[n].set_xticks(np.array(x_vals) - .5)
    ax[n].set_xticklabels([int(hr) for hr in hour_bin_edges])
    ax[n].set_xlabel('Hour')
    ax[n].set_ylabel('Average mm/hr')
    ax[n].set_ylim([0, None])
    ax[n].legend(loc='upper right')
    ax[n].set_title(f'({ascii_lowercase[n]}) ' + name_lookup[metric_name])

lines_labels = [ax[n].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels, loc='center right', ncol=3)

plt.savefig(os.path.join(plot_dir,f'diurnal_cycle_{model_type}_{model_number}_subdomains.pdf'), bbox_inches='tight')

# %%
print(os.path.join(plot_dir,f'diurnal_cycle_{model_type}_{model_number}_subdomains.pdf'))

# %%

# %%
from string import ascii_lowercase

format_lkp = {'cGAN': {'color': 'b', 'label': 'cGAN-qm'}, 'IFS': {'color': 'r', 'label': 'IFS-qm'}, 'Obs (IMERG)': {'label': 'IMERG'}}
plt.rcParams.update({'font.size': 15})

x_vals = list(diurnal_bstrap_results['mean']['IFS'].keys())
bin_width = 24 / len(x_vals)
hour_bin_edges = np.arange(0, 24, bin_width)
fig, ax = plt.subplots(1,3, figsize=(15,5))
fig.tight_layout(pad=1)
name_lookup = {'mean': 'Mean', 'quantile_999': '99.9th percentile', 'quantile_9999': '99.99th percentile', 'quantile_99999': '99.999th percentile' }

for n, metric_name in enumerate(['mean', 'quantile_999', 'quantile_9999']):
    diurnal_dict = diurnal_bstrap_results[metric_name]
    

    for name, arr in diurnal_dict.items():
        label=format_lkp[name]['label']

        mean_vals = [item['mean'] for item in arr.values()]
        std_error_vals = np.array([item['std'] for item in arr.values()])
        ax[n].plot(x_vals, mean_vals, '-o', label=label, color=format_lkp[name].get('color', 'black'))
        ax[n].fill_between(x_vals, y1=mean_vals-2*std_error_vals, y2=mean_vals+2*std_error_vals,
                        color=format_lkp[name].get('color', 'black'), alpha=0.2)
        # ax.fill_between(x_vals, y1=mean_vals-2*std_error_vals, y2=mean_vals+2*std_error_vals,
        #                 color=format_lkp[name].get('color', 'black'), ls='none', capsize=2, alpha=0.6)

        # if name == 'cGAN':
            
            
        #     # ax.fill_between(x_vals, arr['cgan_metric_min'], arr['cgan_metric_max'], alpha=0.4)
        #     # ax.fill_between(x_vals, arr['cgan_metric_mean'] - 2*arr['cgan_metric_std'], arr['cgan_metric_mean']  + 2*arr['cgan_metric_std'], alpha=0.4)
        # else:
        #     ax.plot(x_vals, arr.values(), '-o', label=label, color=format_lkp[name].get('color', 'black'))
        
    max_val = np.max(diurnal_dict['IFS'])
    ax[n].set_xticks(np.array(x_vals) - .5)
    ax[n].set_xticklabels([int(hr) for hr in hour_bin_edges])
    ax[n].set_xlabel('Hour')
    ax[n].set_ylabel('Average mm/hr')
    ax[n].set_ylim([0, None])
    ax[n].legend(loc='lower right')
    ax[n].set_title(f'({ascii_lowercase[n]}) ' + name_lookup[metric_name])

plt.savefig(os.path.join(plot_dir,f'diurnal_cycle_{model_type}_{model_number}.pdf'), bbox_inches='tight')

# %%


format_lkp = {'cGAN': {'color': 'b', 'label': 'cGAN-qm'}, 'IFS': {'color': 'r', 'label': 'IFS-qm'}, 'Obs (IMERG)': {'label': 'IMERG'}}
plt.rcParams.update({'font.size': 15})

x_vals = list(diurnal_results['mean']['IFS'].keys())
bin_width = 24 / len(x_vals)
hour_bin_edges = np.arange(0, 24, bin_width)

for metric_name, diurnal_dict in diurnal_results.items():
    fig, ax = plt.subplots(1,1, figsize=(7,6))

    for name, arr in diurnal_dict.items():
        label=format_lkp[name]['label']

        diurnal_bstrap_result = diurnal_bstrap_results[name]

        if name == 'cGAN':
            
            ax.plot(x_vals, arr['cgan_metric_mean'], '-o', label=label, color=format_lkp[name].get('color', 'black'))
            # ax.fill_between(x_vals, arr['cgan_metric_min'], arr['cgan_metric_max'], alpha=0.4)
            ax.fill_between(x_vals, arr['cgan_metric_mean'] - 2*arr['cgan_metric_std'], arr['cgan_metric_mean']  + 2*arr['cgan_metric_std'], alpha=0.4)
        else:
            ax.plot(x_vals, arr.values(), '-o', label=label, color=format_lkp[name].get('color', 'black'))
        
    max_val = np.max(diurnal_dict['IFS'])
    ax.set_xticks(np.array(x_vals) - .5)
    ax.set_xticklabels([int(hr) for hr in hour_bin_edges])
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average mm/hr')
    ax.set_ylim([0, None])
    ax.legend(loc='lower right')

    plt.savefig(os.path.join(plot_dir,f'diurnal_cycle_{metric_name}_{model_type}_{model_number}.pdf'), bbox_inches='tight')

# %%
os.path.join(plot_dir,f'diurnal_cycle_{metric_name}_{model_type}_{model_number}.pdf')

# %%
# Diurnal cycle for different seasons

from collections import Counter
from dsrnngan.evaluation import evaluation

seasons_dict = {'MAM': [3,4,5], 'OND': [10,11,12]}
hourly_season_data_obs, hourly_season_data_sample, hourly_season_data_fcst, hourly_season_counts = {}, {}, {}, {}

for n, (season, month_range) in enumerate(seasons_dict.items()):
    hourly_season_data_obs[season], hourly_season_counts[season] = evaluation.get_diurnal_cycle( truth_array, 
                                                                   dates, hours, 
                                                                   longitude_range=longitude_range, latitude_range=latitude_range)
    hourly_season_data_sample[season], _ = evaluation.get_diurnal_cycle( cgan_corrected[:,:,:,0], 
                                                                   dates, hours, 
                                                                   longitude_range=longitude_range, latitude_range=latitude_range)
    
    hourly_season_data_fcst[season], _ = evaluation.get_diurnal_cycle( fcst_corrected, 
                                                                   dates, hours, 
                                                                   longitude_range=longitude_range, latitude_range=latitude_range)

# Plot diurnal cycle for the different areas

fig, ax = plt.subplots(len(seasons_dict),1, figsize=(12,12))
fig.tight_layout(pad=3)
smoothed_diurnal_data_dict = {'IMERG': hourly_season_data_obs,
                     'cGAN sample': hourly_season_data_sample,
                     'IFS': hourly_season_data_fcst}

for n, season in enumerate(seasons_dict):
    for name, data in smoothed_diurnal_data_dict.items():
        
        mean_hourly_data = [np.mean(data[season][n] / hourly_season_counts[season][n]) for n in range(23)]
        
        ax[n].plot(mean_hourly_data, '-o',label=name)
    
    ax[n].legend()
    ax[n].set_xlabel('Hour')
    ax[n].set_ylabel('Average mm/hr')
    ax[n].set_title(season)


# %%
# function to locate peak rainfall

def get_peak_rainfall_hour_bin(array, hours, bin_width=1):
    metric_fn = lambda x,y: np.mean(x)

    metric_by_hour, hour_bin_edges = get_metric_by_hour(metric_fn, array, array, hours, bin_width=bin_width)
    max_time_bin = np.argmax(np.array(list(metric_by_hour.values())))
    
    return max_time_bin



# %%
# Diurnal maximum map
import xarray as xr
from datetime import datetime
from dsrnngan.utils.utils import get_local_hour
from scipy.ndimage import uniform_filter, uniform_filter1d
import xarray as xr

hour_bin_edges = np.arange(0, 24, 1)
filter_size = 31 # roughly the size of lake victoria region
digitized_hours = np.digitize(hours, bins=hour_bin_edges)

n_samples = truth_array.shape[0]
raw_diurnal_data_dict = {'IMERG': truth_array,
                         'cGAN-qm':cgan_corrected[:,:,:,0],
                         'IFS-qm': fcst_corrected}

peak_dict = {}

for name in raw_diurnal_data_dict:
    
    da = xr.DataArray(
        data=raw_diurnal_data_dict[name],
        dims=["hour_range", "lat", "lon"],
        coords=dict(
            lon=(longitude_range),
            lat=(latitude_range),
            hour_range=digitized_hours,
        ),
        attrs=dict(
            description="Precipitation.",
            units="mm/hr",
        ),
    )


    smoothed_vals = uniform_filter1d(da.groupby('hour_range').mean().values, 3, axis=0, mode='wrap')
    peak_dict[name] = np.argmax(smoothed_vals, axis=0)


# %%
from dsrnngan.data import data
from dsrnngan.evaluation import plots
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'font.size': 9})

n_cols = 3
n_rows = 1
fig = plt.figure(constrained_layout=True, figsize=(2.5*n_cols, 3*n_rows))
gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, 
                    width_ratios=[1]*(n_cols ),
                    height_ratios=[1,0.05],
                wspace=0.005) 


filter_size=11
min_bias_val = -12
max_bias_val = 12

# Get observation map first to allow differencing
ax = fig.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())
imerg_peak_data = peak_dict['IMERG']
smoothed_imerg_peak_data = uniform_filter(imerg_peak_data.copy(), size=filter_size, mode='reflect')
im_imerg = ax.imshow(np.flip(smoothed_imerg_peak_data, axis=0), extent = [ min(longitude_range), max(longitude_range), 
                                                                          min(latitude_range), max(latitude_range)], 
                transform=ccrs.PlateCarree(), cmap='twilight_shifted', vmin=0, vmax=23)
ax.add_feature(plots.border_feature)
ax.add_feature(plots.disputed_border_feature)
ax.add_feature(plots.lake_feature, alpha=0.4)
ax.coastlines(resolution='10m', color='black', linewidth=0.4)
ax.set_title('IMERG')

lon_ticks = np.arange(25, 50,10)
lat_ticks = np.arange(-10, 15,10)
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)
ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
ax.set_yticklabels([f"{lt}{'N' if lt >0 else 'N'}" for lt in lat_ticks])
n=1
# for name, peak_data in peak_dict.items():
#     if name != 'IMERG':
#         ax = fig.add_subplot(gs[0, n], projection = ccrs.PlateCarree())
#         smoothed_peak_data = uniform_filter(peak_data.copy(), size=filter_size, mode='reflect')

#         im1 = ax.imshow(np.flip(smoothed_peak_data, axis=0), extent = [ min(longitude_range), max(longitude_range), min(latitude_range), max(latitude_range)], 
#                 transform=ccrs.PlateCarree(),
#                 cmap='twilight_shifted', vmin=0, vmax=23)
        
#         n+=1

#     ax.add_feature(plots.border_feature)
#     ax.add_feature(plots.disputed_border_feature)
#     ax.add_feature(plots.lake_feature, alpha=0.4)
#     ax.coastlines(resolution='10m', color='black', linewidth=0.4)
#     ax.set_title(f'{name}')

#     lon_ticks = np.arange(25, 50,10)
#     lat_ticks = np.arange(-10, 15,10)
#     ax.set_xticks(lon_ticks)
#     ax.set_yticks(lat_ticks)
#     ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
#     ax.set_yticklabels([f"{np.abs(lt)}{'N' if lt >=0 else 'S'}" for lt in lat_ticks])

    
        
cbar_ax_time = fig.add_subplot(gs[1, 0])
# cb = fig.colorbar(cm.ScalarMappable(norm=None, cmap=colors.Colormap('twilight_shifted')), cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10, ticks=range(len(hour_bin_edges)))
cb_time = fig.colorbar(im_imerg, cax=cbar_ax_time, orientation='horizontal', shrink = 0.2, aspect=10, ticks=range(len(hour_bin_edges)),
                    )
cb_time.ax.set_xticks(range(0,24,3))
cb_time.ax.set_xticklabels(range(0,24,3))
cb_time.ax.set_xlabel("Peak rainfall hour (EAT)", loc='center')


#### diff plot


# Get observation map first to allow differencing
colors = [ (0, 0, 1), (1,1,1), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list(
          'text', colors, N=24)
n=1
for name, peak_data in peak_dict.items():
    if name != 'IMERG':
        ax = fig.add_subplot(gs[0, n], projection = ccrs.PlateCarree())
        
        data_diff = (peak_data - imerg_peak_data)

        data_diff[data_diff > 12] = 24 - data_diff[data_diff > 12]
        data_diff[data_diff < -12] = 24 + data_diff[data_diff < -12] 
        print('**', name)
        print('Overall bias: ', data_diff.mean())
        print('Overall bias over land: ', data_diff[lsm > 0.5].mean())
        print('Overall bias over sea (excluding LV): ', data_diff[lsm_ocean < 0.5].mean())
        print('Overall bias over LV (lake only): ', data_diff[lv_lsm == 1].mean())
        print('Overall bias over LV (square box): ', data_diff[lv_area == 1].mean())
        smoothed_peak_data_diff = uniform_filter(data_diff.copy(), size=filter_size, mode='reflect')
        # Account for cyclical nature of hours

        im1 = ax.imshow(np.flip(smoothed_peak_data_diff, axis=0),
                        extent = [ min(longitude_range), max(longitude_range), min(latitude_range), max(latitude_range)], 
                transform=ccrs.PlateCarree(), cmap=cm, vmin=min_bias_val, vmax=max_bias_val)
        
        n+=1

        ax.add_feature(plots.border_feature)
        ax.add_feature(plots.disputed_border_feature)
        ax.add_feature(plots.lake_feature, alpha=0.4)
        ax.coastlines(resolution='10m', color='black', linewidth=0.4)
        ax.set_title(f'{name} - IMERG')

        lon_ticks = np.arange(25, 50,10)
        lat_ticks = np.arange(-10, 15,10)
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
        ax.set_yticklabels([f"{np.abs(lt)}{'N' if lt >=0 else 'S'}" for lt in lat_ticks])

    
cbar_ax_bias = fig.add_subplot(gs[-1, 1:3])


cb_bias = fig.colorbar(im1, cax=cbar_ax_bias, orientation='horizontal', shrink = 0.2, aspect=10, ticks=np.arange(min_bias_val, max_bias_val+0.1, 2),
                    )
cb_bias.ax.set_xlabel("Peak rainfall hour bias (hours)", loc='center')
# cb_time.ax.set_xticks(range(0,24,3))
# cb_time.ax.set_xticklabels(np.arange(min_bias_val, max_bias_val, 2))
fp = os.path.join(plot_dir, f"diurnal_cycle_map_{area.replace(' ', '_')}_{model_type}_{model_number}.pdf")
print(fp)
plt.savefig(fp, format='pdf', bbox_inches='tight')


# %%

from dsrnngan.evaluation import plots
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({'font.size': 9})

n_cols = len(peak_dict)
n_rows = 1
fig = plt.figure(constrained_layout=True, figsize=(2.5*n_cols, 3*n_rows))
gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, 
                    width_ratios=[1]*(n_cols ),
                    height_ratios=[1]*(n_rows) + [0.05],
                wspace=0.005) 


filter_size=5
min_bias_val = -12
max_bias_val = 12

colors = [(1, 0, 0), (1,1,1), (0, 0, 1)]
cm = LinearSegmentedColormap.from_list(
          'text', colors, N=24)

# Get observation map first to allow differencing
ax = fig.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())
imerg_peak_data = peak_dict['IMERG']
smoothed_imerg_peak_data = uniform_filter(imerg_peak_data.copy(), size=filter_size, mode='reflect')
im_imerg = ax.imshow(np.flip(smoothed_imerg_peak_data, axis=0),
                     extent = [ min(longitude_range), max(longitude_range), 
                               min(latitude_range), max(latitude_range)], 
                transform=ccrs.PlateCarree(), cmap='twilight_shifted', vmin=0, vmax=23)
ax.add_feature(plots.border_feature)
ax.add_feature(plots.disputed_border_feature)
ax.add_feature(plots.lake_feature, alpha=0.4)
ax.coastlines(resolution='10m', color='black', linewidth=0.4)
ax.set_title('IMERG')

lon_ticks = np.arange(25, 50,10)
lat_ticks = np.arange(-10, 15,10)
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)
ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
ax.set_yticklabels([f"{lt}{'N' if lt >0 else 'N'}" for lt in lat_ticks])
n=1
for name, peak_data in peak_dict.items():
    if name != 'IMERG':
        ax = fig.add_subplot(gs[0, n], projection = ccrs.PlateCarree())
        
        data_diff = (peak_data - imerg_peak_data)
        data_diff[data_diff > 12] = 24 - data_diff[data_diff > 12]
        data_diff[data_diff < -12] = 24 + data_diff[data_diff < -12] 
        
        smoothed_peak_data_diff = uniform_filter(data_diff.copy(), size=filter_size, mode='reflect')
        # Account for cyclical nature of hours

        im1 = ax.imshow(np.flip(smoothed_peak_data_diff, axis=0),
                        extent = [ min(longitude_range), max(longitude_range), min(latitude_range), max(latitude_range)], 
                transform=ccrs.PlateCarree(), cmap=cm, vmin=min_bias_val, vmax=max_bias_val)
        
        n+=1

    ax.add_feature(plots.border_feature)
    ax.add_feature(plots.disputed_border_feature)
    ax.add_feature(plots.lake_feature, alpha=0.4)
    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
    ax.set_title(f'{name} - IMERG')

    lon_ticks = np.arange(25, 50,10)
    lat_ticks = np.arange(-10, 15,10)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
    ax.set_yticklabels([f"{np.abs(lt)}{'N' if lt >=0 else 'S'}" for lt in lat_ticks])

    
        
cbar_ax_time = fig.add_subplot(gs[-1, 0])
cbar_ax_bias = fig.add_subplot(gs[-1, 1:3])
# cb = fig.colorbar(cm.ScalarMappable(norm=None, cmap=colors.Colormap('twilight_shifted')), cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10, ticks=range(len(hour_bin_edges)))
cb_time = fig.colorbar(im_imerg, cax=cbar_ax_time, orientation='horizontal', shrink = 0.2, aspect=10, ticks=range(len(hour_bin_edges)),
                    )
cb_time.ax.set_xticks(range(0,24,3))
cb_time.ax.set_xticklabels(range(0,24,3))
cb_time.ax.set_xlabel("Peak rainfall hour (EAT)", loc='center')

cb_bias = fig.colorbar(im1, cax=cbar_ax_bias, orientation='horizontal', shrink = 0.2, aspect=10, ticks=np.arange(min_bias_val, max_bias_val+0.1, 2),
                    )
cb_bias.ax.set_xlabel("Peak rainfall bias (hours)", loc='center')
# cb_time.ax.set_xticks(range(0,24,3))
# cb_time.ax.set_xticklabels(np.arange(min_bias_val, max_bias_val, 2))
fp = os.path.join(plot_dir, f"diurnal_cycle_map_{area.replace(' ', '_')}_{model_type}_{model_number}.pdf")
print(fp)
plt.savefig(fp, format='pdf', bbox_inches='tight')

# %%
# Peak local time of rainfall

fig, ax = plt.subplots(1,1)
smoothed_diurnal_data_dict = {'Obs (IMERG)': truth_array,
                    'GAN': cgan_corrected[:n_samples,:,:,0],
                    'Fcst': fcst_corrected[:n_samples, :,:]}

for name, arr in smoothed_diurnal_data_dict.items():
    if name != 'Obs (IMERG)':
        metric_by_hour, hour_bin_edges = get_metric_by_hour(mse, obs_array=truth_array, fcst_array=arr, hours=hours, bin_width=3)
        ax.plot(metric_by_hour.keys(), metric_by_hour.values(), label=name)
        ax.set_xticks(np.array(list(metric_by_hour.keys())) - .5)
        ax.set_xticklabels(hour_bin_edges)
ax.legend()



# %%
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

# Dates are in UTC, converting to EAT which is UTC + 3
time_array = [datetime(d.year, d.month, d.day, (hours[n]+3)%24) for n,d in enumerate(dates)]


smoothed_diurnal_data_dict = {'Obs (IMERG)': truth_array,
                    'GAN': cgan_corrected[:n_samples,:,:,0],
                    'Fcst': fcst_corrected[:n_samples, :,:]
                    }

fig, ax = plt.subplots(1,1)
for name, arr in smoothed_diurnal_data_dict.items():
    da = xr.DataArray(
        data=arr,
        dims=["hour", "lat", "lon"],
        coords=dict(
            lon=(longitude_range),
            lat=(latitude_range),
            hour=hours,
        ),
        attrs=dict(
            description="Precipitation.",
            units="mm/hr",
        ),
    )
    summed_data = np.sum(np.sum(da.groupby('hour').sum().values, axis=-1), axis=-1)
    ax.plot(summed_data, label=name)
ax.legend()

# %%
fig, ax = plt.subplots(1,1)


for name, arr in smoothed_diurnal_data_dict.items():
    hours = [hour+3 for hour in hours]
    se_da = xr.DataArray(
        data=np.power(truth_array - arr[:n_samples, :,:], 2),
        dims=["hour", "lat", "lon"],
        coords=dict(
            lon=(longitude_range),
            lat=(latitude_range),
            hour=hours,
        ),
        attrs=dict(
            description="Precipitation.",
            units="mm/hr",
        ),
    )
    summed_data = np.sum(np.sum(se_da.groupby('hour').sum().values, axis=-1), axis=-1)
    ax.plot(summed_data / (truth_array.size / 24), label=name)
ax.legend()

# # mse by time
# se_da = xr.DataArray(
#         data=np.power(truth_array - fcst_corrected[:n_samples, :,:], 2),
#         dims=["hour", "lat", "lon"],
#         coords=dict(
#             lon=(longitude_range),
#             lat=(latitude_range),
#             hour=hours,
#         ),
#         attrs=dict(
#             description="Precipitation.",
#             units="mm/hr",
#         ),
#     )
# n_elements = truth_array.size
# summed_se = np.sum(np.mean(da.groupby('hour').sum().values, axis=-1), axis=-1)
# plt.plot(summed_se)

# %%
import xarray as xr
from datetime import datetime
from dsrnngan.utils.utils import get_local_hour

diurnal_data_dict = {'Obs (IMERG)': truth_array,
                    'GAN': cgan_corrected[:n_samples,:,:,0],
                    'Fcst': fcst_corrected
                    }

time_array = [datetime(d.year, d.month, d.day, hours[n]) for n,d in enumerate(dates)]

truth_da = xr.DataArray(
    data=truth_array,
    dims=["hour", "lat", "lon"],
    coords=dict(
        lon=(longitude_range),
        lat=(latitude_range),
        hour=hours,
    ),
    attrs=dict(
        description="Precipitation.",
        units="mm/hr",
    ),
)


max_hour_arrays = {}
# for name, arr in diurnal_data_dict.items():
name = 'GAN'
arr = diurnal_data_dict[name]


da = xr.DataArray(
    data=arr,
    dims=["hour", "lat", "lon"],
    coords=dict(
        lon=(longitude_range),
        lat=(latitude_range),
        hour=hours,
    ),
    attrs=dict(
        description="Precipitation.",
        units="mm/hr",
    ),
)



# %%

(_, width, height) = grouped_data.shape

hourly_sum = {(l, get_local_hour(h, longitude_range[l], np.mean(latitude_range))): grouped_data[h,:,l] for h in range(0,24) for l in range(len(longitude_range))}



# %%

max_hour_arrays[name] = np.empty((len(latitude_range), len(longitude_range)))
for lt, lat in enumerate(latitude_range):
    for ln, lon in enumerate(longitude_range):

        hourly_dict = {hr: hourly_sum[(ln, hr)][lt] for hr in range(0,24)}
        max_hour_arrays[name][lt,ln] = {v:k for k,v in hourly_dict.items()}[max(hourly_dict.values())]
            
fig, ax = plt.subplots(2,1, 
                       subplot_kw={'projection' : ccrs.PlateCarree()},
                       figsize = (12,16))
n=0
for name, max_hour_array in max_hour_arrays.items():
    if name != 'Obs (IMERG)':
        
        im = plot_contourf(ax[n], max_hour_array - max_hour_arrays['Obs (IMERG)'], name, lat_range=latitude_range, lon_range=longitude_range, value_range=np.linspace(-24, 24, 10))
        n+=1


# %%

# %% [markdown]
# ## ETS

# %%
with open(os.path.join(log_folder, 'bootstrap_ets_results_n50.pkl'), 'rb') as ifh:
    ets_bstrap_result = pickle.load(ifh)

with open(os.path.join(log_folder, 'bootstrap_far_results_n50.pkl'), 'rb') as ifh:
    far_bstrap_result = pickle.load(ifh)
    
with open(os.path.join(log_folder, 'bootstrap_hitrate_results_n50.pkl'), 'rb') as ifh:
    hr_bstrap_result = pickle.load(ifh)
    
with open(os.path.join(plot_dir, 'ets_cl100-medium-nologs_217600.pkl'), 'rb') as ifh:
    ets_result = pickle.load(ifh)

# %%
annotation_quantiles = [0.9, 0.99, 0.999, 0.9999, 0.99999]
annotation_quantile_vals = [np.quantile(truth_array, q ) for q in annotation_quantiles]


# %%
from string import ascii_lowercase

fig, ax = plt.subplots(1,3,figsize=(15,5))
fig.tight_layout(pad=2)
## 
IFS_qm_mean = [v['mean'] for v in ets_bstrap_result['fcst'].values()]
IFS_qm_std = [v['std'] for v in ets_bstrap_result['fcst'].values()]
threshold_vals = ets_bstrap_result['thresholds']

ets_format_lookup = {'fcst': {'label': 'IFS-qm', 'color': 'r'}, 'cgan':  {'label': 'cGAN-qm', 'color': 'b'}}

for source in ['fcst', 'cgan']:
    mean_vals = np.array([v['mean'] for v in ets_bstrap_result[source].values()])
    std = np.array([v['std'] for v in ets_bstrap_result[source].values()])

    ax[0].plot(threshold_vals, mean_vals, '-+', color=ets_format_lookup[source]['color'], label=ets_format_lookup[source]['label'], markersize=10)
    ax[0].fill_between(threshold_vals, y1=mean_vals-2*std, y2=mean_vals+2*std,
                    color=ets_format_lookup[source].get('color', 'black'), alpha=0.2)
max_val = list(ets_bstrap_result['fcst'].values())[0]['mean']

for n, q_val in tqdm(enumerate(annotation_quantile_vals[1:])):
    ax[0].vlines(q_val, 0, max_val, linestyles='--')
    q_val_text = f'{np.round(float(annotation_quantiles[n])*100, 12)}th'.replace('.0th', 'th')
    ax[0].text(1.01*q_val,  max_val, q_val_text)
ax[0].set_xlim([0,47])
ax[0].legend(loc='center right')
ax[0].set_xlabel('Precipitation event threshold (mm/hr)')
ax[0].set_ylabel('Equitable Threat Score')
ax[0].set_title('(a)')

#######################################
## Hit rate and False alarm rate

for source in ['fcst', 'cgan']:
    mean_vals = np.array([v['mean'] for v in far_bstrap_result[source].values()])
    std = np.array([v['std'] for v in far_bstrap_result[source].values()])

    ax[1].plot(threshold_vals, mean_vals, '-+', color=ets_format_lookup[source]['color'], label=ets_format_lookup[source]['label'], markersize=10)
    ax[1].fill_between(threshold_vals, y1=mean_vals-2*std, y2=mean_vals+2*std,
                    color=ets_format_lookup[source].get('color', 'black'), alpha=0.2)
max_val = list(far_bstrap_result['fcst'].values())[0]['mean']

for n, q_val in tqdm(enumerate(annotation_quantile_vals[1:])):
    ax[1].vlines(q_val, 0.7, 1, linestyles='--')
    q_val_text = f'{np.round(float(annotation_quantiles[n])*100, 12)}th'.replace('.0th', 'th')
    ax[1].text(1.05*q_val,  0.72, q_val_text)
    
ax[1].legend(loc='center right')
ax[1].set_xlabel('Precipitation event threshold (mm/hr)')
ax[1].set_ylabel('False Alarm Rate')
ax[1].set_title('(b)')


for source in ['fcst', 'cgan']:
    mean_vals = np.array([v['mean'] for v in hr_bstrap_result[source].values()])
    std = np.array([v['std'] for v in hr_bstrap_result[source].values()])

    ax[2].plot(threshold_vals, mean_vals, '-+', color=ets_format_lookup[source]['color'], label=ets_format_lookup[source]['label'], markersize=10)
    ax[2].fill_between(threshold_vals, y1=mean_vals-2*std, y2=mean_vals+2*std,
                    color=ets_format_lookup[source].get('color', 'black'), alpha=0.2)
max_val = list(hr_bstrap_result['fcst'].values())[0]['mean']

for n, q_val in tqdm(enumerate(annotation_quantile_vals[1:])):
    ax[2].vlines(q_val, 0, 0.45, linestyles='--')
    q_val_text = f'{np.round(float(annotation_quantiles[n])*100, 12)}th'.replace('.0th', 'th')
    ax[2].text(1.05*q_val,  0.42, q_val_text)
    
ax[2].legend(loc='center right')
ax[2].set_xlabel('Precipitation event threshold (mm/hr)')
ax[2].set_ylabel('Hit Rate')
ax[2].set_title('(c)')

fp = os.path.join(plot_dir, f'ets_{model_type}_{model_number}.pdf')
print(fp)
plt.savefig(fp, format='pdf', bbox_inches='tight')

# %% [markdown]
# # Locations of high quantiles

# %%
quantile_boundaries = [np.round(item, 6) for item in quantile_boundaries]

# %%
truth_quantiles = np.quantile(truth_array, quantile_boundaries)
sample_quantiles = np.quantile(samples_gen_array[:,:,:,0], quantile_boundaries)
fcst_quantiles = np.quantile(fcst_array, quantile_boundaries)
fcst_corrected_quantiles = np.quantile(fcst_corrected, quantile_boundaries)

# %%
fig, axs = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()},
                       figsize=(10,10))

threshold = 55
digitized_truth = np.digitize(truth_array, truth_quantiles)
digitized_sample = np.digitize(samples_gen_array[:,:,:,0], sample_quantiles)


im = plot_contourf(axs, np.sum(digitized_sample >= threshold, axis=0) / np.sum(digitized_truth >= threshold, axis=0), title=f'', cmap='Reds', lat_range=latitude_range, 
                   lon_range=longitude_range)
plt.colorbar(im, ax=axs)

# im = plot_contourf(axs[1], , title=f'', cmap='Reds', lat_range=latitude_range, 
#                    lon_range=longitude_range)
# plt.colorbar(im, ax=axs[1])

# %% [markdown]
# # Ratio of quantiles

# %%
truth_quantiles = np.quantile(truth_array, quantile_boundaries, axis=0)
sample_quantiles = np.quantile(samples_gen_array[:,:,:,0], quantile_boundaries, axis=0)
fcst_quantiles = np.quantile(fcst_array, quantile_boundaries, axis=0)
fcst_corrected_quantiles = np.quantile(fcst_corrected, quantile_boundaries, axis=0)

# %%
quantile = 0.9999

# quantile_index = quantile_boundaries.index(quantile)
quantile_index = 59

t = truth_quantiles[quantile_index, :,:]
s = sample_quantiles[quantile_index, :, :]

fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()},
                       figsize=(10,10))

im = plot_contourf(ax, s / t, title=f'', cmap='Reds', lat_range=latitude_range, 
                   lon_range=longitude_range)
plt.colorbar(im, ax=ax)

# %% [markdown]
# ## Check locations of max values

# %%
max_samples_gen = np.max(np.max(samples_gen_array, axis=-1), axis=0)
max_truth = np.max(truth_array, axis=0)

# %%
fig, ax = plt.subplots(2,1, subplot_kw={'projection': ccrs.PlateCarree()},
                       figsize=(10,10))
threshold = 300
im = plot_contourf(ax[0], (max_samples_gen > threshold), title=f'Sample max rainfall > {threshold}', cmap='Reds', lat_range=DEFAULT_LATITUDE_RANGE, 
                   lon_range=DEFAULT_LONGITUDE_RANGE)
plt.colorbar(im, ax=ax[0])

obs_thresold = 80
im = plot_contourf(ax[1], (max_truth > obs_thresold), title=f'IMERG max rainfall > {obs_thresold}', cmap='Reds', lat_range=DEFAULT_LATITUDE_RANGE, 
                   lon_range=longitude_range)
plt.colorbar(im, ax=ax[1])

# %% [markdown]
# ## Single pixel time series
#

# %%

log_folder_consecutive = '/user/work/uz22147/logs/cgan/d9b8e8059631e76f/n1000_201810-201812_e50'

with open(os.path.join(log_folder_consecutive, f'arrays-160000.pkl'), 'rb') as ifh:
    consecutive_arrays = pickle.load(ifh)
    
cons_truth_array = consecutive_arrays['truth']
cons_samples_gen_array = consecutive_arrays['samples_gen']
cons_fcst_array = consecutive_arrays['fcst_array']
cons_ensmean_array = np.mean(consecutive_arrays['samples_gen'], axis=-1)
cons_dates = consecutive_arrays['dates']
cons_hours = consecutive_arrays['hours']


# %% [markdown]
# ## Check training data for high rainfall

# %%
# Check range of values in training data
from dsrnngan.data.tfrecords_generator import create_mixed_dataset, DataGenerator

train = DataGenerator('train',
                           batch_size=1,
                           fcst_shape=(200, 200, 20),
                            con_shape=(200, 200, 2),
                            out_shape=(200, 200, 1),
                           downsample=False, 
                           weights=[0.4, 0.3, 0.2, 0.1], 
                           records_folder='/user/work/uz22147/tfrecords/d34d309eb0e00b04', 
                           )



# %%
from tqdm import tqdm
from dsrnngan.data.data import denormalise

max_rainfall = 0
take_n = 20000

for a in tqdm(train.take(take_n).as_numpy_iterator()):
    ifs_input = a[0]['lo_res_inputs']
    imerg_output = a[1]['output']
    
    if denormalise(imerg_output.max()) > max_rainfall:
        max_rainfall = denormalise(imerg_output.max())

# %% [markdown]
# ## Quantiles with training time

# %%
from sklearn.linear_model import LinearRegression

model_numbers = sorted([153600,147200,140800,134400,128000,121600,115200,108800])

truth_quantiles = {}
sample_quantiles = {}

for model_number in tqdm(model_numbers, total=len(model_numbers)):
    if model_number == 160000:
        log_folder = '/user/work/uz22147/logs/cgan/38fba564f1c9852b/n1000_201806-201905_e50'
    else:
        log_folder = '/user/work/uz22147/logs/cgan/d9b8e8059631e76f/n1000_201806-201905_e50'

    with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
        arrays = pickle.load(ifh)
        
    truth_array = arrays['truth']
    samples_gen_array = arrays['samples_gen']
    fcst_array = arrays['fcst_array']
    ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
    training_dates = arrays['dates']
    training_hours = arrays['hours']

    (n_samples, width, height, ensemble_size) = samples_gen_array.shape

    ## This is needed until code is fixed

    if model_number == 160000:
        for n in range(fcst_array.shape[0]):
            fcst_array[n,:,:] = denormalise(fcst_array[n,:,:])
            
    number_in_top_range = 10000
    quantile_boundaries = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] + list(np.linspace(0.9, 1.0, number_in_top_range))
    truth_quantiles[model_number] = np.quantile(truth_array, quantile_boundaries)
    sample_quantiles[model_number] = np.quantile(samples_gen_array[:,:,:,0], quantile_boundaries)
    print('Number data points per quantile in top range: ', 0.1*len(truth_array.flatten()) / number_in_top_range)
    print('percentile steps in top range: ', 10 / number_in_top_range)
    
fig, ax = plt.subplots(1,1, figsize=(8,6))

for n, model_number in enumerate(model_numbers):
    
    truth_q = truth_quantiles[model_number]
    sample_q = sample_quantiles[model_number]
    
    ax.plot(truth_q, sample_q, label=model_number)

ax.legend()
plt.savefig('cgan_metrics/quantiles_by_iterations.png')

# %%
# !zip -r cgan_metrics.zip cgan_metrics
