import pickle
import os, sys
import copy
from tqdm import tqdm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
from matplotlib import gridspec
from metpy import plots as metpy_plots
from matplotlib.colors import ListedColormap, BoundaryNorm
from properscoring import crps_ensemble

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.utils import load_yaml_file
from dsrnngan.plots import plot_precip, plot_contourf
from dsrnngan.data import denormalise, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE
from dsrnngan import data
from dsrnngan.rapsd import  rapsd
from dsrnngan.scoring import fss
from dsrnngan.evaluation import get_diurnal_cycle
from dsrnngan.benchmarks import get_quantile_areas, get_quantiles_by_area, get_quantile_mapped_forecast

def clip_outliers(data, lower_pc=2.5, upper_pc=97.5):
    
    data_clipped = copy.deepcopy(data)
    data_clipped[data_clipped<np.percentile(data_clipped, lower_pc)] = np.percentile(data_clipped, lower_pc)  #using percentiles rather than indexing a sorted list of the array values allows this to work even when data is a small array. (I've not checked if this works for masked data.)
    data_clipped[data_clipped>np.percentile(data_clipped, upper_pc)] = np.percentile(data_clipped, upper_pc)

    return data_clipped

# This dict chooses which plots to create
metric_dict = {'examples': True,
               'rank_hist': True,
               'rapsd': True,
               'quantiles': True,
               'hist': True,
               'crps': True,
               'fss': True,
               'diurnal': True
               }

################################################################################
## Setup
################################################################################

parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--output-dir', type=str, help='output directory', default=str(HOME))
parser.add_argument('--model-number', type=str, help='Checkpoint to evaluate on', default=str(HOME))
parser.add_argument('--model-type', type=str, help='Choice of model type', default=str(HOME))
args = parser.parse_args()

model_number = args.model_number
model_type = args.model_type

log_folders = {'basic': '/user/work/uz22147/logs/cgan/d9b8e8059631e76f/n1000_201806-201905_e50',
               'full_image': '/user/work/uz22147/logs/cgan/43ae7be47e9a182e_full_image/n1000_201806-201905_e50',
               'cropped': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n2000_201806-201905_e20',
               'cropped_4000': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n4000_201806-201905_e10'}

if model_type not in log_folders:
    raise ValueError('Model type not found')


log_folder = log_folders[model_type]
with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
truth_array = arrays['truth']
samples_gen_array = arrays['samples_gen']
fcst_array = arrays['fcst_array']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]


assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
(n_samples, width, height, ensemble_size) = samples_gen_array.shape

# Find dry and rainy days in sampled dataset
means = [(n, truth_array[n,:,:].mean()) for n in range(n_samples)]
sorted_means = sorted(means, key=lambda x: x[1])

n_extreme_days = 10
wet_day_indexes = [item[0] for item in sorted_means[-10:]]
dry_day_indexes = [item[0] for item in sorted_means[:10]]


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



################################################################################
### Quantile mapping


# Quantiles
step_size = 0.001
range_dict = {0: {'start': 0.1, 'stop': 1, 'interval': 0.1, 'marker': '+', 'marker_size': 64},
              1: {'start': 1, 'stop': 10, 'interval': 1, 'marker': '+', 'marker_size': 512},
              2: {'start': 10, 'stop': 80, 'interval':10, 'marker': '+', 'marker_size': 1024},
              3: {'start': 80, 'stop': 99.1, 'interval': 1, 'marker': '+', 'marker_size': 512},
              4: {'start': 99.1, 'stop': 99.91, 'interval': 0.1, 'marker': '+', 'marker_size': 256},
              5: {'start': 99.9, 'stop': 99.99, 'interval': 0.01, 'marker': '+', 'marker_size': 64 },
              6: {'start': 99.99, 'stop': 99.999, 'interval': 0.001, 'marker': '+', 'marker_size': 20}}
                  
percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
percentiles=np.concatenate(percentiles_list)
quantile_locs = [np.round(item / 100.0, 6) for item in percentiles]


month_ranges = [[1,2], [3,4,5], [6,7,8,9], [10,11,12]]
quantile_threshold = 0.999

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

# identify best threshold and train on all the data
quantile_areas = get_quantile_areas(list(training_dates), month_ranges, latitude_range, longitude_range, hours=training_hours, num_lat_lon_chunks=2)
quantiles_by_area = get_quantiles_by_area(quantile_areas, fcst_data=ifs_train_data, obs_data=imerg_train_data, 
                                          quantile_locs=quantile_locs)

fcst_corrected = get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, 
                                              hours=hours, month_ranges=month_ranges, 
                                              quantile_areas=quantile_areas, 
                                              quantiles_by_area=quantiles_by_area)
                                            #   quantile_threshold=0.99999)
                                            
################################################################################
## Climatological data for comparison.

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

#################################################################################################
## Plot examples
#################################################################################################

if metric_dict['examples']:
    tp_index = data.all_ifs_fields.index('tp')

    # plot configurations
    levels = [0, 0.1, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150] # in units of log10
    precip_cmap = ListedColormap(metpy_plots.ctables.colortables["precipitation"][:len(levels)-1], 'precipitation')
    precip_norm = BoundaryNorm(levels, precip_cmap.N)
    plt.rcParams.update({'font.size': 11})
    spacing = 10
    units = "Rain rate [mm h$^{-1}$]"
    precip_levels=np.arange(0, 1.5, 0.15)

    indexes = wet_day_indexes[:10] 

    num_cols = 6
    num_samples = len(indexes)
    num_rows = num_samples

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
        data_lookup = {'cgan_sample': {'data': img_gens[:,:,0], 'title': 'cGAN sample'},
                    'cgan_mean': {'data': avg_img_gens, 'title': f'cGAN sample average'},
                    'imerg' : {'title': f"IMERG: {date_str}", 'data': truth},
                    'ifs': {'data': fcst, 'title': 'IFS'},
                    'ifs_qmap': {'data': fcst_corr, 'title': 'IFS qmap'}
                    }
        for col, (k, val) in enumerate(data_lookup.items()):
    
            ax = fig.add_subplot(gs[n, col], projection = ccrs.PlateCarree())
            ax.coastlines(resolution='10m', color='black', linewidth=0.4)
            
            clipped_data = clip_outliers(val['data'], lower_pc=0.1, upper_pc=99.9)
            im = ax.imshow(val['data'],
                    interpolation='nearest',
                    norm=precip_norm,
                    cmap=precip_cmap,
                    origin='lower',
                    extent=[min(DEFAULT_LONGITUDE_RANGE), max(DEFAULT_LONGITUDE_RANGE), 
                    min(DEFAULT_LATITUDE_RANGE), max(DEFAULT_LATITUDE_RANGE)],
                    transform=ccrs.PlateCarree(),
                    alpha=0.8)
            ax.add_feature(cfeature.BORDERS)
            ax.set_title(val['title'])
            
    cbar_ax = fig.add_subplot(gs[-1, :])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10,
                    )
    cb.ax.set_xlabel("Precipitation (mm / hr)", loc='center')


    plt.savefig(f'cGAN_samples_IFS_{model_type}_{model_number}.pdf', format='pdf')

##################################################################################
###  Rank histogram
##################################################################################

if metric_dict['rank_hist']:
    rng = np.random.default_rng()
    noise_factor = 1e-6
    n_samples = 100

    temp_truth = np.repeat(truth_array[:n_samples, :, : ,None].copy(), samples_gen_array.shape[-1], axis=-1)
    temp_samples = samples_gen_array[:n_samples, :, :, :].copy()

    temp_truth += rng.random(size=temp_truth.shape, dtype=np.float32)*noise_factor
    temp_samples += rng.random(size=temp_samples.shape, dtype=np.float32)*noise_factor

    ranks = np.sum(temp_truth > temp_samples, axis=-1)

    n_bins = samples_gen_array.shape[-1]

    fig, ax = plt.subplots(1,1)
    (h, _) = np.histogram(ranks.flatten() / ranks.size, bins=n_bins)
    h = h / h.sum()
    ax.plot(h)
    ax.hlines(1/n_bins, 0, n_bins, linestyles='dashed', colors=['r'])

    ax.set_ylim([0, max(h)+0.01])
    ax.set_title('Rank histogram ')
    plt.savefig(f'plots/rank_hist_{model_number}.png')

#################################################################################
## RAPSD

if metric_dict['rapsd']:

    rapsd_truth = []
    rapsd_pred = []
    rapsd_fcst = []
    rapsd_fcst_corrected = []  
    rapsd_fcst_persisted = []

    for n in tqdm(range(n_samples)):
            fft_freq_pred = rapsd(truth_array[n,:,:], fft_method=np.fft)
            rapsd_truth.append(fft_freq_pred)
            
            fft_freq_pred = rapsd(samples_gen_array[n,:,:,0], fft_method=np.fft)
            rapsd_pred.append(fft_freq_pred)

            fft_freq_fcst = rapsd(fcst_array[n, :, :], fft_method=np.fft)
            rapsd_fcst.append(fft_freq_fcst)
            
            fft_freq_fcst_corrected = rapsd(fcst_corrected[n, :, :], fft_method=np.fft)
            rapsd_fcst_corrected.append(fft_freq_fcst_corrected)
            
            fft_freq_fcst_persisted = rapsd(persisted_fcst_array[n, :, :], fft_method=np.fft)
            rapsd_fcst_persisted.append(fft_freq_fcst_persisted)

    rapsd_truth = np.mean(np.stack(rapsd_truth, axis=-1), axis=-1)
    rapsd_pred = np.mean(np.stack(rapsd_pred, axis=-1), axis=-1)
    rapsd_fcst = np.mean(np.stack(rapsd_fcst, axis=-1), axis=-1)
    rapsd_fcst_corrected = np.mean(np.stack(rapsd_fcst_corrected, axis=-1), axis=-1)
    rapsd_fcst_persisted = np.mean(np.stack(rapsd_fcst_persisted, axis=-1), axis=-1)

    fig, ax = plt.subplots(1,1)

    ax.plot(rapsd_truth, label='IMERG', color='k')
    ax.plot(rapsd_fcst, 'r', label='IFS')
    ax.plot(rapsd_fcst_corrected, 'r--', label='IFS qmap')
    ax.plot(rapsd_pred, 'b', label='cGAN sample') # Single member of ensemble
    ax.plot(rapsd_fcst_persisted, 'k--', label='Persisted')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylabel('RAPSD')
    ax.set_xlabel('frequency')
    ax.legend()
    plt.rcParams.update({'font.size': 16})
    plt.savefig(f'plots/rapsd_{model_type}_{model_number}.pdf', format='pdf')

#################################################################################
## Q-Q plot

if metric_dict['quantiles']:

    # Quantiles for annotating plot
    (q_99pt9, q_99pt99) = np.quantile(truth_array, [0.999, 0.9999])

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    marker_handles = None
    for v in range_dict.values():
        
        quantile_boundaries = np.arange(v['start'], v['stop'], v['interval']) / 100
        
        truth_quantiles = np.quantile(truth_array, quantile_boundaries)
        sample_quantiles = np.quantile(samples_gen_array[:,:,:,0], quantile_boundaries)
        fcst_quantiles = np.quantile(fcst_array, quantile_boundaries)
        fcst_corrected_quantiles = np.quantile(fcst_corrected, quantile_boundaries)
        persisted_fcst_quantiles = np.quantile(persisted_fcst_array, quantile_boundaries)

        max_fcst_val = max(max(sample_quantiles), max(fcst_quantiles))
        max_truth_val = max(truth_quantiles)
        
        size=v['marker_size']
        cmap = plt.colormaps["plasma"]
        marker = v['marker']

        s1 = ax.scatter(truth_quantiles, sample_quantiles, c='blue', marker='+', label='cGAN', s=size, cmap=cmap)
        s2 = ax.scatter(truth_quantiles, fcst_quantiles, c='red', marker='x', label='IFS', s=size, cmap=cmap)
        s3 = ax.scatter(truth_quantiles, fcst_corrected_quantiles, c='green', marker='.', label='IFS qmap', s=size, cmap=cmap, alpha=0.7)
        s4 = ax.scatter(truth_quantiles, persisted_fcst_quantiles, c='black', marker='+', label='Persisted', s=size, cmap=cmap)
        
        if not marker_handles:
            marker_handles = [s1, s2, s3, s4]

    # all_marker_handles = list(itertools.chain.from_iterable(marker_handles.values()))
    ax.legend(handles=marker_handles, loc='upper left')
    ax.plot(np.linspace(0, max_truth_val, 100), np.linspace(0, max_truth_val, 100), 'k--')
    ax.set_xlabel('iMERG (mm/hr)')
    ax.set_ylabel('Model (mm/hr)')

    ax.vlines(q_99pt9, 0, max(sample_quantiles), linestyles='--')
    ax.vlines(q_99pt99, 0, max(sample_quantiles), linestyles='--')
    ax.text(q_99pt9 - 5, max(sample_quantiles) - 20, '$99.9^{th}$')
    ax.text(q_99pt99 -6 , max(sample_quantiles) - 20, '$99.99^{th}$')

    plt.rcParams.update({'font.size': 20})
    plt.savefig(f'plots/quantiles_total_{model_type}_{model_number}.pdf', format='pdf')


    # Quantiles for different areas

    percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
    percentiles=np.concatenate(percentiles_list)
    quantile_boundaries = [item / 100 for item in percentiles]

    fig, ax = plt.subplots(max(2, len(special_areas)),1, figsize=(10, 18))
    fig.tight_layout(pad=4)
    for n, (area, area_range) in enumerate(special_areas.items()):


        lat_range = area_range['lat_index_range']
        lon_range = area_range['lon_index_range']
        truth_quantiles = np.quantile(truth_array[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_boundaries)
        sample_quantiles = np.quantile(samples_gen_array[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1], 0], quantile_boundaries)
        fcst_quantiles = np.quantile(fcst_array[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_boundaries)
        fcst_corrected_quantiles = np.quantile(fcst_corrected[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_boundaries)
        
        max_val = max(truth_quantiles)
        
        ax[n].scatter(truth_quantiles, sample_quantiles, marker='+', label='cgan')
        ax[n].scatter(truth_quantiles, fcst_quantiles, marker='x', label='fcst')
        ax[n].scatter(truth_quantiles, fcst_corrected_quantiles, marker='o', label='fcst qmap')
        ax[n].plot(np.arange(0,max_val, 0.1), np.arange(0,max_val, 0.1), 'k--')
        ax[n].set_xlabel('truth')
        ax[n].set_ylabel('model')
        ax[n].set_title(area)
        ax[n].legend(loc='upper left')
        
        max_line_val = max(max(sample_quantiles), max_val, max(fcst_quantiles))
        (q_99pt9, q_99pt99) = np.quantile(truth_array[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], [0.999, 0.9999])
        ax[n].vlines(q_99pt9, 0, max_line_val, linestyles='--')
        ax[n].vlines(q_99pt99, 0, max_line_val, linestyles='--')
        ax[n].text(q_99pt9 , max_line_val - 20, '$99.9^{th}$')
        ax[n].text(q_99pt99 , max_line_val - 20, '$99.99^{th}$')
        
    fig.tight_layout(pad=2.0)
    plt.savefig(f'plots/quantiles_area_{model_type}_{model_number}.pdf', format='pdf')

#################################################################################
## Histograms

if metric_dict['hist']:
    (q_99pt9, q_99pt99) = np.quantile(truth_array, [0.999, 0.9999])


    fig, axs = plt.subplots(2,1, figsize=(10,10))
    fig.tight_layout(pad=4)
    bin_boundaries=np.arange(0,300,4)

    data_dict = {'IMERG': {'data': truth_array, 'histtype': 'stepfilled', 'alpha':0.6, 'facecolor': 'grey'}, 
                'IFS': {'data': fcst_array, 'histtype': 'step', 'edgecolor': 'red'},
                'IFS qmap': {'data': fcst_corrected, 'histtype': 'step', 'edgecolor': 'red', 'linestyle': '--'},
                'cGAN sample': {'data': samples_gen_array[:,:,:,0], 'histtype': 'step', 'edgecolor': 'blue'}}
    rainfall_amounts = {}

    edge_colours = ["blue", "green", "red", 'orange']
    for n, (name, d) in enumerate(data_dict.items()):
        
        axs[0].hist(d['data'].flatten(), bins=bin_boundaries, histtype=d['histtype'], label=name, alpha=d.get('alpha'),
                    facecolor=d.get('facecolor'), edgecolor=d.get('edgecolor'), linestyle= d.get('linestyle'))
        
        axs[1].hist(d['data'].flatten(), bins=bin_boundaries, histtype=d['histtype'], label=name, weights=d['data'].flatten(), alpha=d.get('alpha'),
                    facecolor=d.get('facecolor'), edgecolor=d.get('edgecolor'), linestyle= d.get('linestyle'))

        
    for ax in axs:
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Average hourly rainfall in bin (mm/hr)')
        ax.vlines(q_99pt99, 0, 10**8, linestyles='--')
        ax.text(q_99pt99 + 7 , 10**7, '$99.99^{th}$')
    plt.savefig(f'plots/histograms_{model_type}_{model_number}.pdf', format='pdf')

    #################################################################################
    ## Bias and RMSE
    # RMSE
    rmse_dict = {'single_sample_rmse': np.sqrt(np.mean(np.square(truth_array - samples_gen_array[:,:,:,0]), axis=0)),
                'ensmean_rmse' : np.sqrt(np.mean(np.square(truth_array - np.mean(samples_gen_array, axis=-1)), axis=0)),
                'fcst_rmse' : np.sqrt(np.mean(np.square(truth_array - fcst_array), axis=0))}

    bias_dict = {'single_sample_bias': np.mean(samples_gen_array[:,:,:,0] - truth_array, axis=0),
                'ensmean_bias' : np.mean(ensmean_array - truth_array, axis=0),
                'fcst_bias' : np.mean(fcst_array - truth_array, axis=0)}

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

    plt.savefig(f'plots/rmse_{model_type}_{model_number}.pdf', format='pdf')

    lat_range=np.arange(-10.05, 10.05, 0.1)
    lon_range=np.arange(25.05, 45.05, 0.1)
    fig, ax = plt.subplots(len(bias_dict.keys())+2,2, 
                        subplot_kw={'projection' : ccrs.PlateCarree()},
                        figsize=(12,16))

    max_bias_val = max([v.max() for v in bias_dict.values()])

    for n, (k,v) in enumerate(bias_dict.items()):


        # value_range = list(np.arange(-0.5, 0.5, 0.01))
        value_range = None

        im = plot_contourf(ax[n,0], v, title=k, cmap='RdBu', value_range=list(np.arange(-1.5, 1.5, 3 / 50)), lat_range=latitude_range, lon_range=longitude_range)
        plt.colorbar(im, ax=ax[n,0])
        
        im = plot_contourf(ax[n,1], 100* v / hourly_historical_avg, title=f'100 * {k} / truth_avg', value_range=list(np.arange(-200, 200, 400 / 50)),
                        cmap='RdBu', lat_range=latitude_range, lon_range=longitude_range)
        plt.colorbar(im, ax=ax[n,1])
        
    im = plot_contourf(ax[n+1,0], hourly_historical_avg, title='truth_avg', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+1,0])

    im = plot_contourf(ax[n+1,1], np.mean(np.mean(samples_gen_array, axis=-1), axis=0), title='samples_avg', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+1,1])

    im = plot_contourf(ax[n+2,0], np.mean(fcst_array, axis=0), title='fcst_avg', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+2,0])

    im = plot_contourf(ax[n+2,1], np.mean(fcst_array, axis=0), title='', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+2,1])

    plt.savefig(f'plots/bias_{model_type}_{model_number}.pdf', format='pdf')

#################################################################################

## CRPS

if metric_dict['crps']:
    # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
    crps_score_grid = crps_ensemble(truth_array, samples_gen_array)
    crps_score = crps_score_grid.mean()


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection' : ccrs.PlateCarree()}, figsize=(5,5))

    max_level = 10
    value_range = list(np.arange(0, max_level, max_level / 50))

    im = ax.contourf(longitude_range, latitude_range, np.mean(crps_score_grid, axis=0), transform=ccrs.PlateCarree(),
                        cmap='Reds', 
                        # levels=value_range, norm=colors.Normalize(min(value_range), max(value_range)),
                        extend='both')

    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(im, ax=ax)
    plt.savefig(f'plots/crps_{model_type}_{model_number}.pdf', format='pdf')


#################################################################################

## Fractional Skill Score

if metric_dict['crps']:

    window_sizes = list(range(1,11)) + [20, 40, 60, 80, 100] + [150, 200]
    daily_thresholds = [1, 5, 20, 30, 50] # 1mm/day = drizzle, 50 mm/day = extreme

    fss_cgan = []
    fss_fcst = []
    fss_fcst_qmap = []

    for thr in daily_thresholds:

        tmp_fss_cgan = []
        tmp_fss_fcst = []
        tmp_fss_fcst_qmap = []

        for w in tqdm(window_sizes):
            
            tmp_fss_cgan.append(fss(truth_array, samples_gen_array, w, thr/24.0, mode='constant'))
            tmp_fss_fcst.append(fss(truth_array, fcst_array, w, thr/24.0, mode='constant'))
            tmp_fss_fcst_qmap.append(fss(truth_array, fcst_corrected, w, thr/24.0, mode='constant'))
        
        fss_cgan.append(tmp_fss_cgan)
        fss_fcst.append(tmp_fss_fcst)
        fss_fcst_qmap.append(tmp_fss_fcst_qmap)

    fig, axs = plt.subplots(3, 1, figsize = (16, 16))
    fig.tight_layout(pad=4.0)
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1,10))]

    for n, thr in enumerate(daily_thresholds):
        
        axs[0].plot(window_sizes, [item for item in fss_cgan[n]], label=f'{thr} mm/day', color='b', linestyle=linestyles[n])
        axs[1].plot(window_sizes, [item for item in fss_fcst[n]], label=f'{thr} mm/day', color='b', linestyle=linestyles[n])
        axs[2].plot(window_sizes, [item for item in fss_fcst_qmap[n]], label=f'{thr} mm/day', color='b', linestyle=linestyles[n])

        axs[0].set_title('cGAN sample')
        axs[1].set_title('IFS')
        axs[2].set_title('IFS qmap')

    for ax in axs:    
        ax.hlines(0.5, 0, max(window_sizes), linestyles='dashed', colors=['r'])
        ax.set_ylim(0,1)
        ax.set_xlabel('Neighbourhood size')
        ax.set_ylabel('FSS')
        ax.legend()
        
    plt.savefig(f'plots/fractional_skill_score_{model_type}_{model_number}.pdf', format='pdf')

#################################################################################

if metric_dict['diurnal']:
    ## Diurnal cycle
    hourly_data_obs, hourly_data_sample, hourly_data_fcst, hourly_data_fcst_persisted, hourly_counts = get_diurnal_cycle(truth_array, 
                                                                                                                        samples_gen_array, 
                                                                                                                        fcst_array, 
                                                                                                                        persisted_fcst_array, 
                                                                                                                        dates, hours, longitude_range, latitude_range)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    diurnal_data_dict = {'IMERG': hourly_data_obs,
                        'cGAN sample': hourly_data_sample,
                        'IFS': hourly_data_fcst,
                        'Persisted': hourly_data_fcst_persisted}

    for name, data in diurnal_data_dict.items():
        
        mean_hourly_data = [(data[n]/hourly_counts[n]) for n in range(24)]
        # std_dev_hourly_data = np.array([data[n].std() for n in range(24)])
        
        # ax.errorbar(x=range(24), y=mean_hourly_data, fmt='-o', yerr=2*std_dev_hourly_data, label=name)
        ax.plot(mean_hourly_data, '-o', label=name)
        
    ax.legend()
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average mm/hr')
    plt.savefig(f'plots/diurnal_cycle_{model_type}_{model_number}.pdf')
    
    
    # Seasonal diurnal cycle
    
    seasons_dict = {'MAM': [3,4,5], 'OND': [10,11,12]}
    hourly_season_data_obs, hourly_season_data_sample, hourly_season_data_fcst, hourly_season_data_fcst_persisted, hourly_season_counts = {}, {}, {}, {}, {}

    for n, (season, month_range) in enumerate(seasons_dict.items()):
        hourly_season_data_obs[season], hourly_season_data_sample[season], hourly_season_data_fcst[season], hourly_season_data_fcst_persisted[season], hourly_season_counts[season] = get_diurnal_cycle( truth_array, 
                                                                                                samples_gen_array, 
                                                                                                fcst_array, dates, hours, 
                                                                                                longitude_range=lon_range, latitude_range=lat_range)

    # Plot diurnal cycle for the different areas

    fig, ax = plt.subplots(len(seasons_dict),1, figsize=(12,12))
    fig.tight_layout(pad=3)
    diurnal_data_dict = {'IMERG': hourly_season_data_obs,
                        'cGAN sample': hourly_season_data_sample,
                        'IFS': hourly_season_data_fcst,
                        'Persisted': hourly_season_data_fcst_persisted}

    for n, season in enumerate(seasons_dict):
        for name, data in diurnal_data_dict.items():
            
            mean_hourly_data = [np.mean(data[season][n] / hourly_season_counts[season][n]) for n in range(23)]
            
            ax[n].plot(mean_hourly_data, '-o',label=name)
        
        ax[n].legend()
        ax[n].set_xlabel('Hour')
        ax[n].set_ylabel('Average mm/hr')
        ax[n].set_title(season)
    plt.savefig(f'plots/diurnal_cycle_seasonal_{model_type}_{model_number}.pdf')

#################################################################################