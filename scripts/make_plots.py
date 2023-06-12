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
import tempfile
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
from matplotlib import gridspec
from metpy import plots as metpy_plots
from matplotlib.colors import ListedColormap, BoundaryNorm
from properscoring import crps_ensemble

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.utils import read_config
from dsrnngan.utils.utils import load_yaml_file, get_best_model_number
from dsrnngan.evaluation.plots import plot_contourf, range_dict, quantile_locs, percentiles, plot_quantiles
from dsrnngan.data.data import denormalise, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE
from dsrnngan.data import data
from dsrnngan.evaluation.rapsd import  rapsd
from dsrnngan.evaluation.scoring import fss, get_spread_error_data
from dsrnngan.evaluation.evaluation import get_diurnal_cycle
from dsrnngan.evaluation.benchmarks import QuantileMapper, empirical_quantile_map, quantile_map_grid

def clip_outliers(data, lower_pc=2.5, upper_pc=97.5):
    
    data_clipped = copy.deepcopy(data)
    data_clipped[data_clipped<np.percentile(data_clipped, lower_pc)] = np.percentile(data_clipped, lower_pc)  #using percentiles rather than indexing a sorted list of the array values allows this to work even when data is a small array. (I've not checked if this works for masked data.)
    data_clipped[data_clipped>np.percentile(data_clipped, upper_pc)] = np.percentile(data_clipped, upper_pc)

    return data_clipped

# This dict chooses which plots to create
metric_dict = {'examples': False,
               'scatter': True,
               'rank_hist': False,
               'spread_error': False,
               'rapsd': False,
               'quantiles': False,
               'hist': False,
               'crps': False,
               'fss': False,
               'diurnal': True,
               'confusion_matrix': False
               }

plot_persistence = False

################################################################################
## Setup
################################################################################

parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--output-dir', type=str, help='output directory', default=str(HOME))
parser.add_argument('--model-type', type=str, help='Choice of model type', default=str(HOME))
parser.add_argument('--debug', action='store_true', help='Debug mode')
args = parser.parse_args()

model_type = args.model_type

log_folders = {'full_image': '/user/work/uz22147/logs/cgan/43ae7be47e9a182e_full_image/n1000_201806-201905_e50',
               'cropped': '/user/work/uz22147/logs/cgan/5c577a485fbd1a72/n4000_201806-201905_e10'}

# If in debug mode, then don't overwrite existing plots
if args.debug:
    temp_stats_dir = tempfile.TemporaryDirectory()
    args.output_dir = temp_stats_dir.name

model_number = get_best_model_number(log_folder=log_folders[model_type])

if model_type not in log_folders:
    raise ValueError('Model type not found')


log_folder = log_folders[model_type]
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

# Find dry and rainy days in sampled dataset
means = [(n, truth_array[n,:,:].mean()) for n in range(n_samples)]
sorted_means = sorted(means, key=lambda x: x[1])

n_extreme_days = 10
wet_day_indexes = [item[0] for item in sorted_means[-10:]]
dry_day_indexes = [item[0] for item in sorted_means[:10]]


# Get lat/lon range from log folder
base_folder = '/'.join(log_folder.split('/')[:-1])
config = load_yaml_file(os.path.join(base_folder, 'setup_params.yaml'))
model_config, _, ds_config, data_config, gen_config, dis_config, train_config, val_config = read_config.get_config_objects(config)

# Locations
latitude_range=np.arange(data_config.min_latitude, data_config.max_latitude, data_config.latitude_step_size)
longitude_range=np.arange(data_config.min_longitude, data_config.max_longitude, data_config.longitude_step_size)

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


####################################
### Load in quantile-mapped data (created by scripts/quantile_mapping.py)
####################################

with open(os.path.join(log_folder, f'fcst_qmap_270.pkl'), 'rb') as ifh:
    fcst_corrected = pickle.load(ifh)

with open(os.path.join(log_folder, f'cgan_qmap_15.pkl'), 'rb') as ifh:
    cgan_corrected = pickle.load(ifh)
          
################################################################################
## Climatological data for comparison.
print('Loading climate data',flush=True)

all_imerg_data = []
all_ifs_data = []

if args.debug:
    year_range = [2004]
    month_range = [1]
else:
    year_range = range(2003, 2018)
    month_range = range(1,13)

for year in tqdm(year_range):
    for month in month_range:

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
    print('*********** Plotting Examples **********************')
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


    plt.savefig(f'plots/cGAN_samples_IFS_{model_type}_{model_number}.pdf', format='pdf')

##################################################################################
## Binned scatter plot
##################################################################################

if metric_dict['scatter']:

    h = np.histogram(truth_array, 10)
    bin_edges = h[1]
    bin_edges[0] = 0
    bin_centres = [0.5*(bin_edges[n] + bin_edges[n+1]) for n in range(len(bin_edges) - 1)]

    mean_data = {}
    data_dict = {'IFS': fcst_corrected,
                'cgan': cgan_corrected}

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
    plt.savefig(f'plots/scatter_{model_type}_{model_number}.pdf', format='pdf')


##################################################################################
###  Rank histogram
##################################################################################



if metric_dict['rank_hist']:
    
    print('*********** Plotting Rank histogram **********************') 
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
    plt.savefig(f'plots/rank_hist__{model_type}_{model_number}.pdf', format='pdf')

#################################################################################
## Spread error

if metric_dict['spread_error']:
    print('*********** Plotting spread error **********************')
    plt.rcParams.update({'font.size': 20})
    
    upper_percentile = 99.9
    quantile_step_size = (100-upper_percentile) / 100
    
    data_dict = {'cgan': {'data': samples_gen_array, 'label': 'GAN'},
                 'cgan_qmap': {'data': cgan_corrected, 'label': 'GAN + qmap'}}
    fig, ax = plt.subplots(1,1)
    for k, v in data_dict.items():
        variance_mse_pairs = get_spread_error_data(n_samples=n_samples, observation_array=truth_array, ensemble_array=v['data'], 
                                                   quantile_step_size=quantile_step_size)


        # Currently leaving out the top one
        ens_spread = [np.sqrt(item[0]) for item in variance_mse_pairs[:-1]]
        rmse_err = [np.sqrt(item[1]) for item in variance_mse_pairs[:-1]]

        ax.plot(ens_spread, rmse_err, label=v['label'])
    ax.plot(np.linspace(0,max(ens_spread),10), np.linspace(0,max(ens_spread),10), '--')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Ensemble spread')
    ax.legend()
    ax.set_title(f'Spread error up to {100*(1-quantile_step_size)}th percentile')
    plt.savefig(f'plots/spread_error_{model_type}_{model_number}.pdf', format='pdf')

#################################################################################
## RAPSD



plt.rcParams.update({'font.size': 16})
if metric_dict['rapsd']:
    
    print('*********** Plotting RAPSD **********************')

    rapsd_data_dict = {
                        'GAN': {'data': samples_gen_array[:, :, :, 0], 'color': 'b', 'linestyle': '-'},
                        'Obs (IMERG)': {'data': truth_array, 'color': 'k', 'linestyle': '-'},
                        'Fcst': {'data': fcst_array, 'color': 'r', 'linestyle': '-'},
                        'Fcst + qmap': {'data': fcst_corrected, 'color': 'r', 'linestyle': '--'},
                        'GAN + qmap': {'data': cgan_corrected[:,:,:,0], 'color': 'b', 'linestyle': '--'}}

    rapsd_results = {}
    for k, v in rapsd_data_dict.items():
            rapsd_results[k] = []
            for n in tqdm(range(n_samples)):
            
                    fft_freq_pred = rapsd(v['data'][n,:,:], fft_method=np.fft)
                    rapsd_results[k].append(fft_freq_pred)

            rapsd_results[k] = np.mean(np.stack(rapsd_results[k], axis=-1), axis=-1)

    fig, ax = plt.subplots(1,1)

    for k, v in rapsd_results.items():
        ax.plot(v, label=k, color=rapsd_data_dict[k]['color'], linestyle=rapsd_data_dict[k]['linestyle'])
    
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylabel('Power Spectral Density')
    ax.set_xlabel('Wavenumber')
    ax.legend()
    
    plt.savefig(f'plots/rapsd_{model_type}_{model_number}.pdf', format='pdf')

#################################################################################
## Q-Q plot


if metric_dict['quantiles']:

    print('*********** Plotting Q-Q  **********************')
    quantile_data_dict = {
                    'GAN': {'data': samples_gen_array[:, :, :, 0], 'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'data': truth_array, 'color': 'k'},
                    'Fcst': {'data': fcst_array, 'color': 'r', 'marker': '+', 'alpha': 1},
                    'Fcst + qmap': {'data': fcst_corrected, 'color': 'r', 'marker': 'o', 'alpha': 0.7},
                    'GAN + qmap': {'data': cgan_corrected[:, :, :, 0], 'color': 'b', 'marker': 'o', 'alpha': 0.7}}

    fig, ax = plt.subplots(1,1)
    plot_quantiles(quantile_data_dict=quantile_data_dict, ax=ax, min_data_points_per_quantile=1,
                   save_path=f'plots/quantiles_total_{model_type}_{model_number}.pdf')

    # Quantiles for different areas

    percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
    percentiles=np.concatenate(percentiles_list)
    quantile_boundaries = [item / 100 for item in percentiles]

    fig, ax = plt.subplots(max(2, len(special_areas)),1, figsize=(10, len(special_areas)*10))
    fig.tight_layout(pad=4)
    for n, (area, area_range) in enumerate(special_areas.items()):

        lat_range = area_range['lat_index_range']
        lon_range = area_range['lon_index_range']
        
        quantile_data_dict = {
                    'GAN': {'data': samples_gen_array[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1], 0], 'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'data': truth_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], 'color': 'k'},
                    'Fcst': {'data': fcst_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], 'color': 'r', 'marker': '+', 'alpha': 1},
                    'Fcst + qmap': {'data': fcst_corrected[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], 'color': 'r', 'marker': 'o', 'alpha': 0.7},
                    'GAN + qmap': {'data': cgan_corrected[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1], 0], 'color': 'b', 'marker': 'o', 'alpha': 0.7}}
               
        _, ax[n] = plot_quantiles(quantile_data_dict=quantile_data_dict, ax=ax[n], min_data_points_per_quantile=1)

        
    fig.tight_layout(pad=2.0)
    plt.savefig(f'plots/quantiles_area_{model_type}_{model_number}.pdf', format='pdf')



#################################################################################
## Histograms



if metric_dict['hist']:
    print('*********** Plotting Histograms **********************')
    
    (q_99pt9, q_99pt99) = np.quantile(truth_array, [0.999, 0.9999])


    fig, axs = plt.subplots(2,1, figsize=(10,10))
    fig.tight_layout(pad=4)
    bin_boundaries=np.arange(0,300,4)

    data_dict = {'Obs (IMERG)': {'data': truth_array, 'histtype': 'stepfilled', 'alpha':0.6, 'facecolor': 'grey'}, 
                'IFS': {'data': fcst_array, 'histtype': 'step', 'edgecolor': 'red'},
                'IFS + qmap': {'data': fcst_corrected, 'histtype': 'step', 'edgecolor': 'red', 'linestyle': '--'},
                'cGAN': {'data': samples_gen_array[:,:,:,0], 'histtype': 'step', 'edgecolor': 'blue'},
                'cGAN + qmap': {'data': cgan_corrected[:,:,:,0], 'histtype': 'step', 'edgecolor': 'blue', 'linestyle': '--'}}
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
    rmse_dict = {'GAN_rmse': np.sqrt(np.mean(np.square(truth_array - samples_gen_array[:,:,:,0]), axis=0)),
                'fcst_rmse' : np.sqrt(np.mean(np.square(truth_array - fcst_array), axis=0)),
                'GAN_qmap_rmse': np.sqrt(np.mean(np.square(truth_array - cgan_corrected[:,:,:,0]), axis=0)),
                'fcst_qmap_rmse' : np.sqrt(np.mean(np.square(truth_array - fcst_corrected), axis=0))}

    bias_dict = {'GAN_bias': np.mean(samples_gen_array[:,:,:,0] - truth_array, axis=0),
                'ensmean_bias' : np.mean(ensmean_array - truth_array, axis=0),
                'fcst_bias' : np.mean(fcst_array - truth_array, axis=0), 
                'GAN_qmap_bias': np.mean(cgan_corrected[:,:,:,0] - truth_array, axis=0),
                'fcst_qmap_bias' : np.mean(fcst_corrected - truth_array, axis=0)}

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
    
    # Bias

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
    
    print('*********** Plotting CRPS **********************')
    
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

## Fractions Skill Score

if metric_dict['fss']:
    print('*********** Plotting FSS **********************')
    from dsrnngan.evaluation.evaluation import get_fss_scores
    from dsrnngan.evaluation.plots import plot_fss_scores

    window_sizes = list(range(1,11)) + [20, 40, 60, 80, 100, 120, 150, 200]
    n_samples = truth_array.shape[0]
    
    fss_data_dict = {
                        'cgan': samples_gen_array[:n_samples, :, :, 0],
                        'ifs': fcst_array[:n_samples, :, :],
                        'fcst_qmap': fcst_corrected[:n_samples, :, :],
                        'cgan_qmap': cgan_corrected[:n_samples, :, :, 0]}

    # get quantiles
    quantile_locs = [0.5, 0.985, 0.999, 0.9999]
    hourly_thresholds = np.quantile(truth_array, quantile_locs)

    fss_results = get_fss_scores(truth_array, fss_data_dict, hourly_thresholds, window_sizes, n_samples)

    # Save results
    with open(f'plots/fss_{model_type}_{model_number}.pkl', 'wb+') as ofh:
        pickle.dump(fss_results, ofh)
            
    plot_fss_scores(fss_results=fss_results, output_folder='plots', output_suffix=f'{model_type}_{model_number}')
    
    
    # FSS for regions
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
                        'fcst': fcst_array[:n_samples,lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1]],
                        'fcst_qmap': fcst_corrected[:n_samples, lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1]],
                        'cgan_qmap': cgan_corrected[:n_samples, lat_range_index[0]:lat_range_index[1], lon_range_index[0]:lon_range_index[1], 0]}  
        fss_area_results[area] = get_fss_scores(area_truth_array, fss_data_dict, quantile_locs, window_sizes, n_samples)
        
        plot_fss_scores(fss_results=fss_area_results[area], output_folder='plots', output_suffix=f'{area}_{model_type}_{model_number}')
        

    # FSS for grid scale
    # from dsrnngan.scoring import get_filtered_array

    # mode = 'constant'

    # for thr_index in range(len(hourly_thresholds)):

    #     thr = hourly_thresholds[thr_index] # 0 = median

    #     arrays_filtered = {}

    #     for size in window_sizes:

    #         arrays_filtered[size] = {}

    #         for k, d in tqdm(fss_data_dict.items()):  
    #             arrays_filtered[size][k] = []
    #             for n in range(truth_array.shape[0]):
    #                 # Convert to binary fields with the given intensity threshold
    #                 I = (d >= thr).astype(np.single)

    #                 # Compute fractions of pixels above the threshold within a square
    #                 # neighboring area by applying a 2D moving average to the binary fields        
    #                 arrays_filtered[size][k].append(get_filtered_array(int_array=I, mode=mode, size=size))

    #         for k in arrays_filtered[size]:
    #             arrays_filtered[size][k] = np.stack(arrays_filtered[size][k])

    # with open(f'fss_grid_{model_type}_{model_number}_thr{thr_index}.pkl', 'wb+') as ofh:
    #     pickle.dump(arrays_filtered, ofh)  

#################################################################################

if metric_dict['diurnal']:
    print('*********** Plotting Diurnal cycle **********************')

    
    diurnal_data_dict = {'Obs (IMERG)': truth_array,
                         'GAN': cgan_corrected[:,:,:,0],
                         'Fcst': fcst_corrected
                         }
    ## Diurnal cycle
    diurnal_data_sums = {}
    for k, d in diurnal_data_dict.items():
        
        diurnal_data_sums[k], hourly_counts = get_diurnal_cycle(d, dates, hours, longitude_range, latitude_range)

    fig, ax = plt.subplots(1,1, figsize=(5,5))

    for name, data in diurnal_data_sums.items():
        
        mean_hourly_data = [(data[n]/hourly_counts[n]) for n in range(24)]
        # std_dev_hourly_data = np.array([data[n].std() for n in range(24)])
        
        # ax.errorbar(x=range(24), y=mean_hourly_data, fmt='-o', yerr=2*std_dev_hourly_data, label=name)
        ax.plot(mean_hourly_data, '-o', label=name)
        
    ax.legend()
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average mm/hr')
    
    plt.rcParams.update({'font.size': 20})
    plt.savefig(f'plots/diurnal_cycle_{model_type}_{model_number}.pdf', bbox_inches='tight')
    
    with open(f'plots/diurnal_cycle_{model_type}_{model_number}.pkl', 'wb+') as ofh:
        pickle.dump(diurnal_data_sums, ofh)
    
    
    # # Seasonal diurnal cycle
    # seasons_dict = {'MAM': [3,4,5], 'OND': [10,11,12]}
    # hourly_season_sums, hourly_season_counts = {}, {}

    # hourly_season_sums = {}
    # for k, d in diurnal_data_dict.items():
    #     hourly_season_sums[k] = {}
    #     for n, (season, month_range) in enumerate(seasons_dict.items()):

    #         hourly_season_sums[k][season], hourly_season_counts[season] = get_diurnal_cycle(d, 
    #                                                                                         dates, hours, 
    #                                                                                         longitude_range=longitude_range,
    #                                                                                         latitude_range=latitude_range)

    # fig, ax = plt.subplots(len(seasons_dict),1, figsize=(12,12))
    # fig.tight_layout(pad=3)

    # for n, season in enumerate(seasons_dict):
    #     for name, data in hourly_season_sums.items():
            
    #         mean_hourly_data = [np.mean(data[season][n] / hourly_season_counts[season][n]) for n in range(23)]
    #         ax[n].plot(mean_hourly_data, '-o',label=name)
        
    #     ax[n].legend()
    #     ax[n].set_xlabel('Hour')
    #     ax[n].set_ylabel('Average mm/hr')
    #     ax[n].set_title(season)
    # plt.savefig(f'plots/diurnal_cycle_seasonal_{model_type}_{model_number}.pdf')
    
    # with open(f'plots/diurnal_cycle_seasonal_{model_type}_{model_number}.pkl', 'wb+') as ofh:
    #     pickle.dump(diurnal_data_dict, ofh)
        
    # Diurnal cycles for different areas
    hourly_area_sums, hourly_area_counts = {}, {}

    for n, (area, area_range) in enumerate(special_areas.items()):
        
        hourly_area_sums[area] = {}
        hourly_area_counts[area] = {}
        
        lat_range_ends = area_range['lat_range']
        lon_range_ends = area_range['lon_range']
        lat_range_index = area_range['lat_index_range']
        lon_range_index = area_range['lon_index_range']
        lat_range = np.arange(lat_range_ends[0], lat_range_ends[-1]+0.0001, 0.1)
        lon_range = np.arange(lon_range_ends[0], lon_range_ends[-1]+0.0001, 0.1)
        
        for name, d in diurnal_data_dict.items():
            hourly_area_sums[area][name], hourly_area_counts[area][name] = get_diurnal_cycle( d[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1], 
                                                                                        dates, hours, 
                                                                                        longitude_range=lon_range, 
                                                                                        latitude_range=lat_range)

    # Plot diurnal cycle for the different areas
    fig, ax = plt.subplots(len(special_areas),1, figsize=(12,18))
    fig.tight_layout(pad=3)
    
    # TODO: get errorbars on this
    for n, (area, d) in enumerate(hourly_area_sums.items()):
        for name, arr in d.items():
            mean_hourly_data = [np.mean(arr[n] / hourly_area_counts[area][name][n]) for n in range(23)]
            
            ax[n].plot(mean_hourly_data, '-o', label=name)
        
        ax[n].legend()
        ax[n].set_xlabel('Hour')
        ax[n].set_ylabel('Average mm/hr')
        ax[n].set_title(area)
    plt.savefig(f'plots/diurnal_cycle_area_{model_type}_{model_number}.pdf')
    
    # Diurnal maximum map
    import xarray as xr
    from datetime import datetime
    from dsrnngan.utils.utils import get_local_hour


    diurnal_data_dict = {'Obs (IMERG)': truth_array,
                        'GAN': cgan_corrected[:,:,:,0],
                        'Fcst': fcst_corrected
                        }
    max_hour_arrays = {}
    for name, arr in diurnal_data_dict.items():
        time_array = [datetime(d.year, d.month, d.day, hours[n]) for n,d in enumerate(dates)]

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

        
        grouped_data = da.groupby('hour').sum().values

        (_, width, height) = grouped_data.shape

        hourly_sum = {(l, get_local_hour(h, longitude_range[l], np.mean(latitude_range))): grouped_data[h,:,l] for h in range(0,24) for l in range(len(longitude_range))}

        
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
    plt.savefig(f'plots/diurnal_maximum_map_{model_type}_{model_number}.pdf')


#################################################################################

if metric_dict.get('confusion_matrix'):
    print('*********** Calculating confusion matrices **********************')

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    
    # quantile_locations = [0.1, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    # hourly_thresholds = np.quantile(truth_array, quantile_locations)
    hourly_thresholds = [0.1, 5, 20]

    results = {'hourly_thresholds': hourly_thresholds,
               'conf_mat': []}
    for threshold in hourly_thresholds:
        y_true = (truth_array > threshold).astype(np.int0).flatten()

        y_dict = {
                'ifs': (fcst_array > threshold).astype(np.int0).flatten(),
                'cgan' : (samples_gen_array[:,:,:,0]> threshold).astype(np.int0).flatten(),
                'ifs_qmap': (fcst_corrected > threshold).astype(np.int0).flatten(),
                'persistence': (persisted_fcst_array > threshold).astype(np.int0).flatten()}

        tmp_results_dict = {'threshold': threshold}    
        for k, v in tqdm(y_dict.items()):
            tmp_results_dict[k] = confusion_matrix(y_true, v)
        
        results['conf_mat'].append(tmp_results_dict)
        with open(f'plots/confusion_matrices.pkl', 'wb+') as ofh:
            pickle.dump(results, ofh)