"""
Script to quantile map the IFS data pre training

"""

import pickle
import os, sys
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

import numpy as np
import xarray as xr

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
sys.path.insert(1, str(HOME / 'dsrnngan'))

from dsrnngan.utils.utils import get_best_model_number, date_range_from_year_month_range, load_yaml_file
from dsrnngan.utils.read_config import get_data_paths
from dsrnngan.data.data import denormalise
from dsrnngan.evaluation.benchmarks import QuantileMapper, empirical_quantile_map


parser = ArgumentParser(description='Script for quantile mapping.')

parser.add_argument('--year', type=int)
parser.add_argument('--month', type=int)
parser.add_argument('--num-lat-lon-chunks', type=int, help='Number of chunks to split up spatial data into along each axis', default=20)
parser.add_argument('--model-type', type=str, help='Choice of model type')
parser.add_argument('--output-folder', type=str, help='Folder to save data in', default=None)
parser.add_argument('--debug', action='store_true', help='Debug mode')
args = parser.parse_args()


print('Loading model data', flush=True)
log_folders = {
               'cropped': '/user/work/uz22147/logs/cgan/5c577a485fbd1a72_0876a13533d2542c/n4000_201806-201905_e10'}

# Get best model
log_folder = log_folders[args.model_type]
model_number = 262400
data_paths = get_data_paths()

with open(os.path.join(log_folder, f'fcst_qmapper_{args.num_lat_lon_chunks}.pkl'), 'rb') as ifh:
    qmapper = pickle.load(ifh)


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

###########################
print('## Quantile mapping for IFS', flush=True)
###########################
# Use data prepared by the dsrnngan.data.setupdata script
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

# Need to account for difference in lat/lon ranges; setupdata incorporates the final lat value whereas 
# the generated data doesn't
imerg_train_data = np.concatenate(imerg_train_data, axis=0)[:,:-1,:-1]
ifs_train_data = np.concatenate(ifs_train_data, axis=0)[:,:-1,:-1]

if args.debug:
    imerg_train_data = imerg_train_data[:100,:,:]
    ifs_train_data = ifs_train_data[:100,:,:]
    training_dates = training_dates[:100]
    training_hours = training_hours[:100]
    
# calculate quantiles
month_ranges = [[n for n in range(1,13)]]
qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, longitude_range=longitude_range,
                         num_lat_lon_chunks=1)
ifs_quantile_locs = qmapper.update_quantile_locations(input_data=ifs_train_data, max_step_size=0.01)

q_obs = np.quantile(imerg_train_data, ifs_quantile_locs)
q_model = np.quantile(ifs_train_data, ifs_quantile_locs)

#############################
# Load in training data sequentially
year_month_range = [f'{args.year}{args.month:02d}']

dates = date_range_from_year_month_range(year_month_range)
start_date = dates[0]

ifs_folder = data_paths['GENERAL']['IFS']

if args.output_folder is None:
    args.output_folder = ifs_folder

os.makedirs(args.output_folder, exist_ok=True)


for date in tqdm(dates, total=len(dates)):
    for fcst_hour in ['00', '12']:
        fp_end = f"tp_HRES_1h_EAfrica_{date.strftime('%Y-%m-%d')}_{fcst_hour}h.nc"
        fp = os.path.join(ifs_folder, 'tp', fp_end)
        
        ds = xr.load_dataset(fp)
        
        # filter to lat lon
        ds = ds.sel(longitude=longitude_range, method='backfill')
        ds = ds.sel(latitude=latitude_range, method='backfill')
        # only quantile map the relevant hours
        for ix in range(5, 19):
            # original_data = np.expand_dims(ds.isel(time=ix)['tp'].values, axis=0)
            original_data = ds.isel(time=ix)['tp'].values
            qmapped_data = np.interp(original_data, q_model, q_obs)
            
            # np.squeeze(qmapped_data, axis=0)
            replacement_da = xr.DataArray(qmapped_data, coords={'latitude': ds.latitude, 'longitude': ds.longitude},
                                          dims=['latitude', 'longitude'])
            ds.isel(time=ix)['tp'] = replacement_da
        
        # save to a new folder
        ds.to_netcdf(os.path.join(args.output_folder, 'tpq', fp_end.replace('tp_', 'tpq_')))