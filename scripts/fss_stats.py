# Get an idea of distribution of rainfall
import os, sys
import numpy as np
import pickle
from argparse import ArgumentParser
from datetime import datetime

from pathlib import Path
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.data import load_imerg_raw, load_era5_month_raw
from calendar import monthrange

hours = range(24)

latitude_vals = np.arange(-11.95, 15.95, 0.1)
longitude_vals = np.arange(22.05, 50.05, 0.1) # Asymettric ranges to make sure lat and lon are correct orientation

parser = ArgumentParser(description='Gather monthly rainfall data.')
parser.add_argument('--window-sizes', nargs='+', default=4,
                    help='Window sizes (space separated)')
parser.add_argument('--years', nargs='+', default=2019,
                    help='Years(s) to process (space separated)')
parser.add_argument('--threshold', type=float, default=0.3,
                    help='Rainfall threshold')
parser.add_argument('--output-dir', type=str, help='output directory', default='fss_stats')
args = parser.parse_args()

years = args.years if isinstance(args.years, list) else [args.years]
window_sizes = args.window_sizes if isinstance(args.window_sizes, list) else [args.window_sizes]

window_sizes = [int(item) for item in window_sizes]
years = [int(item) for item in years]

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

print(args)
threshold = args.threshold

for window_size in window_sizes:
    
    num_windows_lat = int(len(latitude_vals) / window_size)
    num_windows_lon = int(len(longitude_vals) / window_size)
    
    fraction_arrs = []
                    
    for year in years:
        print(year)
        for month in tqdm(np.arange(1, 13), total=12):
            for day in range(1, monthrange(year, month)[1] + 1):
                for hour in range(24):
                    fraction_arr = np.empty((num_windows_lat, num_windows_lon))
                    fraction_arr[:] = np.nan
    
                    ds_hr = load_imerg_raw(year, month, day, hour, 
                                            imerg_data_dir='/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/half_hourly/final')
                    ds_hr = ds_hr.sel(lat=latitude_vals).sel(lon=longitude_vals)
                    
                    data = ds_hr['precipitationCal'].values
                    data_shape = data.shape
                    
                    for lat_window in range(num_windows_lat):
                        for lon_window in range(num_windows_lon):
                            window = data[lat_window*window_size:(lat_window+1)*window_size, lon_window*window_size:(lon_window+1)*window_size]
                            fraction = (window > threshold).sum() / (window_size**2)

                            fraction_arr[lat_window, lon_window] = fraction
                    fraction_arrs.append(fraction_arr)
    
    fractions_stacked = np.stack(fraction_arrs, axis=0)           
    with open(os.path.join(args.output_dir, f'fractions_{window_size}_{threshold}.pkl'), 'wb+') as ofh:
        pickle.dump(fractions_stacked, ofh)
                
    