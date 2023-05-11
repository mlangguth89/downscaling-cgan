# Get an idea of distribution of rainfall
import os, sys
import numpy as np
import pickle
from argparse import ArgumentParser
from datetime import datetime
from calendar import monthrange
from pathlib import Path
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.data.data import DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE, load_ifs_raw

hours = range(24)

latitude_vals = np.arange(-11.95, 16.05, 0.1)
longitude_vals = np.arange(22.05, 50.05, 0.1) # Asymettric ranges to make sure lat and lon are correct orientation

parser = ArgumentParser(description='Gather monthly rainfall data.')
parser.add_argument('--year', type=int, help='Year to process')
parser.add_argument('--month', type=int, help='Month to process')
parser.add_argument('--output-dir', type=str, help='output directory', default='ifs_rainfall_data')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

year = args.year
month = args.month
total_rainfall_dict = {}
monthly_rainfall_dict = {}
monthly_rainfall_dict_era5 = {}

print(args)

monthly_rainfall = 0
day_hours = []
for day in range(1, monthrange(year, month)[1]+1):
    daily_rainfall = 0

    for hour in range(24):
        try:
            ds_hr = load_ifs_raw('tp', year=year, month=month, day=day, hour=hour, latitude_vals=DEFAULT_LATITUDE_RANGE,
                                longitude_vals=DEFAULT_LONGITUDE_RANGE)
            
            precip_hourly_values = ds_hr['tp']
            daily_rainfall += precip_hourly_values
            
            day_hours.append((day, hour))
        except FileNotFoundError:
            print(f'No data for {year}-{month}-{day} {hour}')
    monthly_rainfall += daily_rainfall


with open(os.path.join(args.output_dir, f'monthly_rainfall_ifs_{year}_{month}.pkl'), 'wb+') as ofh:
    pickle.dump({'monthly_rainfall': monthly_rainfall, 'day_hours': day_hours}, ofh)

    