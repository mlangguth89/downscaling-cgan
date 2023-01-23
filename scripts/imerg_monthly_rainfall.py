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

from dsrnngan.data import load_imerg_raw, load_era5_month_raw, load_ifs, load_imerg, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE

hours = range(24)

latitude_vals = DEFAULT_LATITUDE_RANGE
longitude_vals = DEFAULT_LONGITUDE_RANGE 

parser = ArgumentParser(description='Gather monthly rainfall data.')
parser.add_argument('--year', type=int, help='Years to process')
parser.add_argument('--output-dir', type=str, help='output directory', default='rainfall_data')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

year = args.year
total_imerg_rainfall_dict = {}
total_ifs_rainfall_dict = {}
monthly_rainfall_imerg = {}
monthly_rainfall_ifs = {}
monthly_rainfall_era5 = {}

print(args)

for month in tqdm(np.arange(1, 13)):
    
    total_imerg_rainfall_dict[month] = {}
    total_ifs_rainfall_dict[month] = {}
    
    monthly_rainfall_imerg[month] = 0
    monthly_rainfall_ifs[month] = 0
    
    for day in range(1, monthrange(year, month)[1]+1):
        total_imerg_rainfall_dict[month][day] = 0
        total_ifs_rainfall_dict[month][day] = 0
        
        for hour in hours:

            precip_hourly_values = load_imerg(datetime(year, month, day), hour=hour,
                            data_dir='/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/half_hourly/final/',
                        latitude_vals=latitude_vals, longitude_vals=longitude_vals)
            
            total_imerg_rainfall_dict[month][day] += precip_hourly_values

            # IFS
            ds_ifs = load_ifs('tp', datetime(year, month, day), hour, log_precip=False, norm=False, 
                              latitude_vals=latitude_vals, longitude_vals=longitude_vals)
            total_ifs_rainfall_dict[month][day] += ds_ifs
                        
        monthly_rainfall_imerg[month] += total_imerg_rainfall_dict[month][day]
        monthly_rainfall_ifs[month] += total_ifs_rainfall_dict[month][day]

    # # era5
    era5_ds = load_era5_month_raw('tp', year=year, month=month, era_data_dir='/bp1/geog-tropical/data/ERA-5/day')
    era5_ds = era5_ds.sel(latitude=latitude_vals, method='backfill').sel(longitude=longitude_vals, method='backfill')
    
    era5_ds_monthly = era5_ds.sum('time')                             
    # assert len(era5_ds.latitude.values) == len(tmp_lat_vals)
    # assert len(era5_ds.longitude.values) == len(tmp_lon_vals)
    
    monthly_rainfall_era5[month] = era5_ds_monthly['tp'].values
    era5_ds.close()
    era5_ds_monthly.close()
    
    
    with open(os.path.join(args.output_dir, f'daily_imerg_rainfall_{year}.pkl'), 'wb+') as ofh:
        pickle.dump(total_imerg_rainfall_dict, ofh)
    
    with open(os.path.join(args.output_dir, f'daily_ifs_rainfall_{year}.pkl'), 'wb+') as ofh:
        pickle.dump(total_ifs_rainfall_dict, ofh)

    with open(os.path.join(args.output_dir, f'monthly_imerg_rainfall_{year}.pkl'), 'wb+') as ofh:
        pickle.dump(monthly_rainfall_imerg, ofh)
    
    with open(os.path.join(args.output_dir, f'monthly_ifs_rainfall_{year}.pkl'), 'wb+') as ofh:
        pickle.dump(monthly_rainfall_ifs, ofh)
        
    with open(os.path.join(args.output_dir, f'monthly_era5_rainfall_{year}.pkl'), 'wb+') as ofh:
        pickle.dump(monthly_rainfall_era5, ofh)