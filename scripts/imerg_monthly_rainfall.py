# Get an idea of distribution of rainfall
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from argparse import ArgumentParser
from datetime import datetime
from calendar import monthrange

from pathlib import Path
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.data import load_era5_month_raw, load_ifs, load_imerg, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE

hours = range(24)

latitude_vals = DEFAULT_LATITUDE_RANGE
longitude_vals = DEFAULT_LONGITUDE_RANGE

parser = ArgumentParser(description='Gather monthly rainfall data.')
parser.add_argument('--year', type=int, help='Years to process')
parser.add_argument('--output-dir', type=str, help='output directory', default=str(HOME/ 'rainfall_data'))
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

year = args.year
total_imerg_rainfall_dict = {}
total_ifs_rainfall_dict = {}
monthly_rainfall_imerg = {}
monthly_rainfall_ifs = {}
monthly_rainfall_era5 = {}
daily_maximum = {}

print(args)

for month in tqdm(np.arange(1, 13)):
    
    total_imerg_rainfall_dict[month] = {}
    total_ifs_rainfall_dict[month] = {}
    
    monthly_rainfall_imerg[month] = 0
    monthly_rainfall_ifs[month] = 0
    
    imerg_daily_rainfall_arrays = []
    ifs_daily_rainfall_arrays = []
    era5_monthly_rainfall_array = None
    
    daily_maximum[month] = {}

    day_range = range(1,3)
    for day in range(1,3):
    # for day in range(1, monthrange(year, month)[1]+1):
        total_imerg_rainfall_dict[month][day] = 0
        total_ifs_rainfall_dict[month][day] = 0
        
        daily_maximum[month][day] = 0
        
        n_imerg_hours = 0
        n_ifs_hours = 0
        for hour in hours:
            
            try:
                precip_hourly_values = load_imerg(datetime(year, month, day), hour=hour,
                                data_dir='/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/half_hourly/final/',
                            latitude_vals=latitude_vals, longitude_vals=longitude_vals)
                
                total_imerg_rainfall_dict[month][day] += precip_hourly_values
                
                if precip_hourly_values.max() > daily_maximum[month][day]:
                    daily_maximum[month][day] = precip_hourly_values.max()
                n_imerg_hours += 1
            except FileNotFoundError:
                print(f'No IMERG data found for {datetime(year, month, day)}, {hour}')
                pass

            if year >= 2016:
                # IFS
                try:
                    ds_ifs = load_ifs('tp', datetime(year, month, day), hour, log_precip=False, norm=False, 
                                    latitude_vals=latitude_vals, longitude_vals=longitude_vals)
                    total_ifs_rainfall_dict[month][day] += ds_ifs
                    
                    n_ifs_hours += 1
                except FileNotFoundError:
                    pass
            
        # correct for missing IFS values
        if n_ifs_hours < 24:
            if n_ifs_hours > 0:
                total_ifs_rainfall_dict[month][day] = total_ifs_rainfall_dict[month][day]* 24 / n_ifs_hours
            else:
                total_ifs_rainfall_dict[month][day] = np.nan
        
        if n_imerg_hours < 24:
            if n_imerg_hours > 0:
                total_imerg_rainfall_dict[month][day] = total_imerg_rainfall_dict[month][day]* 24 / n_imerg_hours
            else:
                total_imerg_rainfall_dict[month][day] = np.nan
        
        monthly_rainfall_imerg[month] += total_imerg_rainfall_dict[month][day]
        monthly_rainfall_ifs[month] += total_ifs_rainfall_dict[month][day]
        
        imerg_daily_rainfall_arrays.append(total_imerg_rainfall_dict[month][day])
        ifs_daily_rainfall_arrays.append(total_ifs_rainfall_dict[month][day])

    # # # era5
    # try:
    #     era5_ds = load_era5_month_raw('tp', year=year, month=month, era_data_dir='/bp1/geog-tropical/data/ERA-5/day')
    #     era5_ds = era5_ds.sel(latitude=latitude_vals, method='backfill').sel(longitude=longitude_vals, method='backfill')
        
    #     era5_ds_monthly = era5_ds.sum('time')                             
    #     # assert len(era5_ds.latitude.values) == len(tmp_lat_vals)
    #     # assert len(era5_ds.longitude.values) == len(tmp_lon_vals)
        
    #     monthly_rainfall_era5[month] = era5_ds_monthly['tp'].values
    #     era5_ds.close()
    #     era5_ds_monthly.close()
    # except FileNotFoundError:
    #     pass
    
    imerg_days = list(total_imerg_rainfall_dict[month].keys())
    ifs_days = list(total_ifs_rainfall_dict[month].keys())
    
    time_dim = pd.date_range(start=f'{year}-{month}-1', end=f'{year}-{month}-{day}', freq='D')
    # only record full months
    if set(imerg_days) == set(day_range): 
        
        imerg_daily_rainfall_array = np.stack(imerg_daily_rainfall_arrays, axis=0)
        monthly_imerg_ds = xr.DataArray(imerg_daily_rainfall_array, coords={'time': time_dim, 'lat': latitude_vals, 'lon': longitude_vals}, dims=['time', 'lat', 'lon'])   
        monthly_imerg_ds.to_netcdf(os.path.join(args.output_dir, f'daily_imerg_rainfall_{month}_{year}.pkl'))
    
    if len(ifs_daily_rainfall_arrays) > 0 and set(ifs_days) == set(day_range): 
        ifs_daily_rainfall_array = np.stack(ifs_daily_rainfall_arrays, axis=0)
        monthly_ifs_ds = xr.DataArray(ifs_daily_rainfall_array, coords={'time': time_dim, 'lat': latitude_vals, 'lon': longitude_vals}, dims=['time', 'lat', 'lon'])
        monthly_ifs_ds.to_netcdf(os.path.join(args.output_dir, f'daily_ifs_rainfall_{month}_{year}.pkl'))
    
    

    
    # with open(os.path.join(args.output_dir, f'daily_imerg_rainfall_{year}.pkl'), 'wb+') as ofh:
    #     pickle.dump(total_imerg_rainfall_dict, ofh)
        
    # with open(os.path.join(args.output_dir, f'daily_imerg_maxima_{year}.pkl'), 'wb+') as ofh:
    #     pickle.dump(total_imerg_rainfall_dict, ofh)
    
    # if year >= 2016:
    #     with open(os.path.join(args.output_dir, f'daily_ifs_rainfall_{year}.pkl'), 'wb+') as ofh:
    #         pickle.dump(total_ifs_rainfall_dict, ofh)
            
    #     with open(os.path.join(args.output_dir, f'monthly_ifs_rainfall_{year}.pkl'), 'wb+') as ofh:
    #         pickle.dump(monthly_rainfall_ifs, ofh)

    # with open(os.path.join(args.output_dir, f'monthly_imerg_rainfall_{year}.pkl'), 'wb+') as ofh:
    #     pickle.dump(monthly_rainfall_imerg, ofh)

    # with open(os.path.join(args.output_dir, f'monthly_era5_rainfall_{year}.pkl'), 'wb+') as ofh:
    #     pickle.dump(monthly_rainfall_era5, ofh)