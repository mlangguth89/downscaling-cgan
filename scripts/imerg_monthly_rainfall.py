# Get an idea of distribution of rainfall
import os, sys
import numpy as np
import pickle

from pathlib import Path
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.data import load_imerg_raw, load_era5_month_raw
from calendar import monthrange

hours = range(24)

latitude_vals = np.arange(-12, 16, 0.25)
longitude_vals = np.arange(22, 50, 0.25)

total_rainfall_dict = {}
monthly_rainfall_dict = {}
monthly_rainfall_dict_era5 = {}

for year in [2016, 2017, 2018, 2019]:

    total_rainfall_dict[year] = {}
    monthly_rainfall_dict[year] = {}
    monthly_rainfall_dict_era5[year] = {}

    print('Year=', year)
    
    for month in tqdm(np.arange(1, 13)):
        
        total_rainfall_dict[year][month] = {}
        monthly_rainfall = 0
        
        for day in range(1, monthrange(year, month)[1]+1):
            daily_rainfall = 0
            hourly_rainfall = []
            for hour in hours:
                # This sselects with method = 'nearest' so grid resolution should match era5
                ds_hr = load_imerg_raw(year, month, day, hour, 
                                       imerg_data_dir='/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/half_hourly/final')
                ds_hr = ds_hr.sel(lat=latitude_vals, method='nearest').sel(lon=longitude_vals, method='nearest')

                # average over half hours
                precip_hourly_values = ds_hr['precipitationCal']
                hourly_rainfall.append(precip_hourly_values)
                
                daily_rainfall += precip_hourly_values
                tmp_lat_vals = ds_hr.lat.values
                tmp_lon_vals = ds_hr.lon.values
                ds_hr.close()
                            
            total_rainfall_dict[year][month][day] = daily_rainfall
            monthly_rainfall += daily_rainfall
        monthly_rainfall_dict[year][month] = monthly_rainfall
        
        # era5
        era5_ds = load_era5_month_raw('tp', year=year, month=month, era_data_dir='/bp1/geog-tropical/users/uz22147/east_africa_data/ERA5')
        era5_ds = era5_ds.sel(latitude=latitude_vals, method='nearest').sel(longitude=longitude_vals, method='nearest')
        
        era5_ds_monthly = era5_ds.sum('time')                             
        assert len(era5_ds.latitude.values) == len(tmp_lat_vals)
        assert len(era5_ds.longitude.values) == len(tmp_lon_vals)
        
        monthly_rainfall_dict_era5[year][month] = era5_ds_monthly['tp'].values
        era5_ds.close()
        era5_ds_monthly.close()

        with open('total_rainfall.pkl', 'wb+') as ofh:
            pickle.dump(total_rainfall_dict, ofh)

        with open('monthly_rainfall.pkl', 'wb+') as ofh:
            pickle.dump(monthly_rainfall_dict, ofh)
            
        with open('monthly_rainfall_era5.pkl', 'wb+') as ofh:
            pickle.dump(monthly_rainfall_dict_era5, ofh)