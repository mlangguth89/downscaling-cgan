import numpy as np
import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from datetime import date
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan import data

# Load iMERG and IFS data

latitude_vals = np.arange(-11.95, 15.95, 0.1)
longitude_vals = np.arange(25.05, 49.05, 0.1)
dates = pd.date_range(start=date(2018,10,1), end=date(2018,12,31))
dates = [d.date() for d in dates]
hours = range(24)
obs_data = []
fcst_data = []
datetimes = []

for d in tqdm(dates):
    for hour in hours:
        
        try:
            ds_imerg = data.load_imerg(date=d, hour=hour, latitude_vals=latitude_vals,
                                        longitude_vals=longitude_vals)
            ds_ifs = data.load_ifs('tp', d, hour, log_precip=False, norm=False, latitude_vals=latitude_vals, longitude_vals=longitude_vals)
            
            obs_data.append(ds_imerg)
            fcst_data.append(ds_ifs)
            
            datetimes.append((d, hour))
        except FileNotFoundError:
            print(f"Nothing found for {d.strftime('%Y%m%d')}, {hour}")

truth_array = np.stack(obs_data, axis=0)
fcst_array = np.stack(fcst_data, axis=0)
    

with open(os.path.join(f"fss_data_{dates[0].strftime('%Y%m%d')}-{dates[-1].strftime('%Y%m%d')}.pkl"), 'wb+') as ofh:
    pickle.dump({'truth': truth_array, 'fcst': fcst_array, 'datetimes': datetimes}, ofh, pickle.HIGHEST_PROTOCOL)