import sys, os
import pickle
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import xarray as xr
import numpy as np
from sklearn.model_selection import StratifiedKFold

HOME = Path(os.getcwd())

sys.path.insert(1, str(HOME))


from dsrnngan.utils import load_yaml_file
from dsrnngan.data import DEFAULT_LATITUDE_RANGE as latitude_range, DEFAULT_LONGITUDE_RANGE as longitude_range
from dsrnngan.benchmarks import get_quantile_areas, get_quantile_mapped_forecast, get_quantiles_by_area
from dsrnngan.scoring import mae_above_threshold

parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--output-dir', type=str, help='output directory', default=str(HOME))
args = parser.parse_args()

month_ranges = [[1,2], [3,4,5], [6,7,8,9], [10,11,12]]

output_fp = os.path.join(args.output_dir, 'mae_vals.pkl')
print(output_fp)

# Quantiles
step_size = 0.001
range_dict = {0: {'start': 0.1, 'stop': 1, 'interval': 0.1, 'marker': '+', 'marker_size': 32},
              1: {'start': 1, 'stop': 10, 'interval': 1, 'marker': '+', 'marker_size': 256},
              2: {'start': 10, 'stop': 70, 'interval':10, 'marker': '+', 'marker_size': 512},
              3: {'start': 70, 'stop': 99.1, 'interval': 1, 'marker': '+', 'marker_size': 256},
              4: {'start': 99.1, 'stop': 99.91, 'interval': 0.1, 'marker': '+', 'marker_size': 128},
              5: {'start': 99.9, 'stop': 99.99, 'interval': 0.01, 'marker': '+', 'marker_size': 32 },
              6: {'start': 99.99, 'stop': 99.999, 'interval': 0.001, 'marker': '+', 'marker_size': 10}}
                  
percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
percentiles=np.concatenate(percentiles_list)
quantile_locs = [np.round(item / 100.0, 6) for item in percentiles]

lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]


imerg_train_data = []
ifs_train_data = []
ifs_dates = []
imerg_dates = []

for year in tqdm([2016, 2017]):
    for month in range(1,13):

        imerg_ds = xr.open_dataarray(f'/user/home/uz22147/repos/rainfall_data/daily_imerg_rainfall_{month}_{year}.nc')

        imerg_data = imerg_ds.sel(lat=latitude_range, method='nearest').sel(lon=longitude_range, method='nearest').values
        
        imerg_dates += [d.astype('M8[D]').astype('O') for d in imerg_ds.time.values]

        for t in range(imerg_data.shape[0]):
            
            snapshot = imerg_data[t, :, :]
            
            imerg_train_data.append(snapshot)
        
        try:
            ifs_ds = xr.open_dataarray(f'/user/home/uz22147/repos/rainfall_data/daily_ifs_rainfall_{month}_{year}.nc')

            ifs_data = ifs_ds.sel(lat=latitude_range, method='nearest').sel(lon=longitude_range, method='nearest').values

            for t in range(ifs_data.shape[0]):
                
                snapshot = ifs_data[t, :, :]
                
                ifs_train_data.append(snapshot)
            ifs_dates += [d.astype('M8[D]').astype('O') for d in ifs_ds.time.values]

        except:
            pass

imerg_train_data = np.stack(imerg_train_data, axis = 0)
ifs_train_data = np.stack(ifs_train_data, axis = 0)

# Make dates consistent
overlapping_dates = np.array(sorted(set(ifs_dates).intersection(imerg_dates)))

imerg_overlapping_date_ix = [n for n, item in enumerate(imerg_dates) if item in overlapping_dates]
ifs_overlapping_date_ix = [n for n, item in enumerate(ifs_dates) if item in overlapping_dates]

imerg_train_data = imerg_train_data[imerg_overlapping_date_ix, :, :]
ifs_train_data = ifs_train_data[ifs_overlapping_date_ix , :, :]


n_splits = 5
n_train_samples = ifs_train_data.shape[0]
X = ifs_train_data.copy().reshape(n_train_samples, -1)
y = obs_data=imerg_train_data.reshape(n_train_samples, -1)
months = [d.month for d in overlapping_dates]

quantile_thresholds = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
mae_vals = []
mae_95_vals = []

for quantile_threshold in tqdm(quantile_thresholds):
    # Stratify by month
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    skf.get_n_splits(X, months)

    cv_mae_vals = []
    cv_mae_95_vals = []
    
    for i, (train_index, test_index) in enumerate(skf.split(X, months)):
        
        # train
        train_dates = overlapping_dates[train_index]
        training_quantile_areas = get_quantile_areas(train_dates, month_ranges, latitude_range, longitude_range)
        
        training_quantiles_by_area = get_quantiles_by_area(training_quantile_areas, 
                                                           fcst_data=ifs_train_data[train_index, :, :], 
                                                           obs_data=imerg_train_data[train_index, :, :], 
                                                           quantile_locs=quantile_locs)

        # test
        test_dates = overlapping_dates[test_index]
        test_quantile_areas = get_quantile_areas(test_dates, month_ranges, latitude_range, longitude_range)

        test_qmapped_fcst = get_quantile_mapped_forecast(fcst=ifs_train_data[test_index, :, :], 
                                                         dates=test_dates, 
                                                         month_ranges=month_ranges, 
                                                         quantile_areas=test_quantile_areas, 
                                                         quantiles_by_area=training_quantiles_by_area, 
                                                         hours=None,
                                                         quantile_threshold=quantile_threshold)
        
        test_obs = imerg_train_data[test_index, :, :]
        
        cv_mae_vals.append(np.abs(test_obs.flatten() - test_qmapped_fcst.flatten()).mean())
        
        cv_mae_95_vals.append(mae_above_threshold(test_obs.flatten(), test_qmapped_fcst.flatten(), 
                                                  percentile_threshold=0.95))
    
    mae_vals.append((quantile_threshold, np.mean(cv_mae_vals)))
    mae_95_vals.append((quantile_threshold, np.mean(cv_mae_95_vals)))
    
    with open(output_fp, 'wb+') as ofh:
        pickle.dump({'mae': mae_vals, 'mae_95': mae_95_vals}, ofh)