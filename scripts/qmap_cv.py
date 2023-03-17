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
from dsrnngan.data import DEFAULT_LATITUDE_RANGE as latitude_range, DEFAULT_LONGITUDE_RANGE as longitude_range, denormalise
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


with open('/user/home/uz22147/repos/downscaling-cgan/201603_201803_2000_arrays.pkl', 'rb') as ifh:
    training_data = pickle.load(ifh)
    
imerg_train_data = denormalise(training_data['obs'])
ifs_train_data = denormalise(training_data['fcst_array'])

training_dates = [item[0] for item in training_data['dates']]
training_hours = [item[0] for item in training_data['hours']]


X = ifs_train_data.copy().reshape(ifs_train_data.shape[0], -1)
y = imerg_train_data.reshape(ifs_train_data.shape[0], -1)
months = [d.month for d in training_dates]

skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(X, months)

# quantile_thresholds = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
quantile_threshold = 0.9999
mae_vals = []
mae_95_vals = []

cv_mae_vals = []
cv_mae_95_vals = []

for i, (train_index, val_index) in enumerate(skf.split(X, months)):
    
    
    ifs_quantile_training_data = ifs_train_data[train_index, :, :]
    imerg_quantile_training_data = imerg_train_data[train_index, :, :]

    ifs_quantile_val_data = ifs_train_data[val_index, :, :]
    imerg_quantile_val_data = imerg_train_data[val_index, :, :]

    quantile_training_dates = np.array(training_dates)[train_index]
    quantile_val_dates = np.array(training_dates)[val_index]
    quantile_training_hours = np.array(training_hours)[train_index]
    quantile_val_hours = np.array(training_hours)[val_index]
    
    # quantile mapping by month, pixel-by-pixel

    date_hour_list = list(zip(quantile_training_dates, quantile_training_hours))
    date_hour_list_val = list(zip(quantile_val_dates, quantile_val_hours))

    date_chunks =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list if item[0].month in month_range] for month_range in month_ranges}
    date_indexes =  {k : [date_hour_list.index(item) for item in chunk] for k, chunk in date_chunks.items()}

    date_chunks_val =  {'_'.join([str(month_range[0]), str(month_range[-1])]): [item for item in date_hour_list_val if item[0].month in month_range] for month_range in month_ranges}
    date_indexes_val =  {k : [date_hour_list_val.index(item) for item in chunk] for k, chunk in date_chunks_val.items()}

    quantiles_by_month_range = {}

    fcst_corrected_single_pixel = np.empty(ifs_quantile_val_data.shape)
    fcst_corrected_single_pixel[:,:,:] = np.nan

    for t_name, d in tqdm(date_indexes.items(), total=len(month_ranges)):
        fcst = ifs_quantile_training_data[d, :, :]
        obs = imerg_quantile_training_data[d, :, :]
        
        quantiles_by_month_range[t_name] = {'fcst': np.quantile(fcst, quantile_locs, axis=0), 'obs': np.quantile(obs,  quantile_locs, axis=0)}
        
        # Correct the forecast in the validation set using these quantiles
        d_val = date_indexes_val[t_name]
        fcst_val = ifs_quantile_val_data[d_val, :, :]
        obs_val = imerg_quantile_val_data[d_val, :, :]
        
        (n_val_samples, x_vals, y_vals) = fcst_val.shape
        
        for x in range(x_vals):
            for y in range(y_vals):
                fcst_corrected_single_pixel[d_val, x, y] = np.interp(fcst_val[:, x, y], 
                                                                    quantiles_by_month_range[t_name]['fcst'][:, x, y], 
                                                                    quantiles_by_month_range[t_name]['obs'][:, x, y])
    
    # quantile mapping by dividing into chunks
    quantiles_by_area_dict = {}
    corrected_fcst_dict = {'single_pixel': fcst_corrected_single_pixel}
    for n_chunks in tqdm(range(1,13)):
        quantile_areas = get_quantile_areas(dates=list(quantile_training_dates), month_ranges=month_ranges, latitude_range=latitude_range, 
                                            longitude_range=longitude_range, num_lat_lon_chunks=n_chunks, hours=quantile_training_hours)
        quantiles_by_area_dict[n_chunks] = get_quantiles_by_area(quantile_areas, fcst_data=ifs_quantile_training_data, obs_data=imerg_quantile_training_data, 
                                                quantile_locs=quantile_locs)
        
        corrected_fcst_dict[n_chunks] = get_quantile_mapped_forecast(fcst=ifs_quantile_val_data, dates=quantile_val_dates, 
                                                hours=quantile_val_hours, month_ranges=month_ranges, 
                                                quantile_areas=quantile_areas, 
                                                quantiles_by_area=quantiles_by_area_dict[n_chunks])


        with open(f'/user/home/uz22147/repos/downscaling-cgan/notebooks/tmp/corrected_fcst_dict_{i}.pkl', 'wb+') as ofh:
            pickle.dump(corrected_fcst_dict, ofh)

        with open(f'/user/home/uz22147/repos/downscaling-cgan/notebooks/tmp/quantiles_by_area_dict_{i}.pkl', 'wb+') as ofh:
            pickle.dump(quantiles_by_area_dict, ofh)
        
#     cv_mae_vals.append(np.abs(test_obs.flatten() - test_qmapped_fcst.flatten()).mean())
    
#     cv_mae_95_vals.append(mae_above_threshold(test_obs.flatten(), test_qmapped_fcst.flatten(), 
#                                                 percentile_threshold=0.95))

# mae_vals.append((quantile_threshold, np.mean(cv_mae_vals)))
# mae_95_vals.append((quantile_threshold, np.mean(cv_mae_95_vals)))

# with open(output_fp, 'wb+') as ofh:
#     pickle.dump({'mae': mae_vals, 'mae_95': mae_95_vals}, ofh)