import sys, os
import pickle
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import xarray as xr
import numpy as np
from sklearn.model_selection import StratifiedKFold
from glob import glob

HOME = Path(os.getcwd())

sys.path.insert(1, str(HOME))


from dsrnngan.utils.utils import load_yaml_file
from dsrnngan.data.data import DEFAULT_LATITUDE_RANGE as latitude_range, DEFAULT_LONGITUDE_RANGE as longitude_range, denormalise
from dsrnngan.evaluation.benchmarks import QuantileMapper, quantile_map_grid
from dsrnngan.evaluation.scoring import mae_above_threshold
parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--output-dir', type=str, help='output directory', default=str(HOME))
args = parser.parse_args()

month_ranges = [[1,2], [3,4,5], [6,7,8,9], [10,11,12]]

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

fps = glob('/user/work/uz22147/quantile_training_data/*_744.pkl')

imerg_train_data = []
ifs_train_data = []
training_dates = []
training_hours = []

for fp in fps:
    with open(fp, 'rb') as ifh:
        training_data = pickle.load(ifh)
        
    imerg_train_data.append(denormalise(training_data['obs']))
    ifs_train_data.append(denormalise(training_data['fcst_array']))

    training_dates += [item[0] for item in training_data['dates']]
    training_hours += [item[0] for item in training_data['hours']]


imerg_train_data = np.concatenate(imerg_train_data, axis=0)
ifs_train_data = np.concatenate(ifs_train_data, axis=0)


X = ifs_train_data.copy().reshape(ifs_train_data.shape[0], -1)
y = imerg_train_data.reshape(ifs_train_data.shape[0], -1)
months = [d.month for d in training_dates]

skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(X, months)

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

    fcst_corrected_single_pixel = quantile_map_grid(array_to_correct=ifs_quantile_val_data, 
                      fcst_train_data=ifs_quantile_training_data, 
                      obs_train_data=imerg_quantile_training_data, 
                      quantiles=quantile_locs,
                      extrapolate='constant')

    # fcst_corrected_single_pixel = np.empty(ifs_quantile_val_data.shape)
    # fcst_corrected_single_pixel[:,:,:] = np.nan

    # for t_name, d in tqdm(date_indexes.items(), total=len(month_ranges)):
    #     fcst_train = ifs_quantile_training_data[d, :, :]
    #     obs_train = imerg_quantile_training_data[d, :, :]
                
    #     # Correct the forecast in the validation set using these quantiles
    #     d_val = date_indexes_val[t_name]
    #     fcst_val = ifs_quantile_val_data[d_val, :, :]
    #     obs_val = imerg_quantile_val_data[d_val, :, :]
        
    #     fcst_corrected_single_pixel[d_val, :, :] = quantile_map_grid(array_to_correct=fcst_val, 
    #                   fcst_train_data=fcst_train, 
    #                   obs_train_data=obs_train, 
    #                   quantiles=quantile_locs,
    #                   extrapolate='constant')
        
    # quantile mapping by dividing into chunks
    quantiles_by_area_dict = {}
    corrected_fcst_dict = {'single_pixel': fcst_corrected_single_pixel}
    for n_chunks in tqdm(np.arange(1,31,5)):
        
        qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, longitude_range=longitude_range,
                         num_lat_lon_chunks=n_chunks)
        qmapper.train(fcst_data=ifs_quantile_training_data, obs_data=imerg_quantile_training_data, 
                      training_dates=quantile_training_dates, training_hours=quantile_training_hours)
        corrected_fcst_dict[n_chunks] = qmapper.get_quantile_mapped_forecast(fcst=ifs_quantile_val_data, 
                                                                             dates=quantile_val_dates, 
                                                                             hours=quantile_val_hours)

    corrected_fcst_abs_err = {k: [] for k in corrected_fcst_dict.keys()}

    for k, v in tqdm(corrected_fcst_dict.items()):
        
        corrected_fcst_abs_err[k].append(np.mean(np.abs(v - imerg_quantile_val_data), axis=0))


    with open(f'/user/work/uz22147/quantile_training_data/cv_results/corrected_fcst_abs_err_{i}.pkl', 'wb+') as ofh:
        pickle.dump(corrected_fcst_abs_err, ofh)

    with open(f'/user/work/uz22147/quantile_training_data/cv_results/quantiles_by_area_dict_{i}.pkl', 'wb+') as ofh:
        pickle.dump(quantiles_by_area_dict, ofh)