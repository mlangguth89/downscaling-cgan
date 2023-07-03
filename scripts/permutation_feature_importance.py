import pickle
import os, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from properscoring import crps_ensemble
from random import shuffle

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.model import setupmodel
from dsrnngan.utils import utils, read_config
from dsrnngan.data.data_generator import DataGenerator
from dsrnngan.data.data import all_fcst_hours
from dsrnngan.evaluation.evaluation import generate_gan_sample
from dsrnngan.evaluation.scoring import mse
from dsrnngan.data.data import denormalise, DATA_PATHS, get_obs_dates

parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--model-type', type=str, help='Choice of model type', default=str(HOME))
parser.add_argument('--num-extra-models', type=int, help='Number of models to assess on, around the best performing model.',
                    default=5)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

##############################################################
# Load previously saved evaluation data
##############################################################

model_type = 'cropped' # Currently this is the only version with the correct data

log_folders = {
               'cropped': '/user/work/uz22147/logs/cgan/5c577a485fbd1a72/n4000_201806-201905_e10'}

log_folder = log_folders[model_type]
best_model_number = int(utils.get_best_model_number(log_folder=log_folder))
# Get config
base_folder = '/'.join(log_folder.split('/')[:-1])
config = utils.load_yaml_file(os.path.join(base_folder, 'setup_params.yaml'))

model_config, _, ds_config, data_config, gen_config, dis_config, train_config, val_config = read_config.get_config_objects(config)
latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(config)

all_model_numbers = utils.get_checkpoint_model_numbers(log_folder=log_folders[model_type])
other_model_distances = sorted([(mn,np.abs(mn-best_model_number)) for mn in all_model_numbers if mn != best_model_number], key=lambda x: x[1])
model_numbers = [best_model_number ] + [item[0] for item in other_model_distances[:args.num_extra_models]]


var_name_lookup = {'lo_res_inputs': data_config.input_fields,
                   'hi_res_inputs': ['orography', 'lsm']}

n_samples = 1000
ensemble_size = 30

if args.debug:
    n_samples = 2

read_config.set_gpu_mode({'gpu_mem_incr': True, 'use_gpu': True, 'disable_tf32': False})

for model_number in model_numbers:

    ##############################################################
    # Load generative model
    ##############################################################

    gen = setupmodel.load_model_from_folder(str(Path(log_folder).parents[0]), model_number=model_number)

    index_max_lookup = {'lo_res_inputs': len(data_config.input_fields), 'hi_res_inputs': data_config.constant_fields}

    if args.debug:
        index_max_lookup['lo_res_inputs'] = 2
        index_max_lookup['hi_res_inputs'] = 1

    date_range = utils.date_range_from_year_month_range(['201806', '201905'])
    dates = get_obs_dates(date_range[0], date_range[-1], 
                            obs_data_source='imerg', data_paths=DATA_PATHS, hour='random')
    hours = np.repeat(all_fcst_hours, len(dates))

    scores = {}
    for permutation_type in ['lo_res_inputs', 'hi_res_inputs']:
        print(f'Performing permutations for {permutation_type}', flush=True)
        
        for permute_index in range(index_max_lookup[permutation_type]):

            ##############################################################
            # Create permuted data generator
            ##############################################################

            data_gen = DataGenerator(dates=dates,
                                    forecast_data_source=data_config.fcst_data_source,
                                    observational_data_source=data_config.obs_data_source,
                                    fields=data_config.input_fields,
                                    latitude_range=latitude_range,
                                    longitude_range=longitude_range,
                                    batch_size=1,
                                    shuffle=False,
                                    constant_fields=True,
                                    normalise=True,
                                    downsample=model_config.downsample,
                                    data_paths=DATA_PATHS,
                                    hour=hours)
            
            # Same date/hour combination but with shuffle=True
            permuted_data_gen = DataGenerator(dates=dates,
                                    forecast_data_source=data_config.fcst_data_source,
                                    observational_data_source=data_config.obs_data_source,
                                    fields=data_config.input_fields,
                                    latitude_range=latitude_range,
                                    longitude_range=longitude_range,
                                    batch_size=1,
                                    shuffle=True,
                                    constant_fields=True,
                                    normalise=True,
                                    downsample=model_config.downsample,
                                    data_paths=DATA_PATHS,
                                    hour=hours)

            crps_scores = []
            data_idx = 0 
            shuffled_data_idx = 0  
            for ix in tqdm(range(n_samples)):
                
                success = False
                for n in range(5):
                    if success:
                        continue
                    try:
                        inputs, outputs = data_gen[data_idx]
                        obs = outputs['output'][0,...]
                        cond = inputs['lo_res_inputs']
                        const = inputs['hi_res_inputs']
                        
                        data_idx += 1
                        success = True
                    except FileNotFoundError:
                        data_idx += 1
                        continue
                if not success:
                    raise FileNotFoundError 
                
                success=False
                for n in range(10):
                    if success:
                        continue
                    try:
                        shuffled_inputs, _ = permuted_data_gen[shuffled_data_idx]
                        shuffled_data_idx += 1
                        success = True

                    except FileNotFoundError:
                        shuffled_data_idx += 1
                        continue
                    
                if not success:
                    raise FileNotFoundError 
                        
                    
                if permutation_type == 'lo_res_inputs':
                    cond[...,permute_index] = shuffled_inputs['lo_res_inputs'][...,permute_index]
                else:
                    const[...,permute_index] = shuffled_inputs['hi_res_inputs'][...,permute_index]                    
                        

                samples_gen = generate_gan_sample(gen, 
                                        cond=cond,
                                        const=const, 
                                        noise_channels=config['GENERATOR']['noise_channels'], 
                                        ensemble_size=ensemble_size, 
                                        batch_size=1, 
                                        )
                
                for ii in range(ensemble_size):
                    
                    sample_gen = samples_gen[ii][0, :, :, 0]
                    
                    # sample_gen shape should be [n, h, w, c] e.g. [1, 940, 940, 1]
                    sample_gen = denormalise(sample_gen)

                    samples_gen[ii] = sample_gen
                
                samples_gen = np.stack(samples_gen, axis=-1)
                
                # Evaluate a metric on the data
                crps_truth_input = np.expand_dims(obs, 0)
                crps_gen_input = np.expand_dims(samples_gen, 0)
                crps_score_grid = crps_ensemble(crps_truth_input, crps_gen_input)
                crps_score = crps_score_grid.mean()
                crps_scores.append(crps_score)
                        
            scores[var_name_lookup[permutation_type][permute_index]] = {'crps': np.mean(crps_scores)}
            
    with open(os.path.join(log_folder, f'permutation_scores_{model_number}.pkl'), 'wb+') as ofh:
        pickle.dump(scores, ofh)