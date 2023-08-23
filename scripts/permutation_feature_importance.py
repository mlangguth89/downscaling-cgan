import pickle
import os, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from properscoring import crps_ensemble
from random import shuffle
from types import SimpleNamespace
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

parser.add_argument('--log-folder', type=str, help='model log folder', required=True)
parser.add_argument('--best-model-number', type=int, help='model number', required=True)
parser.add_argument('--num-extra-models', type=int, help='Number of models to assess on, around the best performing model.',
                    default=5)
parser.add_argument('--ym-ranges', type=list, help='Date range to perform experiment on.',
                    default=['202010','202109'])
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

##############################################################
# Load previously saved evaluation data
##############################################################


log_folder = args.log_folder
best_model_number = args.best_model_number

# Get config
base_folder = '/'.join(log_folder.split('/')[:-1])
data_config = read_config.read_data_config(config_folder=base_folder)
model_config = read_config.read_model_config(config_folder=base_folder)

latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(data_config=data_config)

all_model_numbers = utils.get_checkpoint_model_numbers(log_folder=base_folder)
other_model_distances = sorted([(mn,np.abs(mn-best_model_number)) for mn in all_model_numbers if mn != best_model_number], key=lambda x: x[1])
model_numbers = [best_model_number ] + [item[0] for item in other_model_distances[:args.num_extra_models]]


var_name_lookup = {'lo_res_inputs': data_config.input_fields,
                   'hi_res_inputs': ['orography', 'lsm']}

n_samples = 1000
ensemble_size = 30

if args.debug:
    n_samples = 2

read_config.set_gpu_mode(SimpleNamespace(**{'gpu_mem_incr': True, 'use_gpu': True, 'disable_tf32': False}))

for model_number in model_numbers:

    ##############################################################
    # Load generative model
    ##############################################################

    gen = setupmodel.load_model_from_folder(str(Path(log_folder).parents[0]), model_number=model_number)

    index_max_lookup = {'lo_res_inputs': len(data_config.input_fields), 'hi_res_inputs': len(data_config.constant_fields)}

    if args.debug:
        index_max_lookup['lo_res_inputs'] = 1
        index_max_lookup['hi_res_inputs'] = 1

    date_range = utils.date_range_from_year_month_range(args.ym_ranges)
    # dates = get_obs_dates(date_range[0], date_range[-1], 
    #                         obs_data_source='imerg', data_paths=DATA_PATHS, hour='random')
    hours = np.repeat(all_fcst_hours, len(date_range))

    scores = {}
    for permutation_type in ['lo_res_inputs', 'hi_res_inputs']:
        print(f'Performing permutations for {permutation_type}', flush=True)
        
        for permute_index in range(index_max_lookup[permutation_type]):

            ##############################################################
            # Create permuted data generator
            ##############################################################

            data_gen = DataGenerator(dates=date_range,
                                    batch_size=1,
                                    data_config=data_config,
                                    shuffle=False,
                                    hour=hours)
            
            # Same date/hour combination but with shuffle=True
            permuted_data_gen = DataGenerator(dates=date_range,                                              
                                    data_config=data_config,
                                    batch_size=1,
                                    shuffle=True,
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
                                        noise_channels=model_config.generator.noise_channels, 
                                        ensemble_size=ensemble_size, 
                                        batch_size=1, 
                                        )
                
                for ii in range(ensemble_size):
                    
                    sample_gen = samples_gen[ii][0, :, :, 0]
                    
                    # sample_gen shape should be [n, h, w, c] e.g. [1, 940, 940, 1]
                    if data_config.output_normalisation is not None:
                        sample_gen = denormalise(sample_gen, normalisation_type=data_config.output_normalisation)

                    samples_gen[ii] = sample_gen
                
                samples_gen = np.stack(samples_gen, axis=-1)
                
                # Evaluate a metric on the data
                # TODO: bootstrap this.
                crps_truth_input = np.expand_dims(obs, 0)
                crps_gen_input = np.expand_dims(samples_gen, 0)
                crps_score_grid = crps_ensemble(crps_truth_input, crps_gen_input)
                crps_score = crps_score_grid.mean()
                crps_scores.append(crps_score)
                        
            scores[var_name_lookup[permutation_type][permute_index]] = {'crps': np.mean(crps_scores)}
            
    with open(os.path.join(log_folder, f'permutation_scores_{model_number}.pkl'), 'wb+') as ofh:
        pickle.dump(scores, ofh)