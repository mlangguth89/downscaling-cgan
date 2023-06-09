import pickle
import os, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from properscoring import crps_ensemble

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.model import setupmodel
from dsrnngan.utils import utils, read_config
from dsrnngan.data.data_generator import PermutedDataGenerator
from dsrnngan.evaluation.evaluation import generate_gan_sample
from dsrnngan.data.data import denormalise, all_ifs_fields

parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--model-type', type=str, help='Choice of model type', default=str(HOME))
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

##############################################################
# Load previously saved evaluation data
##############################################################

model_type = 'cropped_v2' # Currently this is the only version with the correct data

log_folders = {
               'cropped_v2': '/user/work/uz22147/logs/cgan/5c577a485fbd1a72/n4000_201806-201905_e10'}


model_number = int(utils.get_best_model_number(log_folder=log_folders[model_type]))

if model_type not in log_folders:
    raise ValueError('Model type not found')


log_folder = log_folders[model_type]
with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
# Get config
base_folder = '/'.join(log_folder.split('/')[:-1])
config = utils.load_yaml_file(os.path.join(base_folder, 'setup_params.yaml'))
    
truth_array = arrays['truth']
samples_gen_array = arrays['samples_gen']
fcst_array = arrays['fcst_array']
cond = arrays['cond']
const = arrays['const']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]

(n_samples, width, height, ensemble_size) = samples_gen_array.shape

if args.debug:
    n_samples = 10

read_config.set_gpu_mode({'gpu_mem_incr': True, 'use_gpu': True, 'disable_tf32': False})
##############################################################
# Load generative model
##############################################################

gen = setupmodel.load_model_from_folder(str(Path(log_folder).parents[0]), model_number=model_number)

index_max_lookup = {'lo_res_inputs': len(all_ifs_fields), 'hi_res_inputs': config['DATA']['constant_fields']}

if args.debug:
    index_max_lookup['lo_res_inputs'] = 2

scores = {}

for permutation_type in ['lo_res_inputs', 'hi_res_inputs']:
    print(f'Performing permutations for {permutation_type}', flush=True)
    scores[permutation_type] = {}
    for permute_index in range(index_max_lookup[permutation_type]):

        ##############################################################
        # Create permuted data generator
        ##############################################################
        # Inputting cond and const like this because the evaluation script currently adds an extra dimension
        data_gen = PermutedDataGenerator(lo_res_inputs=cond[:,0,:,:,:], hi_res_inputs=const[:,0,:,:,:], outputs=truth_array, dates=dates, hours=hours,
                                        input_permutation_config={'type': permutation_type, 
                                                                  'permute_index': permute_index})

        # TODO: decide on a metric!
        crps_scores = []        
        for ix in tqdm(range(n_samples)):
            
            inputs, outputs = data_gen[ix]
            obs = outputs['output'][0,...]
            
            samples_gen = generate_gan_sample(gen, 
                                    cond=inputs['lo_res_inputs'],
                                    const=inputs['hi_res_inputs'], 
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
        
        scores[permutation_type][permute_index]['crps'] = np.mean(crps_scores)
        
with pickle.open(os.path.join(log_folder, 'permutation_scores.pkl'), 'wb+') as ofh:
    pickle.dump(scores, ofh)