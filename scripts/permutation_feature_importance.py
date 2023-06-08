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
from dsrnngan.utils import utils
from dsrnngan.data.data_generator import PermutedDataGenerator
from dsrnngan.evaluation.evaluation import generate_gan_sample

parser = ArgumentParser(description='Cross validation for selecting quantile mapping threshold.')

parser.add_argument('--output-dir', type=str, help='output directory', default=str(HOME))
parser.add_argument('--model-type', type=str, help='Choice of model type', default=str(HOME))
args = parser.parse_args()

##############################################################
# Load previously saved evaluation data
##############################################################

model_type = args.model_type

log_folders = {'basic': '/user/work/uz22147/logs/cgan/d9b8e8059631e76f/n1000_201806-201905_e50',
               'full_image': '/user/work/uz22147/logs/cgan/43ae7be47e9a182e_full_image/n1000_201806-201905_e50',
               'cropped': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n2000_201806-201905_e20',
               'cropped_4000': '/user/work/uz22147/logs/cgan/ff62fde11969a16f/n4000_201806-201905_e10',
               'cropped_v2': '/user/work/uz22147/logs/cgan/5c577a485fbd1a72/n4000_201806-201905_e10'}


model_number = utils.get_best_model_number(log_folder=log_folders[model_type])

if model_type not in log_folders:
    raise ValueError('Model type not found')


log_folder = log_folders[model_type]
with open(os.path.join(log_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)
    
truth_array = arrays['truth']
samples_gen_array = arrays['samples_gen']
fcst_array = arrays['fcst_array']
cond = arrays['cond']
const = arrays['const']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]

##############################################################
# Load generative model
##############################################################

gen = setupmodel.load_model_from_folder(str(Path(log_folder).parents[0]), model_number=model_number)

##############################################################
# Create permuted data generator
##############################################################

data_gen = PermutedDataGenerator(lo_res_inputs=cond, hi_res_inputs=const, outputs=truth_array, dates=dates, hours=hours,
                                 input_permutation_config={'type': 'lo_res_inputs', 'permute_index': 1})

# TODO: decide on a metric!
generate_gan_sample