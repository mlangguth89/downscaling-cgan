import os
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe
import os
import sys
from pathlib import Path
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import unittest
import yaml
from glob import glob
        
HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))


from dsrnngan import data, read_config
from dsrnngan.noise import NoiseGenerator
from dsrnngan.evaluation import setup_inputs
from dsrnngan.data import DATA_PATHS

model_folder='/user/home/uz22147/logs/cgan'

model_weights_root = os.path.join(model_folder, "models")
config_path = os.path.join(model_folder, 'setup_params.yaml')
df_dict = read_config.read_config()['DOWNSCALING']


with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
batch_size = 1  # setup_params["TRAIN"]["batch_size"]
fcst_data_source=setup_params['DATA']['fcst_data_source']
obs_data_source=setup_params['DATA']['obs_data_source']
input_channels = setup_params['DATA']['input_channels']
constant_fields = setup_params['DATA']['constant_fields']
input_image_width = setup_params['DATA']['input_image_width']
output_image_width = input_image_width * df_dict['downscaling_factor']
constants_image_width = input_image_width
problem_type = setup_params["GENERAL"]["problem_type"]
downsample = setup_params['GENERAL']['downsample']
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
num_batches = setup_params["EVAL"]["num_batches"]
add_noise = setup_params["EVAL"]["add_postprocessing_noise"]

val_range = setup_params['VAL'].get('val_range')

min_latitude = setup_params['DATA']['min_latitude']
max_latitude = setup_params['DATA']['max_latitude']
latitude_step_size = setup_params['DATA']['latitude_step_size']
min_longitude = setup_params['DATA']['min_longitude']
max_longitude = setup_params['DATA']['max_longitude']
longitude_step_size = setup_params['DATA']['longitude_step_size']
latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)

class TestEvaluation(unittest.TestCase):
    
    def test_setup_inputs(self):
        
        

        most_recent_model = sorted(glob(os.path.join(model_weights_root, '*.h5')))[-1]
        
        print('setting up inputs')
        gen, batch_gen_valid = setup_inputs(mode=mode,
                                        arch=arch,
                                        records_folder='/user/work/uz22147/tfrecords/d34d309eb0e00b04',
                                        fcst_data_source=fcst_data_source,
                                        obs_data_source=obs_data_source,
                                        latitude_range=latitude_range,
                                        longitude_range=longitude_range,
                                        hour=18,
                                        downscaling_steps=df_dict["steps"],
                                        validation_range=val_range,
                                        downsample=downsample,
                                        input_channels=input_channels,
                                        constant_fields=constant_fields,
                                        filters_gen=filters_gen,
                                        filters_disc=filters_disc,
                                        noise_channels=noise_channels,
                                        latent_variables=latent_variables,
                                        padding=padding,
                                        data_paths=DATA_PATHS)

        print('loading weights')
        gen.load_weights(most_recent_model)

        denormalise_data = True
        rank_samples = 1

        batch_gen_iter = iter(batch_gen_valid)

        inputs, outputs = next(batch_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        truth = outputs['output']
        truth = np.expand_dims(np.array(truth), axis=-1)

        if denormalise_data:
            truth = data.denormalise(truth)

        noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)

        for ii in range(rank_samples):
            img_gen = gen.predict([cond, const, noise_gen()])
    
#     def test_quality_metrics_by_time(self):
        
#         # This works currently only after the test_main bits have been run
#         log_folder = str(HOME / 'system_tests' / 'data' / 'tmp')
#         quality_metrics_by_time(mode="GAN",
#                                 arch='forceconv',
#                                 fcst_data_source='era5',
#                                 obs_data_source='imerg',
#                                 load_constants=False,
#                                 val_years=2019,
#                                 log_fname=os.path.join(log_folder, 'qual.txt'),
#                                 weights_dir=os.path.join(log_folder, 'models'),
#                                 downsample=False,
#                                 model_numbers=[4],
#                                 batch_size=1,  # do inference 1 at a time, in case of memory issues
#                                 num_batches=2,
#                                 filters_gen=2,
#                                 filters_disc=2,
#                                 input_channels=5,
#                                 constant_fields=1,
#                                 latent_variables=1,
#                                 noise_channels=4,
#                                 rank_samples=2,
#                                 padding='reflect')
        
#     def test_rank_metrics_by_time(self):
        
#         log_folder = str(HOME / 'system_tests' / 'data' / 'tmp')
#         rank_metrics_by_time(mode="GAN",
#                                 arch='forceconv',
#                                 fcst_data_source='era5',
#                                 obs_data_source='imerg',
#                                 val_years=2019,
#                                 constant_fields=1,
#                                 load_constants=False,
#                                 log_fname=os.path.join(log_folder, 'rank.txt'),
#                                 weights_dir=os.path.join(log_folder, 'models'),
#                                 downsample=False,
#                                 weights=[0.4, 0.3, 0.2, 0.1],
#                                 add_noise=True,
#                                 noise_factor=0.001,
#                                 model_numbers=[4],
#                                 ranks_to_save=[4],
#                                 batch_size=1,  # ditto
#                                 num_batches=2,
#                                 filters_gen=2,
#                                 filters_disc=2,
#                                 input_channels=5,
#                                 latent_variables=1,
#                                 noise_channels=4,
#                                 padding='reflect',
#                                 rank_samples=10,
#                                 max_pooling=True,
#                                 avg_pooling=True)