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


from dsrnngan import data, read_config, setupdata
from dsrnngan.noise import NoiseGenerator
from dsrnngan.evaluation import setup_inputs, eval_one_chkpt
from dsrnngan.data import DATA_PATHS

model_folder='/user/home/uz22147/logs/cgan/d34d309eb0e00b04'

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

    def test_eval_one_chkpt(self):
        
        gen = setupdata.load_model_from_folder(model_folder='/user/home/uz22147/logs/cgan/d34d309eb0e00b04', model_number=38400)

        records_folder = '/user/work/uz22147/tfrecords/d34d309eb0e00b04/'
        _, data_gen_valid = setupdata.load_data_from_folder('/user/work/uz22147/tfrecords/d34d309eb0e00b04/')

        config = read_config.read_config(os.path.join(records_folder, 'local_config.yaml'))
        latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(config)
        
        noise_factor = config["EVAL"]["postprocessing_noise_factor"]
        
        eval_one_chkpt(
                   mode=mode,
                   gen=gen,
                   fcst_data_source=fcst_data_source,
                   data_gen=data_gen_valid,
                   noise_channels=noise_channels,
                   latent_variables=latent_variables,
                   num_images=2,
                   latitude_range=latitude_range,
                   longitude_range=longitude_range,
                   add_noise=add_noise,
                   ensemble_size=2,
                   noise_factor=noise_factor,
                   denormalise_data=True,
                   normalize_ranks=True,
                   show_progress=True,)