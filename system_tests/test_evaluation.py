import os
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe
import os
import sys
import tempfile
import pickle
import copy

from datetime import datetime
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
from dsrnngan.evaluation import setup_inputs, eval_one_chkpt, evaluate_multiple_checkpoints
from dsrnngan.data import DATA_PATHS, all_ifs_fields, get_ifs_filepath
from system_tests.test_main import create_example_model, test_config, test_data_paths

model_config, local_config, ds_config, data_config, gen_config, dis_config, train_config, val_config = read_config.get_config_objects(test_config)

train_config.batch_size = 1  # setup_params["TRAIN"]["batch_size"]
output_image_width = data_config.input_image_width * ds_config.downscaling_factor
constants_image_width = data_config.input_image_width

lat_range = np.arange(0, 1.1, 0.1) # deliberately asymettrical to test for non-square images
lon_range = np.arange(33, 34, 0.1)

data_folder = HOME / 'system_tests' / 'data'
constants_path = str(data_folder / 'constants')
ifs_path = str(data_folder / 'IFS')
imerg_folder = str(data_folder / 'IMERG/half_hourly/final')

data_paths = copy.deepcopy(DATA_PATHS)
data_paths['GENERAL']['CONSTANTS'] = constants_path


class TestEvaluation(unittest.TestCase):
        
    def setUp(self) -> None:
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name
        self.tmp_data_folder = test_data_paths['TFRecords']['tfrecords_path']
        self.config = test_config
        
        self.config['VAL']['val_range'] = ['201707']
        self.config['VAL']['val_size'] = 5
        self.config['VAL']['ensemble_size'] = 2
        
        if not os.path.isdir(self.tmp_data_folder):
            # Create a dummy model if one doesn't already exist
            self.records_folder, self.model_folder = create_example_model(config=test_config, data_paths=test_data_paths)
        else:
            self.records_folder = '/'.join(glob(os.path.join(self.tmp_data_folder, '*/*.tfrecords'))[0].split('/')[:-1])
            self.model_folder = '/'.join(glob(os.path.join(self.tmp_data_folder, '*/*/models*'))[0].split('/')[:-1])
        
        # Create dummy stats data
        if not os.path.isdir(os.path.join(constants_path, 'tp')):
    
            for field in all_ifs_fields:
                fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, ifs_path)
                ds = xr.load_dataset(fp)
                
                # Mock IFS stats dicts
                var_name = list(ds.data_vars)[0]
                stats = {'min': np.abs(ds[var_name]).min().values,
                    'max': np.abs(ds[var_name]).max().values,
                    'mean': ds[var_name].mean().values,
                    'std': ds[var_name].std().values}
                
                # Saving it as 2017 since that's the defualt 
                output_fp = f'{constants_path}/IFS_norm_{field}_2017_lat0-0lon33-33.pkl'
                with open(output_fp, 'wb') as f:
                    pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
                    
    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        return super().tearDown()
        
    def test_setup_inputs(self):
        
        most_recent_model = sorted(glob(os.path.join(self.model_folder, '*.h5')))[-1]
        
        print('setting up inputs')
        gen, batch_gen_valid = setup_inputs(mode=model_config.mode,
                                        arch=model_config.architecture,
                                        records_folder=self.records_folder,
                                        fcst_data_source=data_config.fcst_data_source,
                                        obs_data_source=data_config.obs_data_source,
                                        latitude_range=lat_range,
                                        longitude_range=lon_range,
                                        hour=18,
                                        downscaling_steps=ds_config.steps,
                                        validation_range=val_config.val_range,
                                        downsample=model_config.downsample,
                                        input_channels=data_config.input_channels,
                                        constant_fields=data_config.constant_fields,
                                        filters_gen=gen_config.filters_gen,
                                        filters_disc=dis_config.filters_disc,
                                        noise_channels=gen_config.noise_channels,
                                        latent_variables=gen_config.latent_variables,
                                        padding=model_config.padding,
                                        data_paths=DATA_PATHS,
                                        shuffle=True)

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

        noise_shape = np.array(cond)[0, ..., 0].shape + (gen_config.noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=train_config.batch_size)

        for ii in range(rank_samples):
            img_gen = gen.predict([cond, const, noise_gen()])

    def test_eval_one_chkpt(self):
        
        gen = setupdata.load_model_from_folder(model_folder=self.model_folder, model_number=2)
        _, data_gen_valid = setupdata.load_data_from_config(self.config, data_paths=data_paths, hour=17)
                
        eval_one_chkpt(
                   mode=model_config.mode,
                   gen=gen,
                   fcst_data_source=data_config.fcst_data_source,
                   data_gen=data_gen_valid,
                   noise_channels=gen_config.noise_channels,
                   latent_variables=gen_config.latent_variables,
                   num_images=5,
                   latitude_range=lat_range,
                   longitude_range=lon_range,
                   ensemble_size=2,
                   noise_factor=1e-3,
                   denormalise_data=True,
                   normalize_ranks=True,
                   show_progress=True)
        
        
    def test_eval_multiple_checkpoints(self):
        
        records_folder = '/user/work/uz22147/tfrecords/d34d309eb0e00b04/'
        config = read_config.read_config(os.path.join(records_folder, 'local_config.yaml'))
        latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(config)
        
        _, _, _, data_config, gen_config, dis_config, train_config, _ = read_config.get_config_objects(test_config)


        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'eval.txt')
            evaluate_multiple_checkpoints(mode='GAN',
                                        arch='forceconv',
                                        fcst_data_source=data_config.fcst_data_source,
                                        obs_data_source=data_config.obs_data_source,
                                        latitude_range=latitude_range,
                                        longitude_range=longitude_range,
                                    validation_range=['201902', '201902'],
                                    log_folder=temp_dir,
                                    weights_dir='/user/work/uz22147/logs/cgan/d34d309eb0e00b04/models',
                                    records_folder=records_folder,
                                    downsample=False,
                                    noise_factor=1e-3,
                                    model_numbers=[38400, 6400],
                                    ranks_to_save=[38400, 6400],
                                    num_images=2,
                                    filters_gen=gen_config.filters_gen,
                                    filters_disc=dis_config.filters_disc,
                                    input_channels=data_config.input_channels,
                                    latent_variables=gen_config.latent_variables,
                                    noise_channels=gen_config.noise_channels,
                                    padding='reflect',
                                    ensemble_size=train_config.ensemble_size,
                                    constant_fields=data_config.constant_fields,
                                    data_paths=DATA_PATHS)
            
        