import os
import numpy as np
import xarray as xr
import os
import sys
import tempfile
import pickle
import copy

from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import unittest
from glob import glob

HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))

from dsrnngan.data import setupdata
from dsrnngan.model.setupmodel import load_model_from_folder
from dsrnngan.utils import read_config
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.evaluation.evaluation import setup_inputs, eval_one_chkpt, evaluate_multiple_checkpoints, create_single_sample
from dsrnngan.data.data import DATA_PATHS, all_ifs_fields, get_ifs_filepath, denormalise
from system_tests.test_main import create_example_model, model_config, data_config, test_data_paths

model_config.train.batch_size = 1  # setup_params["TRAIN"]["batch_size"]
output_image_width = data_config.input_image_width * model_config.downscaling_factor
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
        
        
        if not os.path.isdir(self.tmp_data_folder):
            # Create a dummy model if one doesn't already exist
            self.records_folder, self.model_folder = create_example_model(model_config=model_config, data_paths=test_data_paths)
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
        gen, batch_gen_valid = setup_inputs(model_config=model_config,
                                            data_config=data_config,
                                        records_folder=self.records_folder,
                                        hour=18,
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
            truth = denormalise(truth)

        noise_shape = np.array(cond)[0, ..., 0].shape + (model_config.generator.noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=model_config.train.batch_size)

        for ii in range(rank_samples):
            img_gen = gen.predict([cond, const, noise_gen()])

    
    def test_create_single_sample(self):
        
        gen = dsrnngan.data.load_model_from_folder.load_model_from_folder(model_folder=self.model_folder, model_number=2)
        _, data_gen_valid = setupdata.load_data_from_config(self.config, data_paths=data_paths, hour=17)
        
        obs, samples_gen, fcst, imerg_persisted_fcst, cond, const, date, hour = create_single_sample(mode=model_config.mode,
                    data_idx=0,
                    batch_size=1,
                    gen=gen,
                    fcst_data_source=data_config.fcst_data_source,
                    data_gen=data_gen_valid,
                    noise_channels=model_config.generator.noise_channels,
                    latent_variables=model_config.generator.latent_variables,
                    latitude_range=lat_range,
                    longitude_range=lon_range,
                    ensemble_size=2,
                    denormalise_data=True,
                    seed=1)
        
        
        input_shuffle_config = {'type': 'lo_res_inputs', 'shuffle_index': 1}
        obs_permuted, samples_gen_permuted, fcst_permuted, imerg_persisted_fcst_permuted, cond_permuted, const_permuted, date_permuted, hour_permuted = create_single_sample(mode=model_config.mode,
                    data_idx=0,
                    batch_size=1,
                    gen=gen,
                    fcst_data_source=data_config.fcst_data_source,
                    data_gen=data_gen_valid,
                    noise_channels=model_config.generator.noise_channels,
                    latent_variables=model_config.generator.latent_variables,
                    latitude_range=lat_range,
                    longitude_range=lon_range,
                    ensemble_size=2,
                    denormalise_data=True,
                    input_shuffle_config=input_shuffle_config,
                    seed=1)
        
        self.assertEqual(date, date_permuted)
        self.assertEqual(hour, hour_permuted)
        

    def test_eval_one_chkpt(self):
        
        gen = dsrnngan.data.load_model_from_folder.load_model_from_folder(model_folder=self.model_folder, model_number=2)
        _, data_gen_valid = setupdata.load_data_from_config(self.config, data_paths=data_paths, hour=17)
                
        eval_one_chkpt(
                   model_config=model_config,
                   data_config=data_config,    
                   gen=gen,
                   data_gen=data_gen_valid,
                   num_images=5,
                   latitude_range=lat_range,
                   longitude_range=lon_range,
                   ensemble_size=2,
                   noise_factor=1e-3,
                   denormalise_data=True,
                   normalize_ranks=True,
                   show_progress=True)
        
        
    def test_eval_multiple_checkpoints(self):
        
        weights_dir = '/user/work/uz22147/logs/cgan/7ed5693482c955aa_small-cl1000'
        data_config = read_config.read_data_config(config_folder=weights_dir)
        model_config = read_config.read_model_config(config_folder=weights_dir)

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluate_multiple_checkpoints(model_config=model_config,
                                        data_config=data_config,
                                        log_folder=temp_dir,
                                        weights_dir=os.path.join(weights_dir, 'models'),
                                        records_folder=None,
                                        noise_factor=1e-3,
                                        model_numbers=[38400, 6400],
                                        num_images=2,
                                        ensemble_size=2,
                                        shuffle=True)
            

        
