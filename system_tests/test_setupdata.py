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

from dsrnngan.data import setupdata
        
HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))


from dsrnngan.utils import read_config
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.data.data import DATA_PATHS, all_ifs_fields, get_ifs_filepath
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


class TestSetupData(unittest.TestCase):
        
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
    
    def test_setup_data(self):
        
        # Check the basic version runs
        data_gen_train, data_gen_valid = setupdata.setup_data(
            records_folder=None, # Use case of permuted features is currently using full image dataset
            fcst_data_source=data_config.fcst_data_source,
            obs_data_source=data_config.obs_data_source,
            latitude_range=lat_range,
            longitude_range=lon_range,
            load_full_image=True,
            validation_range=val_config.val_range,
            training_range=train_config.training_range,
            batch_size=1,
            downsample=False,
            data_paths=data_paths,
            hour=17)
        
        normal_inputs = []
        normal_outputs = []
        
        for n in range(5):
            inputs, outputs = data_gen_valid[0]
            normal_inputs.append(inputs)
            normal_outputs.append(outputs)
        
