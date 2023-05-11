import sys, os
import unittest
import numpy as np
import tempfile
import yaml
import pickle
from pathlib import Path
from glob import glob
import xarray as xr
from datetime import datetime

HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))

from dsrnngan.main import main, parser
from dsrnngan.data.tfrecords_generator import write_data
from dsrnngan.data.data import DATA_PATHS, all_ifs_fields, get_ifs_filepath

data_folder = HOME / 'system_tests' / 'data'
ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = str(data_folder / 'ERA5')
imerg_folder = str(data_folder / 'IMERG' / 'half_hourly' / 'final')
tfrecords_folder = str(data_folder / 'tmp_model_test')

# Create test config
lat_range = np.arange(0, 1.1, 0.1) # deliberately asymettrical to test for non-square images
lon_range = np.arange(33, 34, 0.1)
config_path = str(HOME / 'config' /'local_config.yaml')
with open(config_path, 'r') as f:
    test_config = yaml.safe_load(f)

test_config['DOWNSCALING'] = {'downscaling_factor': 1, 'steps': [1]}
test_config['TRAIN']['steps_per_checkpoint'] = 1
test_config['TRAIN']['img_chunk_width'] = 5

test_config['DATA'] = {
    'fcst_data_source': 'ifs',
    'obs_data_source': 'imerg',
    'input_channels': 20,
    'constant_fields': 2,
    'input_image_width': 10, # Assumes a square image
    'num_samples': 10,
    'num_samples_per_image': 1,
    'normalise': True,
    'min_latitude': float(np.round(min(lat_range), 1)),
    'max_latitude': float(np.round(max(lat_range), 1)),
    'latitude_step_size': 0.1,
    'min_longitude': float(np.round(min(lon_range), 1)),
    'max_longitude': float(np.round(max(lon_range), 1)),
    'longitude_step_size': 0.1
}

test_data_paths = {'GENERAL': {'IFS': ifs_path, 'IMERG': imerg_folder, 'LSM': os.path.join(constants_path, 'lsm_HRES_EAfrica.nc'), 
                                'OROGRAPHY': os.path.join(constants_path, 'h_HRES_EAfrica.nc'), 'CONSTANTS': constants_path}, 
                    'TFRecords': {'tfrecords_path': tfrecords_folder}}

def create_example_model(config: dict=test_config,
                         data_paths: dict=test_data_paths, 
                         lat_range: list=lat_range, 
                         lon_range: list=lon_range):
    """
    Function for running example data through tfrecords and train a model. Used in other tests (e.g. evaluation) as well as 
    the test for main

    Args:
        data_paths (dict): Dict containing paths to data sources
        records_dir (str): Directory with tensorflow records in it. Will also be the root folder for cGAN model output if cgan_model_folder is None
        lat_range (list): Range of latitude values
        lon_range (list): Range of longitude values
        

    Returns:
    """
            
    hash_dir = write_data(year_month_range=['201707', '201708'],
        data_label='train',
        forecast_data_source='ifs', 
        observational_data_source='imerg',
        hours=[18],
        num_class=4,
        normalise=True,
        data_paths=data_paths,
        constants=True,
        latitude_range=lat_range,
        longitude_range=lon_range,
        debug=True,
        config=config)

    records_dir = data_paths['TFRecords']['tfrecords_path']
    log_folder = os.path.join(records_dir, 'cgan_output')
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    records_folder = hash_dir

    args = parser.parse_args([
                            '--restart',
                            '--num-samples',
                            '10',
                            '--records-folder',
                            records_folder, 
                            '--output-suffix',
                            'asdf'])

    log_folder = main(records_folder=args.records_folder, restart=args.restart, do_training=args.do_training, 
        evalnum=args.evalnum,
        noise_factor=args.noise_factor,
        num_samples_override=args.num_samples,
        num_images=args.num_images,
        eval_model_numbers=args.eval_model_numbers,
        val_start=args.val_ym_start,
        val_end=args.val_ym_end,
        ensemble_size=args.ensemble_size,
        shuffle_eval=not args.no_shuffle_eval,
        save_generated_samples=args.save_generated_samples,
        training_weights=args.training_weights, debug=True,
        output_suffix=args.output_suffix,
        log_folder=log_folder)
    
    return records_folder, log_folder

class TestMain(unittest.TestCase):
    
    for field in all_ifs_fields:
        fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, str(data_folder / 'IFS'))
        ds = xr.load_dataset(fp)
        
        # Mock IFS stats dicts (to avoid having to create them in data collection stage)
        var_name = list(ds.data_vars)[0]
        stats = {'min': np.abs(ds[var_name]).min().values,
            'max': np.abs(ds[var_name]).max().values,
            'mean': ds[var_name].mean().values,
            'std': ds[var_name].std().values}
        
        # Saving it as 2017 since that's the defualt 
        output_fp = f'{constants_path}/IFS_norm_{field}_2017_lat0-1lon33-33.pkl'
        with open(output_fp, 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
        
    def test_main_ifs_imerg(self):
        """
        This is an end-to-end test to check the training and tfrecord ; still to implement 
        something that checks the evaluation (requires checkpoints created successfully)
        """
        with tempfile.TemporaryDirectory() as tempdir:
            test_data_paths['TFRecords']['tfrecords_path'] = tempdir
            create_example_model(config=test_config, data_paths=test_data_paths, lat_range=lat_range, lon_range=lon_range)

if __name__ == '__main__':
    unittest.main()