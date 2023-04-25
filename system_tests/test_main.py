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
from dsrnngan import utils
from dsrnngan.tfrecords_generator import write_data
from dsrnngan.data import DATA_PATHS, all_ifs_fields, get_ifs_filepath

data_folder = HOME / 'system_tests' / 'data'
ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = str(data_folder / 'ERA5')
imerg_folder = str(data_folder / 'IMERG' / 'half_hourly' / 'final')

class TestMain(unittest.TestCase):
    
    for field in all_ifs_fields:
        fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, str(data_folder / 'IFS'))
        ds = xr.load_dataset(fp)
        
        # Mock IFS stats dicts
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
        This test is just designed to check the training will run; still to implement 
        something that checks the evaluation (requires checkpoints created successfully)
        """
        
        lat_range = np.arange(0, 1.1, 0.1) # deliberately asymettrical to test for non-square images
        lon_range = np.arange(33, 34, 0.1)
        
        with tempfile.TemporaryDirectory() as tempdir:
            
            data_paths = {'GENERAL': {'IFS': ifs_path, 'IMERG': imerg_folder, 'LSM': os.path.join(constants_path, 'lsm_HRES_EAfrica.nc'), 
                                      'OROGRAPHY': os.path.join(constants_path, 'h_HRES_EAfrica.nc'), 'CONSTANTS': constants_path}, 
                          'TFRecords': {'tfrecords_path': tempdir}}
            config_path = str(HOME / 'dsrnngan' / 'local_config.yaml')
            with open(config_path, 'r') as f:
                setup_params = yaml.safe_load(f)
            
            setup_params['DOWNSCALING'] = {'downscaling_factor': 1, 'steps': [1]}
            setup_params['TRAIN']['steps_per_checkpoint'] = 1

            setup_params['DATA'] = {
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
            
            # Save config to tfrecords folder
            
            
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
                config=setup_params)

                
            log_folder = str(HOME / 'system_tests' / 'data' / 'tmp')
            if not os.path.isdir(log_folder):
                os.mkdir(log_folder)
                
            # setup_params['SETUP']['log_folder'] = log_folder

            records_folder = hash_dir

            args = parser.parse_args([
                                  '--restart',
                                  '--num-samples',
                                  '10',
                                  '--records-folder',
                                  records_folder])

            main(records_folder=args.records_folder, restart=args.restart, do_training=args.do_training, 
                evalnum=args.evalnum,
                evaluate=args.evaluate,
                plot_ranks=args.plot_ranks,
                noise_factor=args.noise_factor,
                num_samples_override=args.num_samples,
                num_images=args.num_images,
                model_numbers=args.model_numbers,
                val_start=args.val_ym_start,
                val_end=args.val_ym_end,
                ensemble_size=args.ensemble_size,
                shuffle_eval=not args.no_shuffle_eval,
                save_generated_samples=args.save_generated_samples,
                training_weights=args.training_weights, debug=True)
                


if __name__ == '__main__':
    unittest.main()