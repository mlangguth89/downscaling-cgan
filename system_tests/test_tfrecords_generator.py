import sys, os
import unittest
import copy
import tempfile
from types import SimpleNamespace
from pathlib import Path
from glob import glob
import numpy as np
from datetime import datetime, date
import tensorflow as tf

tf.data.experimental.enable_debug_mode()
tf.compat.v1.enable_eager_execution()

HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))

from dsrnngan.data.tfrecords_generator import write_data, create_dataset
from dsrnngan.data.data import all_ifs_fields, all_era5_fields, IMERG_PATH, ERA5_PATH, DATA_PATHS
from dsrnngan.utils.read_config import read_data_config, get_data_paths, get_lat_lon_range_from_config
from system_tests.test_data import create_dummy_stats_data

data_folder = HOME / 'system_tests' / 'data'

ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = ERA5_PATH
imerg_folder = IMERG_PATH

test_data_dir = HOME / 'system_tests' / 'data'

class TestTfrecordsGenerator(unittest.TestCase):
    
    def setUp(self) -> None:
        

        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name
        self.data_config_dict = {'data_paths': 'BLUE_PEBBLE', 
                            'fcst_data_source': 'ifs', 
                            'obs_data_source': 'imerg', 
                            'normalisation_year': 2017, 
                            'input_image_width': 10, 
                            'num_samples': 3, 
                            'num_samples_per_image': 1, 
                            'normalise': True,
                            'num_classes': 1, 
                            'min_latitude': -1.95, 
                            'max_latitude': 1.95, 
                            'latitude_step_size': 0.1, 
                            'min_longitude': 32.05, 
                            'max_longitude': 34.95, 
                            'longitude_step_size': 0.1, 
                            'input_fields': ['2t', 'cape', 'cp', 'r200', 'r700', 'r950'], 
                            'constant_fields': ['lakes', 'sea', 'orography'], 
                            'paths': {'BLUE_PEBBLE':
                                {'GENERAL': {'IMERG': str(test_data_dir/ 'IMERG/half_hourly/final'),
                                                'IFS': str(test_data_dir/  'IFS'),
                                                'ERA5': str(test_data_dir/ 'ERA5'),
                                                'OROGRAPHY': str(test_data_dir/ 'constants/h_HRES_EAfrica.nc'),
                                                'LSM': str(test_data_dir/ 'constants/lsm_HRES_EAfrica.nc'),
                                                'LAKES': str(test_data_dir/ 'constants/lsm_HRES_EAfrica.nc'),
                                                'SEA':  str(test_data_dir/ 'constants/lsm_HRES_EAfrica.nc'),
                                                'CONSTANTS': os.path.join(self.temp_dir.name, 'constants')},
                                        'TFRecords': {'tfrecords_path': self.temp_dir.name}}}}

        self.data_config = read_data_config(data_config_dict=self.data_config_dict)
        
        self.lat_range, self.lon_range = get_lat_lon_range_from_config(data_config=self.data_config)
        
        self.constants_path = os.path.join(self.temp_dir_name, 'constants')
        os.mkdir(self.constants_path)
        
        create_dummy_stats_data(constants_path=self.constants_path, lon_range=[32,34], lat_range=[-1, 1])

        return super().setUp()
    
    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        return super().tearDown()
            
    def test_write_ifs_data(self):
                    
        output_dir = write_data(['201707', '201712'],
                                data_label='train',
                hours=[18],
                data_config=self.data_config,
                debug=True)
        files_0 = glob(os.path.join(output_dir, '*train*'))
        self.assertGreater(len(files_0), 0)
            
            
    def test_write_era5_data(self):
        
        #TODO: this test needs some subsampled ERA5 and iMERG data to not take ages
        with tempfile.TemporaryDirectory() as tmpdirname:

            test_data_dir = HOME / 'system_tests' / 'data'
            data_paths = {'GENERAL': {
                                      'IMERG': str(test_data_dir / 'IMERG/half_hourly/final'),
                                      'ERA5': str(test_data_dir / 'ERA5'),
                                    'OROGRAPHY': str(test_data_dir / 'constants/h_HRES_EAfrica.nc'),
                                      'LSM': str(test_data_dir / 'constants/lsm_HRES_EAfrica.nc'),
                                      'CONSTANTS': str(test_data_dir / 'constants')},
                        'TFRecords': {'tfrecords_path': tmpdirname}}
            
            write_data(start_date=datetime(2018, 1, 1),
                       end_date=datetime(2019,1,31),
                        forecast_data_source='era5', 
                        observational_data_source='imerg',
                        hours=[18],
                        img_chunk_width=4,
                        img_size=4,
                        num_class=4,
                        log_precip=True,
                        fcst_norm=True,
                        data_paths=data_paths,
                        scaling_factor=1,
                        latitude_range=np.arange(0, 1, 0.1),
                        longitude_range=np.arange(33, 34, 0.1))
                
            # Check that files exist in output dir
            folder = Path(tmpdirname)
            hash_folder = str([f for f in folder.iterdir() if f.is_dir()][0])
            files_2018 = glob(os.path.join(hash_folder, '*2018*'))
            self.assertGreater(len(files_2018), 0)
            
            files_2019 = glob(os.path.join(hash_folder, '*2019*'))
            self.assertEqual(len(files_2019), 0)
            
            files_yaml = glob(os.path.join(hash_folder, '*.yaml'))
            self.assertEqual(len(files_yaml), 3)
            
            # Check that it behaves correctly when radar data doesn't exist

        
            write_data(start_date=date(2017,1,1),
                       end_date=date(2017,3,1),
                forecast_data_source='era5', 
                observational_data_source='imerg',
                hours=[12],
                img_chunk_width=4,
                img_size=4,
                num_class=4,
                log_precip=True,
                fcst_norm=True,
                data_paths=data_paths,
                constants=False,
                scaling_factor=1,
                latitude_range= np.arange(0, 2, 0.1),
                longitude_range=np.arange(33, 35, 0.1))

            # Check that no new files written when no radar data
            folder = Path(tmpdirname)
            hash_folder = str([f for f in folder.iterdir() if f.is_dir()][0])
            files_1 = glob(os.path.join(hash_folder, '*'))
            self.assertEqual(set(files_1), set(files_2018 + files_2019 + files_yaml))
        
    def test_create_dataset_ifs_nimrod(self):
        '''
        Note that the tfrecords have been created using the test_write_ifs_data function, so 
        the forecast and output shapes will be the same as for them
        '''
           
        # TODO: run it twice to check random seed is working
        ds  = create_dataset(2018,
                1,
                fcst_shape=(4, 4, len(all_ifs_fields)),
                con_shape=None,
                out_shape=(4, 4, 1),
                folder=str(HOME / 'system_tests/data/tfrecords/IFS_nimrod'),
                shuffle_size=1024,
                repeat=True, 
                seed=1)
        element_spec = ds.element_spec
        self.assertEqual(set(element_spec[0].keys()), {'lo_res_inputs'})
        self.assertEqual(set(element_spec[1].keys()), {'output'})
        
        # unpacking values makes the parsing actually process so we can check there are no errors
        sample_vals = list(ds.take(1))[0]
        lo_res_vals = sample_vals[0]['lo_res_inputs'].numpy()
        output_vals = sample_vals[1]['output'].numpy()
        
        self.assertFalse((lo_res_vals == np.nan).any())
        self.assertFalse((output_vals == np.nan).any())
        self.assertEqual(lo_res_vals.shape, (4, 4, len(all_ifs_fields)))
        self.assertEqual(output_vals.shape, (4, 4, 1))

        
    def test_create_dataset_ifs_imerg(self):
        '''
        Note that the tfrecords have been created using the test_write_era5_data function, so 
        the forecast and output shapes will be the same as for them
        '''
        data_config = copy.copy(self.data_config)
        data_config.paths['BLUE_PEBBLE'] = DATA_PATHS
        data_config.paths['BLUE_PEBBLE']['TFRecords']['tfrecords_path'] = self.temp_dir_name
        data_config.paths['BLUE_PEBBLE']['GENERAL']['CONSTANTS'] = self.constants_path
        hash_dir = write_data(['201707'],
                                data_label='train',
                hours=[18],
                data_config=self.data_config,
                debug=True)

        # Without cropping
        height = data_config.input_image_width
        width = data_config.input_image_width
        ds  = create_dataset(data_label='train',
                   clss=0,
                   fcst_shape=(height, width, len(data_config.input_fields)),
                   con_shape=(height, width, len(data_config.constant_fields)),
                   out_shape=(height, width, 1),
                   folder=hash_dir,
                   shuffle_size=1024,
                   crop_size=None,
                   repeat=True,
                   seed=1)
        
        element_spec = ds.element_spec

        self.assertEqual(set(element_spec[0].keys()), {'lo_res_inputs', 'hi_res_inputs'})
        self.assertEqual(set(element_spec[1].keys()), {'output'})
        

        sample_vals = list(ds.take(1))[0]
        lo_res_vals = sample_vals[0]['lo_res_inputs'].numpy()
        hi_res_vals = sample_vals[0]['hi_res_inputs'].numpy()
        output_vals = sample_vals[1]['output'].numpy()
        
        # unpacking values makes the parsing actually process so we can check there are no errors
        self.assertFalse((lo_res_vals == np.nan).any())
        self.assertFalse((hi_res_vals == np.nan).any())
        self.assertFalse((output_vals == np.nan).any())
        self.assertEqual(lo_res_vals.shape, (height, width, len(data_config.input_fields)))
        self.assertEqual(hi_res_vals.shape, (height, width, len(data_config.constant_fields)))
        self.assertEqual(output_vals.shape, (height, width, 1))
        
        #############################
        # with cropping
        crop_size = 4
        ds  = create_dataset(data_label='train',
                   clss=0,
                   fcst_shape=(height, width, len(data_config.input_fields)),
                   con_shape=(height, width, len(data_config.constant_fields)),
                   out_shape=(height, width, 1),
                   folder=hash_dir,
                   shuffle_size=1024,
                   crop_size=crop_size,
                   repeat=True,
                   seed=(1,1)
                   )
        
        element_spec = ds.element_spec

        cropped_sample_vals = list(ds.take(1))[0]
        lo_res_vals_cropped = cropped_sample_vals[0]['lo_res_inputs'].numpy()
        hi_res_vals_cropped = cropped_sample_vals[0]['hi_res_inputs'].numpy()
        output_vals_cropped = cropped_sample_vals[1]['output'].numpy()
        self.assertEqual(lo_res_vals_cropped.shape, (crop_size, crop_size, len(data_config.input_fields)))
        self.assertEqual(hi_res_vals_cropped.shape, (crop_size, crop_size, len(data_config.constant_fields)))
        self.assertEqual(output_vals_cropped.shape, (crop_size, crop_size, 1))
        
        (x_ixs, y_ixs) = np.where(lo_res_vals[:,:,0] == lo_res_vals_cropped[0,0,0])
        
        lores_match_found = False
        hires_match_found = False
        output_match_found = False
        for n, x_ix in enumerate(x_ixs):

            if np.array_equal(lo_res_vals_cropped[:,:,0], lo_res_vals[x_ix:x_ix +crop_size,y_ixs[n]:y_ixs[n] +crop_size,0]):
                lores_match_found = True
            
            if np.array_equal(hi_res_vals_cropped[:,:,0], hi_res_vals[x_ix:x_ix +crop_size,y_ixs[n]:y_ixs[n] +crop_size,0]):
                hires_match_found = True
                
            if np.array_equal(output_vals_cropped[:,:,0], output_vals[x_ix:x_ix +crop_size,y_ixs[n]:y_ixs[n] +crop_size,0]):
                output_match_found = True
        
        self.assertTrue(lores_match_found)
        self.assertTrue(hires_match_found)
        self.assertTrue(output_match_found)
        
        
        #############################
        # with rotation
        seed=(2,1)
        ds  = create_dataset(data_label='train',
                   clss=0,
                   fcst_shape=(height, width, len(data_config.input_fields)),
                   con_shape=(height, width, len(data_config.constant_fields)),
                   out_shape=(height, width, 1),
                   folder=hash_dir,
                   shuffle_size=1024,
                   crop_size=None,
                   repeat=True,
                   seed=seed,
                   rotate=True
                   )
        
        element_spec = ds.element_spec
        
        rng = np.random.default_rng(seed=seed[0])
        angle = rng.choice([0,180],1)[0]

        cropped_sample_vals = list(ds.take(1))[0]
        lo_res_vals_cropped = cropped_sample_vals[0]['lo_res_inputs'].numpy()
        hi_res_vals_cropped = cropped_sample_vals[0]['hi_res_inputs'].numpy()
        output_vals_cropped = cropped_sample_vals[1]['output'].numpy()
        
        t=1

        
if __name__ == '__main__':
    unittest.main()