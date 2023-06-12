import os
import unittest
import copy
import numpy as np

from datetime import datetime
from numpy import testing
from unittest.mock import patch

from pathlib import Path
HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'

from dsrnngan.data.data_generator import DataGenerator, PermutedDataGenerator
from dsrnngan.data import data
from system_tests.test_data import create_dummy_stats_data

ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = str(data_folder / 'ERA5')
imerg_folder = str(data_folder / 'IMERG/half_hourly/final')

longitude_vals = [33, 34]
latitude_vals = [0, 1]

data_paths = copy.deepcopy(data.DATA_PATHS)
data_paths['GENERAL']['CONSTANTS'] = constants_path
        
class TestDataGenerator(unittest.TestCase):
    
    def setUp(self) -> None:
        
        create_dummy_stats_data()
        return super().setUp()
    
    
    def test_basic_generator(self):
        
        date_range = [datetime(2017,7,4), datetime(2017,7,5)]
        batch_size = 2
        
        data_gen = DataGenerator([datetime(2017,7,4), datetime(2017,7,5)], batch_size=batch_size, 
                                 forecast_data_source='ifs', observational_data_source='imerg', data_paths=data_paths,
                                    shuffle=False, constants=True, hour=17, longitude_range=longitude_vals,
                                    latitude_range=latitude_vals, normalise=True,
                                    downsample=False, seed=None)
        
        
        data = [data_gen[n] for n in range(len(date_range))]
        self.assertEqual(set(data[0][0].keys()), {'lo_res_inputs', 'hi_res_inputs', 'dates', 'hours'})
        self.assertEqual(set(data[0][1].keys()), {'output'})
        
        self.assertEqual(data[0][0]['lo_res_inputs'].shape, (batch_size, 2, 2, 20))
        self.assertEqual(data[0][0]['hi_res_inputs'].shape, (batch_size, 2, 2, 2))
        self.assertEqual(data[0][0]['dates'].shape, (batch_size,))
        self.assertEqual(data[0][0]['dates'].shape, (batch_size,))

        self.assertEqual(data[0][1]['output'].shape, (batch_size, 2, 2))
    
    def test_repeat_data(self):
        # Test that generator keeps producing if repeat_data is True
        date_range = [datetime(2017,7,4), datetime(2017,7,5)]
        batch_size = 1
        
        data_gen = DataGenerator(date_range, batch_size=batch_size, 
                                 forecast_data_source='ifs', observational_data_source='imerg', data_paths=data_paths,
                                 shuffle=False, constants=True, hour=17, longitude_range=longitude_vals,
                                 latitude_range=latitude_vals, normalise=True,
                                    downsample=False, seed=None, repeat_data=False)
        
        self.assertEqual(data_gen[2][0]['lo_res_inputs'].size, 0)
        
        repeat_data_gen = DataGenerator(date_range, batch_size=batch_size, 
                                 forecast_data_source='ifs', observational_data_source='imerg', data_paths=data_paths,
                                 shuffle=False, constants=True, hour=17, longitude_range=longitude_vals,
                                 latitude_range=latitude_vals, normalise=True,
                                    downsample=False, seed=None, repeat_data=True)
        
        self.assertGreater(repeat_data_gen[2][0]['lo_res_inputs'].size, 0)
        np.testing.assert_allclose(repeat_data_gen[2][0]['lo_res_inputs'], repeat_data_gen[0][0]['lo_res_inputs'])
    
    def test_input_all_hours(self):
        # Test that generator keeps producing if repeat_data is True
        date_range = [datetime(2017,7,4), datetime(2017,7,5)]
        hours = [2,3]
        batch_size = 1
        
        data_gen = DataGenerator(date_range, batch_size=batch_size, 
                                 forecast_data_source='ifs', observational_data_source='imerg', data_paths=data_paths,
                                 shuffle=False, constants=True, hour=hours, longitude_range=longitude_vals,
                                 latitude_range=latitude_vals, normalise=True,
                                    downsample=False, seed=None, repeat_data=False)
        
        data = [data_gen[n] for n in range(len(date_range)+1)]
        self.assertEqual(data[2][0]['lo_res_inputs'].size, 0)
        self.assertListEqual(list(data_gen.hours), hours)
        
    def test_permuted_generator(self):
        
        # create data first
        date_range = [datetime(2017,7,4), datetime(2017,7,5)]
        data_gen = DataGenerator([datetime(2017,7,4), datetime(2017,7,5)], 1, forecast_data_source='ifs', observational_data_source='imerg', data_paths=data_paths,
                                    shuffle=False, constants=True, hour=17, longitude_range=longitude_vals,
                                    latitude_range=latitude_vals, normalise=True,
                                    downsample=False, seed=5)
        
        data = [data_gen[n] for n in range(len(date_range))]
        
        # cast data into arrays
        hi_res_inputs = np.stack([item[0]['hi_res_inputs'][0,:,:,:] for item in data], axis=0)
        lo_res_inputs = np.stack([item[0]['lo_res_inputs'][0,:,:,:] for item in data], axis=0)
        dates = np.stack([item[0]['dates'] for item in data], axis=0)
        hours = np.stack([item[0]['hours'] for item in data], axis=0)

        outputs = np.stack([item[1]['output'][0,:,:] for item in data], axis=0)
    
        for permutation_type in ['lo_res_inputs', 'hi_res_inputs']:
            for permuted_index in [0,1]:
                
                input_permutation_config = {'type': permutation_type, 'permute_index': permuted_index}
                permuted_data_gen = PermutedDataGenerator(lo_res_inputs=lo_res_inputs, hi_res_inputs=hi_res_inputs,
                                                        outputs=outputs, dates=dates, hours=hours, 
                                                        input_permutation_config=input_permutation_config, seed=5)
                
                permuted_data = [permuted_data_gen[n] for n in range(len(date_range))]
                
                self.assertEqual(set(data[0][0].keys()), {'lo_res_inputs', 'hi_res_inputs', 'dates', 'hours'})
                self.assertEqual(set(data[0][1].keys()), {'output'})
                
                self.assertEqual(data[0][0]['lo_res_inputs'].shape, (1, 2, 2, 20))
                self.assertEqual(data[0][0]['hi_res_inputs'].shape, (1, 2, 2, 2))
                self.assertEqual(data[0][0]['dates'].shape, (1,))
                self.assertEqual(data[0][0]['dates'].shape, (1,))

                self.assertEqual(data[0][1]['output'].shape, (1, 2, 2))
                
                # Check data permuted correctly
                if permutation_type == 'lo_res_inputs':
                    np.testing.assert_allclose(permuted_data[0][0]['lo_res_inputs'][:,:,:, permuted_index], data[1][0]['lo_res_inputs'][:,:,:,permuted_index])
                    np.testing.assert_allclose(permuted_data[1][0]['lo_res_inputs'][:,:,:,permuted_index], data[0][0]['lo_res_inputs'][:,:,:,permuted_index])
                elif permutation_type == 'hi_res_inputs':
                    np.testing.assert_allclose(permuted_data[0][0]['hi_res_inputs'][:,:,:, permuted_index], data[1][0]['hi_res_inputs'][:,:,:,permuted_index])
                    np.testing.assert_allclose(permuted_data[1][0]['hi_res_inputs'][:,:,:,permuted_index], data[0][0]['hi_res_inputs'][:,:,:,permuted_index])