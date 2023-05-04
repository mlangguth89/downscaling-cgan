import os
import unittest
import copy
import numpy as np

from datetime import datetime
from numpy import testing

from pathlib import Path
HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'

from dsrnngan.data_generator import DataGenerator, PermutedDataGenerator
from dsrnngan.data import DATA_PATHS
from system_tests.test_data import create_dummy_stats_data

ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = str(data_folder / 'ERA5')
imerg_folder = str(data_folder / 'IMERG/half_hourly/final')

longitude_vals = [33, 34]
latitude_vals = [0, 1]

data_paths = copy.deepcopy(DATA_PATHS)
data_paths['GENERAL']['CONSTANTS'] = constants_path
        
class TestDataGenerator(unittest.TestCase):
    
    def setUp(self) -> None:
        
        create_dummy_stats_data()
        return super().setUp()
    
    
    def test_basic_generator(self):
        
        date_range = [datetime(2017,7,4), datetime(2017,7,5)]
        data_gen = DataGenerator([datetime(2017,7,4), datetime(2017,7,5)], 1, forecast_data_source='ifs', observational_data_source='imerg', data_paths=data_paths,
                                    shuffle=False, constants=True, hour=17, longitude_range=longitude_vals,
                                    latitude_range=latitude_vals, normalise=True,
                                    downsample=False, seed=None)
        
        
        data = [data_gen[n] for n in range(len(date_range))]
        
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
        
        for permuted_index in [0,1,2]:
            permuted_data_gen = PermutedDataGenerator(lo_res_inputs=lo_res_inputs, hi_res_inputs=hi_res_inputs,
                                                    outputs=outputs, dates=dates, hours=hours, permute_fcst_index=permuted_index, seed=5)
            
            permuted_data = [permuted_data_gen[n] for n in range(len(date_range))]
            
            # Check data permuted correctly
            np.testing.assert_allclose(permuted_data[0][0]['lo_res_inputs'][:,:,:,permuted_index], data[1][0]['lo_res_inputs'][:,:,:,permuted_index])
            np.testing.assert_allclose(permuted_data[1][0]['lo_res_inputs'][:,:,:,permuted_index], data[0][0]['lo_res_inputs'][:,:,:,permuted_index])