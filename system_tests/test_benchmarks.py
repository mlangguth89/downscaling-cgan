import sys, os
import unittest
import tempfile
import pickle
import time
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from numpy import testing

HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'

from dsrnngan.evaluation.benchmarks import QuantileMapper

sys.path.append(str(HOME))

class TestBenchmarks(unittest.TestCase):
    
    def test_quantile_mapping(self):
        
        # load test data
        with open(str(data_folder / 'quantile_mapping' / 'quantile_mapping_test_data.pkl'), 'rb') as ifh:
            train_data = pickle.load(ifh)
        
        ifs_train_data = train_data['ifs_train_data']
        imerg_train_data = train_data['imerg_train_data']
        fcst_array = train_data['fcst_array'][:,:,:,0]
        training_dates = train_data['training_dates']
        training_hours = train_data['training_hours']
        latitude_range = train_data['lat_range']
        longitude_range= train_data['lon_range']
        
        (_, test_width, test_height) = fcst_array.shape

        month_ranges = [[1],[2]]
        quantile_locs = [0, 0.1, 0.2, 0.3, 0.8, 0.9]

        qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, 
                                 longitude_range=longitude_range, quantile_locs=quantile_locs, num_lat_lon_chunks=2)
        # identify best threshold and train on all the data
        qmapper.train(fcst_data=ifs_train_data, obs_data=imerg_train_data, training_dates=training_dates, training_hours=training_hours)
        
        # Check that 1.0 quantile is inserted
        self.assertIn(1.0, [item[0] for item in qmapper.quantiles_by_area[list(qmapper.quantiles_by_area.keys())[0]]['fcst_quantiles']])
        
        # Add data to test set that is larger than max of train set
        fcst_array = np.append(fcst_array, (ifs_train_data.max() + 10)*np.ones((1, test_width, test_height)), axis=0)
                
        # Just select some random dates for the test set
        dates = np.random.choice(training_dates, fcst_array.shape[0])
        hours = np.random.choice(training_hours, fcst_array.shape[0])

        fcst_corrected = qmapper.get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, hours=hours)
        
        # Check no nulls
        self.assertEqual(np.isnan(fcst_corrected).sum(), 0)
        
        if ifs_train_data.max() > imerg_train_data.max():
            self.assertLess(fcst_corrected.max(), fcst_array.max())
        else:
            self.assertGreater(fcst_corrected.max(), fcst_array.max())
                
            
