import sys
import unittest
import pickle
import numpy as np
from pathlib import Path


HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'

from dsrnngan.evaluation.benchmarks import QuantileMapper, empirical_quantile_map, quantile_map_grid

sys.path.append(str(HOME))

class TestBenchmarks(unittest.TestCase):
    
    def setUp(self) -> None:
        
        # load test data
        with open(str(data_folder / 'quantile_mapping' / 'quantile_mapping_test_data.pkl'), 'rb') as ifh:
            train_data = pickle.load(ifh)
        
        self.ifs_train_data = train_data['ifs_train_data']
        self.imerg_train_data = train_data['imerg_train_data']
        self.fcst_array = train_data['fcst_array'][:,:,:,0]
        self.training_dates = train_data['training_dates']
        self.training_hours = train_data['training_hours']
        self.latitude_range = train_data['lat_range']
        self.longitude_range= train_data['lon_range']
        return super().setUp()
    
    def test_empirical_quantile_mapping(self):
        
        array_to_correct = self.fcst_array.copy()
        (_, test_width, test_height) = array_to_correct.shape

        quantile_locs = [0, 0.1, 0.2, 0.3, 0.8, 0.9]
            
        # Add data to test set that is larger than max of train set
        array_to_correct = np.append(array_to_correct, (self.ifs_train_data.max() + 10)*np.ones((1, test_width, test_height)), axis=0)
        
        fcst_corrected = np.empty(array_to_correct.shape)
        fcst_corrected[:,:,:] = np.nan
        
        for w in range(test_width):
            for h in range(test_height):
                
                result = empirical_quantile_map(obs_train=self.imerg_train_data[:,w,h], 
                                                               model_train=self.ifs_train_data[:,w,h], s=array_to_correct[:,w,h],
                                                               quantiles=quantile_locs, extrapolate='constant')
                fcst_corrected[:,w,h] = result
                
                # Check no nulls
                self.assertEqual(np.isnan(result).sum(), 0)
        
        
        if self.ifs_train_data.max() > self.imerg_train_data.max():
            self.assertLess(fcst_corrected.max(), array_to_correct.max())
        else:
            self.assertGreater(fcst_corrected.max(), array_to_correct.max())
    
    def test_quantil_map_grid(self):
        
        qmapped_fcst = quantile_map_grid(self.fcst_array, fcst_train_data=self.ifs_train_data, 
                      obs_train_data=self.imerg_train_data, quantile_locations=np.linspace(0,1,10), neighbourhood_size=2)
        self.assertEqual(np.isnan(qmapped_fcst).sum(), 0)

        
        qmapped_fcst = quantile_map_grid(self.fcst_array, fcst_train_data=self.ifs_train_data, 
                      obs_train_data=self.imerg_train_data, quantile_locations=np.linspace(0,1,10), neighbourhood_size=0)
        self.assertEqual(np.isnan(qmapped_fcst).sum(), 0)

    def test_QuantileMapper(self):
                
        (_, test_width, test_height) = self.fcst_array.shape

        month_ranges = [[1],[2]]
        quantile_locs = [0, 0.1, 0.2, 0.3, 0.8, 0.9]

        qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=self.latitude_range, 
                                 longitude_range=self.longitude_range, quantile_locs=quantile_locs, num_lat_lon_chunks=2)
        # identify best threshold and train on all the data
        qmapper.train(fcst_data=self.ifs_train_data, obs_data=self.imerg_train_data, 
                      training_dates=self.training_dates, training_hours=self.training_hours)
        
        # Check that 1.0 quantile is inserted
        self.assertIn(1.0, [item[0] for item in qmapper.quantiles_by_area[list(qmapper.quantiles_by_area.keys())[0]]['fcst_quantiles']])
        
        # Add data to test set that is larger than max of train set
        fcst_array = np.append(fcst_array, (self.ifs_train_data.max() + 10)*np.ones((1, test_width, test_height)), axis=0)
                
        # Just select some random dates for the test set
        dates = np.random.choice(self.training_dates, fcst_array.shape[0])
        hours = np.random.choice(self.training_hours, fcst_array.shape[0])

        fcst_corrected = qmapper.get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, hours=hours)
        
        # Check no nulls
        self.assertEqual(np.isnan(fcst_corrected).sum(), 0)
        
        if self.ifs_train_data.max() > self.imerg_train_data.max():
            self.assertLess(fcst_corrected.max(), fcst_array.max())
        else:
            self.assertGreater(fcst_corrected.max(), fcst_array.max())
                
            
