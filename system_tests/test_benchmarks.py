import sys
import unittest
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr


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
            
        # Test that it works with an ensemble for either the obs or forecast
        w = 2
        h = 3
        ensemble_train_data = np.stack([self.ifs_train_data]*5, axis=-1)
        empirical_quantile_map(obs_train=self.imerg_train_data[:,w,h], 
                               model_train=ensemble_train_data, s=array_to_correct[:,w,h],
                               quantiles=quantile_locs, extrapolate='constant')
    
    def test_quantile_map_grid(self):
        
        # try with neighbourhood
        qmapped_fcst = quantile_map_grid(self.fcst_array, fcst_train_data=self.ifs_train_data, 
                      obs_train_data=self.imerg_train_data, quantiles=np.linspace(0,1,10))
        self.assertEqual(np.isnan(qmapped_fcst).sum(), 0)

        # Try with ensemble
        ensemble_train_data = np.stack([self.ifs_train_data]*5, axis=-1)

        qmapped_fcst_ens = quantile_map_grid(self.fcst_array, fcst_train_data=ensemble_train_data, 
                      obs_train_data=self.imerg_train_data, quantiles=np.linspace(0,1,10))
        self.assertEqual(np.isnan(qmapped_fcst).sum(), 0)
        
        self.assertEqual(qmapped_fcst_ens.shape, qmapped_fcst.shape)

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
        self.assertIn(1.0, [item for item in qmapper.quantile_locs])
        
        # Add data to test set that is larger than max of train set
        fcst_array = np.append(self.fcst_array, (self.ifs_train_data.max() + 10)*np.ones((1, test_width, test_height)), axis=0)
                
        # Just select some random dates for the test set
        np.random.seed(seed=0)
        dates = np.random.choice(self.training_dates, fcst_array.shape[0], replace=False)
        np.random.seed(seed=0)
        hours = np.random.choice(self.training_hours, fcst_array.shape[0], replace=False)

        fcst_corrected = qmapper.get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, hours=hours)
        
        # Check no nulls
        self.assertEqual(np.isnan(fcst_corrected).sum(), 0)
        
        if self.ifs_train_data.max() > self.imerg_train_data.max():
            self.assertLess(fcst_corrected.max(), fcst_array.max())
        else:
            self.assertGreater(fcst_corrected.max(), fcst_array.max())
                
        # Check that quantile mapping is fitting well to the training data
        fcst_corrected_train =  qmapper.get_quantile_mapped_forecast(fcst=self.ifs_train_data, dates=self.training_dates, hours=self.training_hours)

        self.assertGreater(pearsonr(np.quantile(self.imerg_train_data, quantile_locs), np.quantile(fcst_corrected_train, quantile_locs)).statistic, 0.999)

    def test_auto_quantiles(self):
        # Check that, if no quantiles are specified, the quantile mapper creates sensible ones based on the training distribution
        
        month_ranges = [[1],[2]]

        qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=self.latitude_range, 
                                 longitude_range=self.longitude_range, num_lat_lon_chunks=2)
        # identify best threshold and train on all the data
        mult_factor = 20
        qmapper.train(fcst_data=np.concatenate([self.ifs_train_data]*mult_factor, axis=0),
                      obs_data=np.concatenate([self.imerg_train_data]*mult_factor, axis=0), 
                      training_dates=np.concatenate([self.training_dates]*mult_factor, axis=0), 
                      training_hours=np.concatenate([self.training_hours]*mult_factor, axis=0))
        
        self.assertGreaterEqual(1 - qmapper.quantile_locs[-2], 1 / self.ifs_train_data.size)
        
    def test_max_val(self):
        from dsrnngan.evaluation.plots import quantile_locs
        # load test data
        with open(str(data_folder / 'quantile_mapping' / 'quantile_mapping_test_data_2.pkl'), 'rb') as ifh:
            train_data = pickle.load(ifh)
        
        ifs_train_data = train_data['ifs_train_data']
        imerg_train_data = train_data['imerg_train_data']
        training_dates = train_data['training_dates']
        training_hours = train_data['training_hours']
        latitude_range = train_data['lat_range']
        longitude_range= train_data['lon_range']
        
        month_ranges = [list(range(1,13))]

        qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, 
                                 longitude_range=longitude_range, quantile_locs=quantile_locs, num_lat_lon_chunks=1)
        # identify best threshold and train on all the data
        qmapper.train(fcst_data=ifs_train_data, obs_data=imerg_train_data, 
                      training_dates=training_dates, training_hours=training_hours)
        quantiles = qmapper.quantiles_by_area['t1_12']['lat0_lon0']
        self.assertEqual(ifs_train_data.max(), np.max(quantiles['fcst_quantiles']))
        fcst_corrected_train =  qmapper.get_quantile_mapped_forecast(ifs_train_data, 
                                                                     dates=training_dates, 
                                                                     hours=training_hours)
        
        self.assertLess(fcst_corrected_train.max(), 30)

        
    def test_minimum_samples_per_quantile(self):
        quantile_locs = [0, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
        qmapper = QuantileMapper(month_ranges=[[1,2]], latitude_range=self.latitude_range, 
                                 longitude_range=self.longitude_range, quantile_locs=quantile_locs, num_lat_lon_chunks=1,
                                 min_data_points_per_quantile=1)
        
        # identify best threshold and train on all the data
        qmapper.train(fcst_data=self.ifs_train_data, obs_data=self.imerg_train_data, 
                      training_dates=self.training_dates, training_hours=self.training_hours)
        
        self.assertListEqual(qmapper.quantile_locs, [0, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99, 0.999, 0.9999])