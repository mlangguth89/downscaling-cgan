import sys, os
import unittest
import tempfile
import pickle
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta

from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'
tmp_dir = str(data_folder / 'tmp')

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

sys.path.append(str(HOME))

from dsrnngan.evaluation import scoring

class TestScoring(unittest.TestCase):
    
    def setUp(self) -> None:
        
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
    
    def test_csi(self):
        
        csi = scoring.critical_success_index(self.imerg_train_data, self.ifs_train_data, threshold=1)

        self.assertIsInstance(csi, float)
        
        metric_fn = lambda x,y: scoring.critical_success_index(x,y, threshold=1)
        
        csi_by_hour = scoring.get_metric_by_hour(metric_fn=metric_fn, obs_array=self.imerg_train_data, fcst_array=self.ifs_train_data, 
                                                 hours=np.random.choice(range(12), self.imerg_train_data.shape[0]),
                                                 bin_width=4)
    def test_ets(self):
        
        ets = scoring.equitable_threat_score(self.imerg_train_data, self.ifs_train_data, threshold=1)

        self.assertIsInstance(ets, float)
        
        metric_fn = lambda x,y: scoring.equitable_threat_score(x,y, threshold=1)
        
        ets_by_hour = scoring.get_metric_by_hour(metric_fn=metric_fn, obs_array=self.imerg_train_data, fcst_array=self.ifs_train_data, 
                                                 hours=np.random.choice(range(12), self.imerg_train_data.shape[0]),
                                                 bin_width=4)
        
        (n_samples, width, height) = self.imerg_train_data.shape
        # check with mask
        mask = np.random.rand(width, height) > 0.5
        mask = np.stack([mask]*n_samples, axis=0)
        ets_masked = scoring.equitable_threat_score(self.imerg_train_data, self.ifs_train_data, threshold=1, mask=mask)

        
    def test_get_skill_score_results(self):
        
        
        csi_dict = { 'Fcst': self.ifs_train_data
                                }

        csi_results = scoring.get_skill_score_results(
            skill_score_function=scoring.critical_success_index,
            data_dict=csi_dict, obs_array=self.imerg_train_data,
                    hours=np.random.choice(range(24), self.imerg_train_data.shape[0]),
                    hourly_thresholds=[1, 5, 10]
                    )
        
        self.assertIsInstance(csi_results, list)

                
        