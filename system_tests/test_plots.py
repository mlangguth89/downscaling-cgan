import sys, os
import unittest
import tempfile
import pickle
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Polygon

from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'
tmp_dir = str(data_folder / 'tmp')

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

sys.path.append(str(HOME))

from dsrnngan.evaluation import plots, evaluation

class TestPlots(unittest.TestCase):
    
    def setUp(self) -> None:
        
        with open(os.path.join(data_folder, 'plot_test_data.pkl'), 'rb') as ifh:
            data = pickle.load(ifh)
        
        self.gridded_data = data['data']
        self.latitude_range = data['latitude_range']
        self.longitude_range = data['longitude_range']
        
        with open(str(data_folder / 'quantile_mapping' / 'quantile_mapping_test_data.pkl'), 'rb') as ifh:
            train_data = pickle.load(ifh)
        
        self.ifs_train_data = train_data['ifs_train_data']
        self.imerg_train_data = train_data['imerg_train_data']
        
        return super().setUp()
    
    def test_contourf(self):
        
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
        plots.plot_contourf(ax, self.gridded_data, 'test', value_range=None, lon_range=self.longitude_range, lat_range=self.latitude_range,
                              cmap='Reds')
        plt.savefig(os.path.join(tmp_dir, 'test_contourf.pdf'), format='pdf')

    def test_adjust_map_borders(self):
        
        from shapely.geometry import Polygon
        from cartopy.feature import NaturalEarthFeature, auto_scaler
        from dsrnngan.evaluation.plots import EABorderFeature

        border_to_remove = [[47.5,7.85], [47.86, 7.52], [52.54, 10.26], [47.08, 12.7]]
        border_removal_poly = Polygon(border_to_remove)
        
        basic_feature = NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land',
                                            auto_scaler, edgecolor='black', facecolor='never')
        self.assertEqual(len([item for item in list(basic_feature.geometries()) if border_removal_poly.contains(item)]), 1)

            
        feature = EABorderFeature(
                                    'cultural', 'admin_0_boundary_lines_land',
                                     auto_scaler, border_to_remove=border_to_remove, edgecolor='black', facecolor='never')
        self.assertEqual(len([item for item in list(feature.geometries()) if border_removal_poly.contains(item)]), 0)

    def test_plot_precipitation(self):
        
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
        plots.plot_precipitation(ax, data=self.gridded_data, title='test', longitude_range=self.longitude_range, latitude_range=self.latitude_range)
        plt.savefig(os.path.join(tmp_dir, 'test_plot_precip.pdf'), format='pdf')
        
    def test_plot_quantile(self):
        
        
        quantile_data_dict = {
                    'fcst': {'data': self.ifs_train_data, 'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'data': self.imerg_train_data, 'color': 'k'}
                    }
        
        fig, ax = plt.subplots(1,1)
        
        plots.plot_quantiles(quantile_data_dict, min_data_points_per_quantile=10, ax=ax, 
                             save_path=os.path.join(tmp_dir, 'test_plot_quantiles.pdf'))
            
    def test_diurnal_cycle(self):
        
        
        start_date = datetime(2018,12,1)
        dates = [start_date + timedelta(days=n) for n in range(gridded_data.shape[0])]
        hours = range(0,24)
        hourly_sum, hourly_counts = evaluation.get_diurnal_cycle(self.gridded_data,
                                                       dates, hours, 
                                                       longitude_range=self.longitude_range,
                                                       latitude_range=self.latitude_range)
        self.assertIsInstance(hourly_sum, dict)
        self.assertIsInstance(hourly_counts, dict)
        self.assertEqual(set(hourly_counts.keys()), set(hours))
        
