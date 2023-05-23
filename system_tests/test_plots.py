import sys, os
import unittest
import tempfile
import pickle

from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

HOME = Path(__file__).parents[1]
data_folder = HOME / 'system_tests' / 'data'
tmp_dir = str(data_folder / 'tmp')

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

sys.path.append(str(HOME))

from dsrnngan.evaluation import plots

class TestPlots(unittest.TestCase):
    
    def setUp(self) -> None:
        
        with open(os.path.join(data_folder, 'plot_test_data.pkl'), 'rb') as ifh:
            data = pickle.load(ifh)
        
        self.gridded_data = data['data']
        self.latitude_range = data['latitude_range']
        self.longitude_range = data['longitude_range']
        
        return super().setUp()
    
    def test_contourf(self):
        
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
        plots.plot_contourf(ax, self.gridded_data, 'test', value_range=None, lon_range=self.longitude_range, lat_range=self.latitude_range,
                              cmap='Reds')
        plt.savefig(os.path.join(tmp_dir, 'test_contourf.pdf'), format='pdf')
        
    def test_plot_precipitation(self):
        
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
        plots.plot_precipitation(ax, data=self.gridded_data, title='test', longitude_range=self.longitude_range, latitude_range=self.latitude_range)
        plt.savefig(os.path.join(tmp_dir, 'test_plot_precip.pdf'), format='pdf')
        
    def test_plot_quantile(self):
        
        with open(str(data_folder / 'quantile_mapping' / 'quantile_mapping_test_data.pkl'), 'rb') as ifh:
            train_data = pickle.load(ifh)
        
        ifs_train_data = train_data['ifs_train_data']
        imerg_train_data = train_data['imerg_train_data']
        
        quantile_data_dict = {
                    'fcst': {'data': ifs_train_data, 'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'data': imerg_train_data, 'color': 'k'}
                    }
        
        fig, ax = plt.subplots(1,1)
        
        plots.plot_quantiles(quantile_data_dict, min_data_points_per_quantile=10, ax=ax, 
                             save_path=os.path.join(tmp_dir, 'test_plot_quantiles.pdf'))