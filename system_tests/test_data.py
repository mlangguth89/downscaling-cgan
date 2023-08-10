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

sys.path.append(str(HOME))

from dsrnngan.data.data import infer_lat_lon_names, FIELD_TO_HEADER_LOOKUP_IFS, load_hires_constants, load_imerg_raw, load_era5_day_raw, \
    VAR_LOOKUP_ERA5, interpolate_dataset_on_lat_lon, \
    get_era5_stats, load_fcst_stack, all_ifs_fields, all_era5_fields, load_era5, \
    load_ifs, get_imerg_filepaths, ERA5_PATH, get_ifs_filepath, \
    load_fcst_radar_batch, log_plus_1, filter_by_lat_lon, load_ifs_raw, \
    VAR_LOOKUP_IFS, get_ifs_stats, file_exists, get_dates, load_land_sea_mask, load_orography


#TODO: get local data working with repo: currently only works in situ on the cluster
# era5_path = str(data_folder / 'ERA5')
# era5_daily_path = str(data_folder / 'ERA5_daily')
# # TODO: need to correctly sample the data in HDF5 format
# imerg_folder = str(data_folder / 'IMERG' / 'tmp')
ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = str(data_folder / 'ERA5')
imerg_folder = str(data_folder / 'IMERG/half_hourly/final')

def create_dummy_stats_data(year=2017, fields=all_ifs_fields, constants_path=constants_path, 
                            lat_range=[0,1], lon_range=[33,34]):
    
    for field in fields:
        fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, ifs_path)
        ds = xr.load_dataset(fp)
        
        # Mock IFS stats dicts
        var_name = list(ds.data_vars)[0]
        stats = {'min': np.abs(ds[var_name]).min().values,
            'max': np.abs(ds[var_name]).max().values,
            'mean': ds[var_name].mean().values,
            'std': ds[var_name].std().values}
        
        # Saving it as 2017 since that's the defualt 
        output_fp = f'{constants_path}/IFS_norm_{field}_{year}_lat{lat_range[0]}-{lat_range[-1]}lon{lon_range[0]}-{lon_range[-1]}.pkl'
        with open(output_fp, 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
    

class TestLoad(unittest.TestCase):
    
    def setUp(self) -> None:
        # Create dummy stats to avoid having to recalculate them
        
        self.temp_stats_dir = tempfile.TemporaryDirectory()
        self.temp_stats_dir_name = self.temp_stats_dir.name
                      
        if not os.path.isdir(os.path.join(constants_path, 'tp')):
        
            create_dummy_stats_data()
            create_dummy_stats_data(lon_range=[33,33], lat_range=[0,0])
            
            for field in all_ifs_fields:
                fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, ifs_path)
                ds = xr.load_dataset(fp)
                    
                # Create mock data for calculating stats in IFS
                all_dates = list(pd.date_range(start='2017-01-01', end='2017-12-01', freq='D'))
                all_dates = [item.date() for item in all_dates]

                # create folder structure
                output_suffix = get_ifs_filepath(field, datetime(2016, 1, 1), 12, constants_path).replace(constants_path, '')
                folders = [item for item in output_suffix.split('/')[:-1] if item]

                current_folder = constants_path
                for folder in folders:
                    current_folder = os.path.join(current_folder, folder)
                    if not os.path.isdir(current_folder):
                        os.mkdir(current_folder)
                
                # save replicas of files
                for date in all_dates:
                    year = date.year
                    month = date.month
                    day = date.day
                    
                    output_path = get_ifs_filepath(field, datetime(year, month, day), 12, constants_path)

                    ds.to_netcdf(output_path)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.temp_stats_dir.cleanup()
        return super().tearDown()
    
    
    def test_load_ifs_raw(self):

        year = 2017
        month = 7
        day = 5
        hour = 4
        
        latitude_vals = [0.05, 0.15, 0.25]
        longitude_vals = [33, 34]

        lat_coords = []
        lon_coords = []
        
        for field in all_ifs_fields:
            
            ds = load_ifs_raw(field, year, month, day, hour, ifs_data_dir=str(ifs_path),
                              latitude_vals=latitude_vals, longitude_vals=longitude_vals, 
                              interpolate=False)
            
            self.assertIsInstance(ds, xr.Dataset)
            
            data_var = list(ds.data_vars)[0]
            
            # Check only lat/lon coord, not time, and that dims are correctly ordered
            self.assertEqual(ds[data_var].values.shape, (len(latitude_vals), len(longitude_vals)))
        
            # this also checks that the longitude values are in ascending order
            testing.assert_allclose(ds.latitude.values, np.array([0.05, 0.15, 0.25]), atol=1e-7)
            testing.assert_allclose(ds.longitude.values, np.array([33.05, 34.05]), atol=1e-7)        
            
            lat_var_name, lon_var_name = infer_lat_lon_names(ds)
            
            # check that lat lon are ascending
            self.assertListEqual(list(ds[lat_var_name].values), sorted(ds[lat_var_name].values))
            self.assertListEqual(list(ds[lon_var_name].values), sorted(ds[lon_var_name].values))
            
            lat_coords.append(tuple(sorted(ds.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(ds.coords[lon_var_name].values)))
        
        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)

    def test_load_land_sea_mask(self):

        lsm_path = data_folder / 'constants' / 'lsm_HRES_EAfrica.nc'

        latitude_vals = np.arange(-1, 1, 0.1)
        longitude_vals = np.arange(33, 34, 0.1)
        lsm = load_land_sea_mask(filepath=lsm_path, latitude_vals=latitude_vals, 
                                longitude_vals=longitude_vals)
        self.assertIsInstance(lsm, np.ndarray)
        shape = lsm.shape
        
        self.assertEqual(shape[0], len(latitude_vals))
        self.assertEqual(shape[1], len(longitude_vals))
                
        # check values between 0 and 1
        self.assertLessEqual(lsm.max(), 1.0)
        self.assertGreaterEqual(lsm.min(), 0)

    def test_load_orography(self):
        
        latitude_vals = [-11.95, -11.85, -11.75, -11.65, -11.55, -11.450000000000001, -11.350000000000001, -11.250000000000002, -11.150000000000002, -11.050000000000002, -10.950000000000003, -10.850000000000003, -10.750000000000004, -10.650000000000004, -10.550000000000004, -10.450000000000005, -10.350000000000005, -10.250000000000005, -10.150000000000006, -10.050000000000006, -9.950000000000006, -9.850000000000007, -9.750000000000007, -9.650000000000007, -9.550000000000008, -9.450000000000008, -9.350000000000009, -9.250000000000009, -9.15000000000001, -9.05000000000001, -8.95000000000001, -8.85000000000001, -8.75000000000001, -8.650000000000011, -8.550000000000011, -8.450000000000012, -8.350000000000012, -8.250000000000012, -8.150000000000013, -8.050000000000013, -7.9500000000000135, -7.850000000000014, -7.750000000000014, -7.650000000000015, -7.550000000000015, -7.450000000000015, -7.350000000000016, -7.250000000000016, -7.150000000000016, -7.050000000000017, -6.950000000000017, -6.850000000000017, -6.750000000000018, -6.650000000000018, -6.5500000000000185, -6.450000000000019, -6.350000000000019, -6.2500000000000195, -6.15000000000002, -6.05000000000002, -5.950000000000021, -5.850000000000021, -5.750000000000021, -5.650000000000022, -5.550000000000022, -5.450000000000022, -5.350000000000023, -5.250000000000023, -5.1500000000000234, -5.050000000000024, -4.950000000000024, -4.8500000000000245, -4.750000000000025, -4.650000000000025, -4.550000000000026, -4.450000000000026, -4.350000000000026, -4.250000000000027, -4.150000000000027, -4.050000000000027, -3.9500000000000277, -3.850000000000028, -3.7500000000000284, -3.6500000000000288, -3.550000000000029, -3.4500000000000295, -3.35000000000003, -3.25000000000003, -3.1500000000000306, -3.050000000000031, -2.9500000000000313, -2.8500000000000316, -2.750000000000032, -2.6500000000000323, -2.5500000000000327, -2.450000000000033, -2.3500000000000334, -2.2500000000000338, -2.150000000000034, -2.0500000000000345, -1.9500000000000348, -1.8500000000000352, -1.7500000000000355, -1.6500000000000359, -1.5500000000000362, -1.4500000000000366, -1.350000000000037, -1.2500000000000373, -1.1500000000000377, -1.050000000000038, -0.9500000000000384, -0.8500000000000387, -0.7500000000000391, -0.6500000000000394, -0.5500000000000398, -0.45000000000004015, -0.3500000000000405, -0.25000000000004086, -0.1500000000000412, -0.05000000000004157, 0.04999999999995808, 0.14999999999995772, 0.24999999999995737, 0.349999999999957, 0.44999999999995666, 0.5499999999999563, 0.649999999999956, 0.7499999999999556, 0.8499999999999552, 0.9499999999999549, 1.0499999999999545, 1.1499999999999542, 1.2499999999999538, 1.3499999999999535, 1.449999999999953, 1.5499999999999527, 1.6499999999999524, 1.749999999999952, 1.8499999999999517, 1.9499999999999513, 2.049999999999951, 2.1499999999999506, 2.2499999999999503, 2.34999999999995, 2.4499999999999496, 2.549999999999949, 2.649999999999949, 2.7499999999999485, 2.849999999999948, 2.9499999999999478, 3.0499999999999474, 3.149999999999947, 3.2499999999999467, 3.3499999999999464, 3.449999999999946, 3.5499999999999456, 3.6499999999999453, 3.749999999999945, 3.8499999999999446, 3.9499999999999442, 4.049999999999944, 4.149999999999945, 4.249999999999943, 4.349999999999941, 4.4499999999999424, 4.549999999999944, 4.649999999999942, 4.74999999999994, 4.849999999999941, 4.9499999999999424, 5.04999999999994, 5.149999999999938, 5.24999999999994, 5.349999999999941, 5.449999999999939, 5.549999999999937, 5.649999999999938, 5.74999999999994, 5.8499999999999375, 5.949999999999935, 6.049999999999937, 6.149999999999938, 6.249999999999936, 6.349999999999934, 6.449999999999935, 6.549999999999937, 6.649999999999935, 6.7499999999999325, 6.849999999999934, 6.949999999999935, 7.049999999999933, 7.149999999999931, 7.2499999999999325, 7.349999999999934, 7.449999999999932, 7.54999999999993, 7.649999999999931, 7.7499999999999325, 7.84999999999993, 7.949999999999928, 8.04999999999993, 8.149999999999931, 8.249999999999929, 8.349999999999927, 8.449999999999928, 8.54999999999993, 8.649999999999928, 8.749999999999925, 8.849999999999927, 8.949999999999928, 9.049999999999926, 9.149999999999924, 9.249999999999925, 9.349999999999927, 9.449999999999925, 9.549999999999923, 9.649999999999924, 9.749999999999925, 9.849999999999923, 9.949999999999921, 10.049999999999923, 10.149999999999924, 10.249999999999922, 10.34999999999992, 10.449999999999921, 10.549999999999923, 10.64999999999992, 10.749999999999918, 10.84999999999992, 10.949999999999921, 11.049999999999919, 11.149999999999917, 11.249999999999918, 11.34999999999992, 11.449999999999918, 11.549999999999915, 11.649999999999917, 11.749999999999918, 11.849999999999916, 11.949999999999914, 12.049999999999915, 12.149999999999917, 12.249999999999915, 12.349999999999913, 12.449999999999914, 12.549999999999915, 12.649999999999913, 12.749999999999911, 12.849999999999913, 12.949999999999914, 13.049999999999912, 13.14999999999991, 13.249999999999911, 13.349999999999913, 13.44999999999991, 13.549999999999908, 13.64999999999991, 13.749999999999911, 13.849999999999909, 13.949999999999907, 14.049999999999908, 14.14999999999991, 14.249999999999908, 14.349999999999905, 14.449999999999907, 14.549999999999908, 14.649999999999906, 14.749999999999904, 14.849999999999905, 14.949999999999907]
        longitude_vals = [25.05, 25.150000000000002, 25.250000000000004, 25.350000000000005, 25.450000000000006, 25.550000000000008, 25.65000000000001, 25.75000000000001, 25.850000000000012, 25.950000000000014, 26.050000000000015, 26.150000000000016, 26.250000000000018, 26.35000000000002, 26.45000000000002, 26.550000000000022, 26.650000000000023, 26.750000000000025, 26.850000000000026, 26.950000000000028, 27.05000000000003, 27.15000000000003, 27.250000000000032, 27.350000000000033, 27.450000000000035, 27.550000000000036, 27.650000000000038, 27.75000000000004, 27.85000000000004, 27.950000000000042, 28.050000000000043, 28.150000000000045, 28.250000000000046, 28.350000000000048, 28.45000000000005, 28.55000000000005, 28.650000000000052, 28.750000000000053, 28.850000000000055, 28.950000000000056, 29.050000000000058, 29.15000000000006, 29.25000000000006, 29.350000000000062, 29.450000000000063, 29.550000000000065, 29.650000000000066, 29.750000000000068, 29.85000000000007, 29.95000000000007, 30.05000000000007, 30.150000000000073, 30.250000000000075, 30.350000000000076, 30.450000000000077, 30.55000000000008, 30.65000000000008, 30.75000000000008, 30.850000000000083, 30.950000000000085, 31.050000000000086, 31.150000000000087, 31.25000000000009, 31.35000000000009, 31.45000000000009, 31.550000000000093, 31.650000000000095, 31.750000000000096, 31.850000000000097, 31.9500000000001, 32.0500000000001, 32.150000000000105, 32.2500000000001, 32.35000000000011, 32.4500000000001, 32.55000000000011, 32.650000000000105, 32.750000000000114, 32.85000000000011, 32.95000000000012, 33.05000000000011, 33.15000000000012, 33.250000000000114, 33.35000000000012, 33.45000000000012, 33.550000000000125, 33.65000000000012, 33.75000000000013, 33.85000000000012, 33.95000000000013, 34.050000000000125, 34.150000000000134, 34.25000000000013, 34.350000000000136, 34.45000000000013, 34.55000000000014, 34.650000000000134, 34.75000000000014, 34.850000000000136, 34.950000000000145, 35.05000000000014, 35.15000000000015, 35.25000000000014, 35.35000000000015, 35.450000000000145, 35.55000000000015, 35.65000000000015, 35.750000000000156, 35.85000000000015, 35.95000000000016, 36.05000000000015, 36.15000000000016, 36.250000000000156, 36.350000000000165, 36.45000000000016, 36.55000000000017, 36.65000000000016, 36.75000000000017, 36.850000000000165, 36.95000000000017, 37.05000000000017, 37.150000000000176, 37.25000000000017, 37.35000000000018, 37.45000000000017, 37.55000000000018, 37.650000000000176, 37.750000000000185, 37.85000000000018, 37.95000000000019, 38.05000000000018, 38.15000000000019, 38.250000000000185, 38.35000000000019, 38.45000000000019, 38.550000000000196, 38.65000000000019, 38.7500000000002, 38.85000000000019, 38.9500000000002, 39.050000000000196, 39.150000000000205, 39.2500000000002, 39.35000000000021, 39.4500000000002, 39.55000000000021, 39.650000000000205, 39.75000000000021, 39.85000000000021, 39.950000000000216, 40.05000000000021, 40.15000000000022, 40.25000000000021, 40.35000000000022, 40.450000000000216, 40.550000000000225, 40.65000000000022, 40.75000000000023, 40.85000000000022, 40.95000000000023, 41.050000000000225, 41.15000000000023, 41.25000000000023, 41.350000000000236, 41.45000000000023, 41.55000000000024, 41.65000000000023, 41.75000000000024, 41.850000000000236, 41.950000000000244, 42.05000000000024, 42.15000000000025, 42.25000000000024, 42.35000000000025, 42.450000000000244, 42.55000000000025, 42.65000000000025, 42.750000000000256, 42.85000000000025, 42.95000000000026, 43.05000000000025, 43.15000000000026, 43.250000000000256, 43.350000000000264, 43.45000000000026, 43.55000000000027, 43.65000000000026, 43.75000000000027, 43.850000000000264, 43.95000000000027, 44.05000000000027, 44.150000000000276, 44.25000000000027, 44.35000000000028, 44.45000000000027, 44.55000000000028, 44.650000000000276, 44.750000000000284, 44.85000000000028, 44.95000000000029, 45.05000000000028, 45.15000000000029, 45.250000000000284, 45.35000000000029, 45.45000000000029, 45.550000000000296, 45.65000000000029, 45.7500000000003, 45.85000000000029, 45.9500000000003, 46.050000000000296, 46.150000000000304, 46.2500000000003, 46.35000000000031, 46.4500000000003, 46.55000000000031, 46.650000000000304, 46.75000000000031, 46.85000000000031, 46.950000000000315, 47.05000000000031, 47.15000000000032, 47.25000000000031, 47.35000000000032, 47.450000000000315, 47.550000000000324, 47.65000000000032, 47.75000000000033, 47.85000000000032, 47.95000000000033, 48.050000000000324, 48.15000000000033, 48.25000000000033, 48.350000000000335, 48.45000000000033, 48.55000000000034, 48.65000000000033, 48.75000000000034, 48.850000000000335, 48.950000000000344, 49.05000000000034, 49.15000000000035, 49.25000000000034, 49.35000000000035, 49.450000000000344, 49.55000000000035, 49.65000000000035, 49.750000000000355, 49.85000000000035, 49.95000000000036, 50.05000000000035, 50.15000000000036, 50.250000000000355, 50.350000000000364, 50.45000000000036, 50.55000000000037, 50.65000000000036, 50.75000000000037, 50.850000000000364, 50.95000000000037, 51.05000000000037, 51.150000000000375, 51.25000000000037]
        
        # oro_ds = xr.load_dataset('/bp1/geog-tropical/users/uz22147/east_africa_data/constants/h_HRES_EAfrica.nc')
        # oro_ds = filter_by_lat_lon(oro_ds, lon_range=lon_range_list, lat_range=lat_range_list)
        
        # oro_path = data_folder / 'constants' / 'h_HRES_EAfrica.nc'
        oro_path = '/bp1/geog-tropical/users/uz22147/east_africa_data/constants/h_HRES_EAfrica.nc'

        # latitude_vals = np.arange(-1, 1, 0.1)
        # longitude_vals = np.arange(33, 34, 0.1)
        h = load_orography(filepath=oro_path,  latitude_vals=latitude_vals, 
                                longitude_vals=longitude_vals,
                                interpolate=False)
        self.assertIsInstance(h, np.ndarray)
        shape = h.shape
        
        self.assertEqual(shape[0], len(latitude_vals))
        self.assertEqual(shape[1], len(longitude_vals))
        
        # check values between 0 and 1
        self.assertLessEqual(h.max(), 1.0)
        self.assertGreaterEqual(h.min(), 0)
    
    def test_load_hires_constants(self):
        
        lsm_path = data_folder / 'constants' / 'lsm_HRES_EAfrica.nc'
        oro_path = data_folder / 'constants' / 'h_HRES_EAfrica.nc'
        
        data_paths = {'lsm': lsm_path, 'orography': oro_path, 'lakes': lsm_path,
                      'sea': lsm_path}
        
        latitude_vals = np.arange(-1, 1, 0.1)
        longitude_vals = np.arange(33, 34, 0.1)
        batch_size = 4
        
        fields = ['lsm', 'orography', 'lakes', 'sea']
        c = load_hires_constants( fields=fields,
                                    data_paths=data_paths,
                                 batch_size=batch_size, latitude_vals=latitude_vals, 
                                 longitude_vals=longitude_vals)
        self.assertEqual(c.shape, (batch_size, len(latitude_vals), len(longitude_vals), len(fields)))
        
        # check values between 0 and 1
        self.assertLessEqual(c.max(), 1.0)
        self.assertGreaterEqual(c.min(), 0)
    
    def test_load_era5_raw(self):

        year = 2018
        month = 12
        day = 1

        lat_coords = []
        lon_coords = []

        var_name_lookup = VAR_LOOKUP_ERA5

        for v in var_name_lookup:
    
            ds1 = load_era5_day_raw(v, year=year, month=month, day=day,
                                    latitude_vals=[0, 0.1, 0.2], longitude_vals=[33, 33.1, 33.2],
                                    era_data_dir=str(era5_path))

            self.assertIsInstance(ds1, xr.Dataset)
            
            lat_var_name, lon_var_name = infer_lat_lon_names(ds1)
            
            # check that lat lon are ascending
            self.assertListEqual(list(ds1[lat_var_name].values), sorted(ds1[lat_var_name].values))
            self.assertListEqual(list(ds1[lon_var_name].values), sorted(ds1[lon_var_name].values))
       
            lat_coords.append(tuple(sorted(ds1.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(ds1.coords[lon_var_name].values)))

        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)

    def test_precipitation_scaling(self):

        # Check that pre
        # year = 2018
        # month = 12
        # day = 1

        # ds_raw = load_era5_day_raw('tp', year=year, month=month, day=day, 
        #                            latitude_vals=[0, 0.1], longitude_vals=[33, 33.1],
        #                            era_data_dir=str(era5_path))
        
        # testing.assert_allclose(ds_raw['tp'].values[0], np.array([[3.36921681, 3.35620437], [3.24860298, 3.19705309]]), atol=1e-8)
        
        ## Same for IFS
        year = 2017
        month = 7
        day = 5
        hour = 18

        ds_raw = load_ifs_raw('tp', year=year, month=month, day=day, hour=hour,
                                   latitude_vals=[-1.75, -1.85], longitude_vals=[34.35, 34.45],
                                   ifs_data_dir=str(ifs_path), interpolate=False)
        
        testing.assert_allclose(ds_raw['tp'].values, np.array(np.array([[3.98702919e-04, 6.53415918e-06], [0.00081483, 0.]])), atol=1e-8)

    def test_era5_load_norm_logs(self):

        year = 2018
        month = 12
        day = 1
        lat_vals = np.arange(0, 1, 0.1)
        lon_vals = np.arange(33, 34, 0.1)

        var_name_lookup = VAR_LOOKUP_ERA5

        for v in var_name_lookup:
            if v == 'tp':
                # check log precipitation is working
                ds_tp_no_log = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=False, norm=False,
                                         fcst_dir=str(era5_path),
                                         latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
                ds_tp_log = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=True, norm=False,
                                      fcst_dir=str(era5_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
                ds_tp_log_norm = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=True, norm=True,
                                           fcst_dir=str(era5_path),
                                           latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)

                testing.assert_array_equal(ds_tp_log, ds_tp_log_norm)
                testing.assert_array_equal(ds_tp_log, log_plus_1(ds_tp_no_log))

            # Try loading with normalisation
            ds_tp_norm = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=False, norm=True,
                                   fcst_dir=str(era5_path),
                                   latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
            ds_tp_no_norm = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=False, norm=False,
                                      fcst_dir=str(era5_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)

            normalisation = var_name_lookup.get('normalisation')
            stats_dict = get_era5_stats(v, output_dir=constants_path, era_data_dir=era5_path,
                                        latitude_vals=lat_vals, longitude_vals=lon_vals)

            if normalisation == 'minmax':
                testing.assert_array_equal(ds_tp_norm,
                                 (ds_tp_no_norm - stats_dict['min']) / (stats_dict['max'] - stats_dict['min']))

    def test_ifs_load_norm_logs(self):

        year = 2017
        month = 7
        day = 5
        hour=4
        
        lat_vals = np.arange(0, 1, 0.1)
        lon_vals = np.arange(33, 34, 0.1)

        var_name_lookup = VAR_LOOKUP_IFS

        for v in var_name_lookup:
            if v == 'tp':
                # check log precipitation is working
                tp_no_log = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour,  norm=False,
                                         fcst_dir=str(ifs_path),
                                         latitude_vals=lat_vals, longitude_vals=lon_vals, 
                                         constants_path=constants_path)
                tp_log_norm = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour, norm=True,
                                      fcst_dir=str(ifs_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, 
                                      constants_path=constants_path)

                testing.assert_array_equal(tp_log_norm, log_plus_1(tp_no_log))
            
            # Try loading with normalisation
            arr_norm = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour, norm=True,
                                   fcst_dir=str(ifs_path),
                                   latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
            arr_no_norm = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour, norm=False,
                                      fcst_dir=str(ifs_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)

            self.assertEquals(np.isnan(arr_norm).sum(), 0)
            self.assertEquals(np.isnan(arr_no_norm).sum(), 0)

            normalisation = var_name_lookup.get('normalisation')
            stats_dict = get_ifs_stats(v, output_dir=constants_path, ifs_data_dir=ifs_path,
                                        latitude_vals=lat_vals, longitude_vals=lon_vals)

            if normalisation == 'minmax':
                testing.assert_array_equal(arr_norm,
                                 (arr_no_norm - stats_dict['min']) / (stats_dict['max'] - stats_dict['min']))


    def test_filter_on_lat_lon(self):
        
        year = 2018
        month = 12
        day = 30
        hour = 18

        # Fetch raw era5 data
        raw_ds = load_era5_day_raw('tp', year=year, month=month, day=day,
                                   era_data_dir=era5_path, interpolate=False
                                   )
        
        filtered_ds = filter_by_lat_lon(raw_ds, lon_range=[33, 34], lat_range=[0, 1],
                                        lat_var_name='latitude', lon_var_name='longitude')
        
        testing.assert_array_equal(filtered_ds.latitude.values, np.arange(0, 1.25, 0.25))
        testing.assert_array_equal(filtered_ds.longitude.values, np.arange(33, 34.25, 0.25))
        
        # Check with imerg
        # Fetch raw imerg data
        latitude_vals=np.arange(-2, 2, 0.1)
        longitude_vals=np.arange(29, 31, 0.1)
        rainy_patch_ds = xr.open_dataset(os.path.join(data_folder, 'IMERG/rainy_patch.nc'))
        filtered_ds = filter_by_lat_lon(rainy_patch_ds, lon_range=longitude_vals, lat_range=latitude_vals)
        
        # This is to check that it isn't returning duplicate lat/lon values (which happens with sel and method='nearest')
        self.assertEqual(len(filtered_ds.lon.values), len(set(filtered_ds.lon.values)))
        self.assertEqual(len(filtered_ds.lat.values), len(set(filtered_ds.lat.values)))

    def test_interpolation(self):

        year = 2018
        month = 12
        day = 30
        hour = 18

        # Fetch raw imerg data
        raw_ds = load_era5_day_raw('tp', year=year, month=month, day=day,
                                   era_data_dir=era5_path, interpolate=False
                                   )
        raw_ds = raw_ds.sel(latitude=[0, 0.25, 0.5], longitude=np.arange(33, 34.5, 0.25), method='nearest')

        regridded_ds = interpolate_dataset_on_lat_lon(raw_ds, 
                                                      latitude_vals=[0.1], longitude_vals=[33.1],
                                                      interp_method='bilinear',)
        regridded_val = regridded_ds.tp.values[0][0][0]

        # Check that the linear interpolation is looking reasonable
        x1 = raw_ds.longitude.values[0]
        x2 = raw_ds.longitude.values[1]
        y1 = raw_ds.latitude.values[0]
        y2 = raw_ds.latitude.values[1]

        f11 = raw_ds.sel(longitude=x1, latitude=y1, method='nearest').tp.values[0]
        f12 = raw_ds.sel(longitude=x1, latitude=y2, method='nearest').tp.values[0]
        f21 = raw_ds.sel(longitude=x2, latitude=y1, method='nearest').tp.values[0]
        f22 = raw_ds.sel(longitude=x2, latitude=y2, method='nearest').tp.values[0]

        x = 33.1
        y = 0.1

        fxy1 = ((x2 - x) / (x2 - x1)) * f11 + ((x - x1) / (x2 - x1)) * f21
        fxy2 = ((x2 - x) / (x2 - x1)) * f12 + ((x - x1) / (x2 - x1)) * f22

        fxy = ((y2 - y) / (y2 - y1)) * fxy1 + ((y - y1) / (y2 - y1)) * fxy2

        # Not an exact match, but I'm not sure if there are other things
        # that xesmf is doing
        self.assertAlmostEqual(regridded_val, fxy, 5)

        # Check that it throws an error if the data isn't large enough to interpolate with
        with self.assertRaises(ValueError):
            interpolate_dataset_on_lat_lon(raw_ds, 
                                           latitude_vals=[-2, -1, 0], 
                                           longitude_vals=[30, 31, 32, 33])

        ##
        # Check interpolation to smaller grid with era5

    def test_imerg_fps(self):
        
        fps = get_imerg_filepaths(2019, 1, 1, 0, file_ending='.nc')
        
        self.assertEqual(fps[0].split('/')[-1], '3B-HHR.MS.MRG.3IMERG.20190101-S000000-E002959.0000.V06B.nc')
        self.assertEqual(fps[1].split('/')[-1], '3B-HHR.MS.MRG.3IMERG.20190101-S003000-E005959.0030.V06B.nc')
        
    def test_load_imerg(self):

        year = 2018
        month = 12
        day = 30
        hour = 17
        latitude_vals = [0, 0.1, 0.2]
        longitude_vals = [33, 34]

        ds = load_imerg_raw(year=year, month=month, day=day, hour=hour,
                            latitude_vals=latitude_vals,
                            longitude_vals=longitude_vals,
                            imerg_data_dir=imerg_folder)
        
        self.assertIsInstance(ds, xr.Dataset)
        
        # Check that dimensions are ordered properly
        self.assertListEqual(list(ds['precipitationCal'].dims), ['lat', 'lon'])
        self.assertEqual(ds['precipitationCal'].values.shape, (len(latitude_vals), len(longitude_vals)))
        
        # this also checks that the longitude values are in ascending order
        testing.assert_allclose(ds.lat.values, np.array([0.05, 0.15, 0.25]), atol=1e-7)
        testing.assert_allclose(ds.lon.values, np.array([33.05, 34.05]), atol=1e-7)
        
        # Check for NaNs
        self.assertFalse(np.isnan(ds['precipitationCal']).any())
        
        with self.assertRaises(ValueError):
            load_imerg_raw(year=year, month=month, day=day, hour=hour,
                            latitude_vals=latitude_vals,
                            longitude_vals=[330.05, 330.15],
                            imerg_data_dir=imerg_folder)

    def test_era5_stats(self):

        var_name_lookup = VAR_LOOKUP_ERA5
        for v in var_name_lookup:
            stats = get_era5_stats(v, year=2018, era_data_dir=constants_path,
                                   longitude_vals=[33, 34], latitude_vals=[0, 1])

            self.assertIsInstance(stats, dict)
            # Make sure negative velocities handled correctly
            self.assertGreater(stats['min'] + 1e-16, 0)
            self.assertGreater(stats['max'], 0)
            self.assertGreater(stats['std'], 0)
            
    def test_ifs_stats(self):

        var_name_lookup = VAR_LOOKUP_IFS
        for v in var_name_lookup:
            stats = get_ifs_stats(v, year=2017,
                                   longitude_vals=[33, 34], latitude_vals=[0, 1],
                                   output_dir=self.temp_stats_dir.name)

            self.assertIsInstance(stats, dict)
            # Make sure negative velocities handled correctly
            self.assertGreater(stats['min'] + 1e-16, 0)
            self.assertGreater(stats['max'], 0)
            self.assertGreater(stats['std'], 0)

    def test_load_fcst_stack(self):
        
        longitude_vals = [33, 34]
        latitude_vals = [0, 1]

        # Check it works with IFS
        ifs_stack = load_fcst_stack('ifs', all_ifs_fields, '20170705', 4, fcst_dir=ifs_path,
                                    norm=True, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                                    constants_dir=constants_path)
        ifs_shape = ifs_stack.shape
        self.assertEqual(ifs_shape[2], len(all_ifs_fields))
        self.assertFalse(np.isnan(ifs_stack).any())

    def test_load_fcst_radar_batch(self):
        
        longitude_vals = [33, 34]
        latitude_vals = [0, 1]

        # TODO: get NIMROD data sample to enable this test to work with IFS + NIMROD
        # Check it works with IFS: 

        ifs_batch_dates = ['20170704'] 
        ifs_input_batch, imerg_batch = load_fcst_radar_batch(ifs_batch_dates, fcst_data_source='ifs', obs_data_source='imerg', fcst_fields=all_ifs_fields, 
                                          fcst_dir=ifs_path, obs_data_dir=imerg_folder, latitude_range=latitude_vals,
                                          longitude_range=longitude_vals, constants_dir=constants_path,
                                          constants=True, hour=17, normalise_inputs=False)
        self.assertEqual(len(ifs_input_batch), 2)
        self.assertEqual(len(ifs_input_batch[0]), len(ifs_batch_dates))
        self.assertEqual(len(ifs_input_batch[1]), len(ifs_batch_dates))
        
        self.assertFalse(np.isnan(ifs_input_batch[0]).any())
        self.assertFalse(np.isnan(ifs_input_batch[1]).any())
        self.assertFalse(np.isnan(imerg_batch).any())

    def test_get_dates(self):
        data_paths = {'GENERAL': {
                            'IFS': str(data_folder / 'IFS'),
                            'IMERG': str(data_folder / 'IMERG/half_hourly/final'),
                            'ERA5': str(data_folder / 'ERA5'),
                            'CONSTANTS': str(data_folder / 'constants')}}
        dates = get_dates(2018, obs_data_source='imerg', fcst_data_source='era5',
              data_paths=data_paths)
        self.assertListEqual(['20181230', '20181231'], dates)
       
    def test_file_exists(self):

        data_paths = {'GENERAL': {
                            'IFS': str(data_folder / 'IFS'),
                            'IMERG': str(data_folder / 'IMERG/half_hourly/final'),
                            'ERA5': str(data_folder / 'ERA5'),
                            'CONSTANTS': str(data_folder / 'constants')}}
        
        ## ERA5
        era5_file_exists = file_exists(data_source='era5', year=2018, month=12, day=31,
                                       data_paths=data_paths)
        self.assertTrue(era5_file_exists)
        
        era5_file_exists = file_exists(data_source='era5', year=2001, month=1, day=31,
                                       data_paths=data_paths)
        self.assertFalse(era5_file_exists)
        
        ## IFS
        ifs_file_exists = file_exists(data_source='ifs', year=2017, month=7, day=4,
                                       data_paths=data_paths)
        self.assertTrue(ifs_file_exists)
        
        ifs_file_exists = file_exists(data_source='ifs', year=2017, month=1, day=31,
                                       data_paths=data_paths)
        self.assertFalse(ifs_file_exists)
        
        ## IMERG
        imerg_file_exists = file_exists(data_source='imerg', year=2018, month=12, day=30,
                                       data_paths=data_paths)
        self.assertTrue(imerg_file_exists)
        
        imerg_file_exists = file_exists(data_source='imerg', year=2018, month=1, day=31,
                                       data_paths=data_paths)
        self.assertFalse(imerg_file_exists)
        
if __name__ == '__main__':
    unittest.main()
