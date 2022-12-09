""" Data generator class for full-image evaluation of precipitation downscaling network """
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from dsrnngan.data import load_fcst_radar_batch, load_hires_constants, all_fcst_hours, DATA_PATHS, all_ifs_fields, all_era5_fields
from dsrnngan import read_config
return_dic = True

fields_lookup = {'ifs': all_ifs_fields, 'era5': all_era5_fields}

class DataGenerator(Sequence):
    def __init__(self, dates, batch_size, forecast_data_source, observational_data_source, data_paths=DATA_PATHS,
                 log_precip=True, shuffle=True, constants=True, hour='random', longitude_range=None,
                 latitude_range=None, fcst_norm=True,
                 downsample=False, seed=None):

        self.dates = dates

        if isinstance(hour, str):
            if hour == 'random':
                self.hours = np.repeat(all_fcst_hours, len(self.dates))
                self.dates = np.tile(self.dates, len(all_fcst_hours))
            else:
                assert False, f"Unsupported hour {hour}"

        elif isinstance(hour, (int, np.integer)):
            self.hours = np.repeat(hour, len(self.dates))
            self.dates = np.tile(self.dates, 1)  # lol

        elif isinstance(hour, (list, np.ndarray)):
            self.hours = np.repeat(hour, len(self.dates))
            self.dates = np.tile(self.dates, len(hour))

            if forecast_data_source == 'era5':
                raise ValueError('ERA5 data only supports daily')

        else:
            raise ValueError(f"Unsupported hour {hour}")

        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(seed)
            self.shuffle_data()

        self.forecast_data_source = forecast_data_source
        self.observational_data_source = observational_data_source
        self.data_paths = data_paths
        self.batch_size = batch_size

        self.fcst_fields = fields_lookup[self.forecast_data_source.lower()]
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.hour = hour
        self.fcst_norm = fcst_norm
        self.downsample = downsample
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
               
        if self.downsample:
            # read downscaling factor from file
            self.ds_factor = read_config.read_config()['DOWNSCALING']["downscaling_factor"]
        
        if not constants:
            # Dummy constants
            self.constants = np.ones((self.batch_size, len(self.latitude_range), len(self.longitude_range), 1))
        elif constants is True:
            self.constants = load_hires_constants(self.batch_size,
                                                  lsm_path=data_paths['GENERAL']['LSM'], 
                                                  oro_path=data_paths['GENERAL']['OROGRAPHY'],
                                                  latitude_vals=latitude_range, longitude_vals=longitude_range)
        else:
            self.constants = np.repeat(constants, self.batch_size, axis=0)

    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def _dataset_downsampler(self, radar):
        kernel_tf = tf.constant(1.0/(self.ds_factor*self.ds_factor), shape=(self.ds_factor, self.ds_factor, 1, 1), dtype=tf.float32)
        image = tf.nn.conv2d(radar, filters=kernel_tf, strides=[1, self.ds_factor, self.ds_factor, 1], padding='VALID',
                             name='conv_debug', data_format='NHWC')
        return image

    def __getitem__(self, idx):
        # Get batch at index idx
        dates_batch = self.dates[idx*self.batch_size:(idx+1)*self.batch_size]
        hours_batch = self.hours[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load and return this batch of images
        data_x_batch, data_y_batch = load_fcst_radar_batch(
            dates_batch,
            fcst_dir=self.data_paths['GENERAL'][self.forecast_data_source.upper()],
            obs_data_dir=self.data_paths['GENERAL'][self.observational_data_source.upper()],
            constants_dir=self.data_paths['GENERAL']['CONSTANTS'],
            fcst_fields=self.fcst_fields,
            log_precip=self.log_precip,
            fcst_data_source=self.forecast_data_source,
            obs_data_source=self.observational_data_source,
            hour=hours_batch,
            norm=self.fcst_norm,
            latitude_range=self.latitude_range,
            longitude_range=self.longitude_range)
        
        if self.downsample:
            # replace forecast data by coarsened radar data!
            data_x_batch = self._dataset_downsampler(data_y_batch[..., np.newaxis])

        
        if return_dic:
            return {"lo_res_inputs": data_x_batch,
                    "hi_res_inputs": self.constants,
                    "dates": dates_batch, "hours": hours_batch},\
                    {"output": data_y_batch}
        else:
            return data_x_batch, self.constants, data_y_batch, dates_batch

    def shuffle_data(self):
        assert len(self.hours) == len(self.dates)
        p = np.random.permutation(len(self.hours))
        self.hours = self.hours[p]
        self.dates = self.dates[p]
        return

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


if __name__ == "__main__":
    pass
