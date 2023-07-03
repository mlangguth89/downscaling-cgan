import gc
import os
import copy
import pickle
import numpy as np
from tqdm import tqdm
from typing import Iterable, Generator

from dsrnngan.data import tfrecords_generator
from dsrnngan.data import setupdata
from dsrnngan.data.tfrecords_generator import DataGenerator
from dsrnngan.data.data import DATA_PATHS, denormalise
from dsrnngan.utils.utils import date_range_from_year_month_range, load_yaml_file
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.utils import read_config


def setup_batch_gen(records_folder: str,
                    fcst_shape: tuple,
                    con_shape: tuple,
                    out_shape: tuple,
                    batch_size: int=64,
                    downsample: bool=False,
                    weights=None,
                    crop_size: int=None,
                    seed: int=None,
                    val=False,
                    val_fixed=False,
                    ):

    tfrecords_generator.return_dic = False
    print(f"downsample flag is {downsample}")
    train = DataGenerator('train',
                           batch_size=batch_size,
                           fcst_shape=fcst_shape,
                           con_shape=con_shape,
                           out_shape=out_shape,
                           downsample=downsample, 
                           weights=weights, 
                           crop_size=crop_size,
                           records_folder=records_folder,
                           seed=seed)

    # note -- using create_fixed_dataset with a batch size not divisible by 16 will cause problems [is this true?]
    # create_fixed_dataset will not take a list
    
    # if val:
    #     # assume that val_size is small enough that we can just use one batch
    #     val = tfrecords_generator.create_fixed_dataset(val_years, batch_size=val_size, downsample=downsample,
    #                                                    folder=records_folder)
    #     val = val.take(1)
    #     if val_fixed:
    #         val = val.cache()
    # else:
    #     val = tfrecords_generator.create_fixed_dataset(val_years, batch_size=batch_size, downsample=downsample,
    #                                                    folder=records_folder)
    return train, None


def setup_full_image_dataset(year_month_range,
                             fcst_data_source,
                             obs_data_source,
                             fcst_fields,
                             constant_fields,
                             data_paths,
                             latitude_range,
                             longitude_range,
                             batch_size=2,
                             downsample=False,
                             hour='random',
                             shuffle=True
                             ):

    from dsrnngan.data.data_generator import DataGenerator as DataGeneratorFull
    from dsrnngan.data.data import get_obs_dates

    date_range = date_range_from_year_month_range(year_month_range)
    dates = get_obs_dates(date_range[0], date_range[-1], 
                          obs_data_source=obs_data_source, data_paths=data_paths, hour=hour)
    data_full = DataGeneratorFull(dates=dates,
                                  forecast_data_source=fcst_data_source, 
                                  observational_data_source=obs_data_source,
                                  fields=fcst_fields,
                                  latitude_range=latitude_range,
                                  longitude_range=longitude_range,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  constant_fields=constant_fields,
                                  normalise=True,
                                  downsample=downsample,
                                  data_paths=data_paths,
                                  hour=hour)
    return data_full


def setup_data(fcst_data_source: str,
               obs_data_source: str,
               latitude_range: list[float],
               longitude_range: list[float],
               training_range: list[float]=None,
               validation_range: list[float]=None,
               fcst_fields: list=None,
               records_folder: str=None,
               hour: str='random',
               downsample: bool=False,
               fcst_shape: tuple[int]=(20, 20, 9),
               con_shape: tuple[int]=(200, 200, 1),
               out_shape: tuple[int]=(200, 200, 1),
               constant_fields: list=None, # If True will load constants data
               weights: Iterable=None,
               batch_size: int=None,
               load_full_image: bool=False,
               seed: int=None,
               data_paths: dict=DATA_PATHS,
               crop_size: int=None,
               shuffle: bool=True) -> tuple[Generator]:
    """
        Setup data for training or validation; if load_ful

    Args:
        fcst_data_source (str): Forecast data source, e.g. ifs
        obs_data_source (str): Observation data source e.g. imerg
        latitude_range (list[float]): Latitude range 
        longitude_range (list[float]): Longitude range
        training_range (list[float], optional): Range of training dates, list of YYYYMM format (just the start and end required). Defaults to None.
        validation_range (list[float], optional): Range of validation dates, list of YYYYMM format (just the start and end required). Defaults to None.
        fcst_fields (list, optional): List of fcst fields, for creating full image dataset. Defaults to None for which case the default local config is used
        records_folder (str, optional): Folder with tfrecords in it (required if load_full_image=False). Defaults to None.
        hour (str, optional): Hour to load. Defaults to 'random'.
        downsample (bool, optional): _description_. Defaults to False.
        fcst_shape (tuple[int], optional): _description_. Defaults to (20, 20, 9).
        con_shape (tuple[int], optional): _description_. Defaults to (200, 200, 1).
        out_shape (tuple[int], optional): _description_. Defaults to (200, 200, 1).
        constant_fields (list, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to None.
        load_full_image (bool, optional): _description_. Defaults to False.
        seed (int, optional): _description_. Defaults to None.
        data_paths (dict, optional): _description_. Defaults to DATA_PATHS.
        crop_size (int, optional): _description_. Defaults to None.
        permute_var_index (int, optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to True.

    Returns:
        tuple[Generator]: Tuple containing training data generator and validation data generator
    """
    if load_full_image:
        if training_range is None:
            batch_gen_train = None
        else:
            batch_gen_train = setup_full_image_dataset(training_range,
                                          fcst_data_source=fcst_data_source,
                                          obs_data_source=obs_data_source,
                                          fcst_fields=fcst_fields,
                                          latitude_range=latitude_range,
                                          longitude_range=longitude_range,
                                          batch_size=batch_size,
                                          downsample=downsample,
                                          constant_fields=constant_fields,
                                          data_paths=data_paths,
                                          hour=hour,
                                          shuffle=shuffle)
        if validation_range is None:
            batch_gen_valid = None
        else:
            batch_gen_valid = setup_full_image_dataset(validation_range,
                                          fcst_data_source=fcst_data_source,
                                          obs_data_source=obs_data_source,
                                          fcst_fields=fcst_fields,
                                          latitude_range=latitude_range,
                                          longitude_range=longitude_range,
                                          batch_size=batch_size,
                                          downsample=downsample,
                                          constant_fields=constant_fields,
                                          data_paths=data_paths,
                                          hour=hour,
                                          shuffle=shuffle)

    else:
        if not records_folder:
            raise ValueError('No records folder given')
        batch_gen_train, batch_gen_valid = setup_batch_gen(
            val=False,
            records_folder=records_folder,
            fcst_shape=fcst_shape,
            con_shape=con_shape,
            out_shape=out_shape,
            batch_size=batch_size,
            downsample=downsample,
            weights=weights,
            crop_size=crop_size,
            seed=seed)

    gc.collect()
    return batch_gen_train, batch_gen_valid



def load_data_from_config(config: dict, records_folder: str=None,
                          batch_size: int=1, load_full_image: bool=True, 
                          data_paths: dict=DATA_PATHS, hour: int='random',
                          fcst_fields: list=None):
    """Load data based on config file

    Args:
        config (dict): _description_
        records_folder (str, optional): Folder path to tf records. Defaults to None.
        batch_size (int, optional): Size of batches. Defaults to 1.
        load_full_image (bool, optional): If True will load full sized image. Defaults to True.
        data_paths (dict, optional): Dict containing data paths. Defaults to DATA_PATHS.
        hour (int, optional): Hour to load. Defaults to 'random'.

    Returns:
        _type_: _description_
    """
    
    model_config, _, _, data_config, _, _, train_config, val_config = read_config.get_config_objects(config)
    
    latitude_range=np.arange(data_config.min_latitude, data_config.max_latitude + data_config.latitude_step_size, data_config.latitude_step_size)
    longitude_range=np.arange(data_config.min_longitude, data_config.max_longitude + data_config.latitude_step_size, data_config.longitude_step_size)
    
    data_gen_train, data_gen_valid = setupdata.setup_data(
        records_folder=records_folder,
        fcst_data_source=data_config.fcst_data_source,
        obs_data_source=data_config.obs_data_source,
        fcst_fields=fcst_fields,
        latitude_range=latitude_range,
        longitude_range=longitude_range,
        load_full_image=load_full_image,
        validation_range=val_config.val_range,
        training_range=train_config.training_range,
        batch_size=batch_size,
        downsample=model_config.downsample,
        data_paths=data_paths,
        hour=hour)
    
    return data_gen_train, data_gen_valid

def load_data_from_folder(records_folder: str=None, 
                          batch_size: int=1, load_full_image: bool=True, 
                          data_paths: dict=DATA_PATHS):
    """
    Loads data from folder: if load full image is true then this fetches data directly from source,
    if not it will load from the previously created tfrecords

    Args:
        records_folder (str, optional): Folder with TF records in it; only required if load_full_image=False. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 1.
        load_full_image (bool, optional): Whether or not to load the full image. Defaults to True.
        data_paths (dict, optional): Dict of data paths
        

    Returns:
        data_gen_train, data_gen_valid: generators for training and validation data
    """
    setup_params = load_yaml_file(os.path.join(records_folder, 'local_config.yaml'))
    data_gen_train, data_gen_valid = load_data_from_config(setup_params, batch_size, load_full_image, data_paths)
    return data_gen_train, data_gen_valid

def generate_prediction(data_iterator, generator, 
                        noise_channels, ensemble_size=1, 
                        batch_size=1, denormalise_data=False):
        
    inputs, outputs = next(data_iterator)
    cond = inputs['lo_res_inputs']
    const = inputs['hi_res_inputs']
    dates = inputs['dates']
    hours = inputs['hours']
    
    truth = outputs['output']
    fcst = copy.copy(cond)
    truth = np.expand_dims(np.array(truth), axis=-1)

    if denormalise_data:
        truth = denormalise(truth)
        fcst = denormalise(fcst)
    
    img_gens = []
    for _ in range(ensemble_size):
        
        noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        img_gens.append(generator.predict([cond, const, noise_gen()]))
    
    return img_gens, truth, fcst, dates, hours


if __name__=='__main__':
    
    from dsrnngan.data.data import DATA_PATHS, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE, input_fields

    from dsrnngan.data.data import get_obs_dates
    
    fcst_data_source = 'ifs'
    obs_data_source='imerg'
    full_ym_range = ['201603', '201803']
    n_samples = 'all'
    
    date_range = date_range_from_year_month_range(full_ym_range)
    all_dates = get_obs_dates(date_range[0], date_range[-1], 
                        obs_data_source=obs_data_source, data_paths=DATA_PATHS)
    all_year_months = sorted(set([f"{d.year}{d.month:02d}" for d in all_dates]))

    for ym in all_year_months:
        ym_range = [ym]

        if n_samples == 'all':
            ym_date_range = date_range_from_year_month_range(ym_range)
            ym_dates = get_obs_dates(ym_date_range[0], ym_date_range[-1], 
                                obs_data_source=obs_data_source, data_paths=DATA_PATHS)
            n_samples = 24*len(ym_dates)
    
        data_gen = setup_full_image_dataset(ym_range,
                                fcst_data_source=fcst_data_source,
                                obs_data_source=obs_data_source,
                                constant_fields=True,
                                data_paths=DATA_PATHS,
                                latitude_range=DEFAULT_LATITUDE_RANGE,
                                longitude_range=DEFAULT_LONGITUDE_RANGE,
                                batch_size=1,
                                downsample=False,
                                hour='random',
                                shuffle=False
                                )
        
        tpidx = input_fields.index('tp')
        
        obs_vals, fcst_vals, dates, hours = [], [], [], []
        
        data_idx = 0
            
        for kk in tqdm(range(n_samples)):
            
            try:
                inputs, outputs = data_gen[data_idx]
            except FileNotFoundError:
                print('Could not load file, attempting retries')
                success = False
                for retry in range(5):
                    data_idx += 1
                    print(f'Attempting retry {retry} of 5')
                    try:
                        inputs, outputs = data_gen[data_idx]
                        success = True
                    except FileNotFoundError:
                        pass
                if not success:
                    print(f'Stopping at {dates[data_idx]}')
                    raise FileNotFoundError
            
            inputs, outputs = data_gen[data_idx]

            try:
                cond = inputs['lo_res_inputs']
                fcst = cond[0, :, :, tpidx]
                const = inputs['hi_res_inputs']
                obs = outputs['output'][0, :, :]
                date = inputs['dates']
                hour = inputs['hours']
            except IndexError:
                print(inputs['dates'])
                print(data_idx)
                print(len(dates))
                
                break
            
            obs_vals.append(obs)
            fcst_vals.append(fcst)
            dates.append(date)
            hours.append(hour)
            
            data_idx += 1
            
        obs_array = np.stack(obs_vals, axis=0)
        fcst_array = np.stack(fcst_vals, axis=0)
            
        arrays = {'obs': obs_array, 'fcst_array': fcst_array, 
                'dates': dates, 'hours': hours}
        
        with open(f"training_data_{'_'.join(ym_range)}_{n_samples}.pkl", 'wb+') as ofh:
            pickle.dump(arrays, ofh, pickle.HIGHEST_PROTOCOL)