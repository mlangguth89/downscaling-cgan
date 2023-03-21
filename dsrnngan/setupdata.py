import gc
import os
import copy
import pickle
from glob import glob
import numpy as np
from tqdm import tqdm

from dsrnngan import tfrecords_generator
from dsrnngan.tfrecords_generator import DataGenerator
from dsrnngan.data import DATA_PATHS, denormalise
from dsrnngan.utils import date_range_from_year_month_range, load_yaml_file
from dsrnngan.noise import NoiseGenerator
from dsrnngan import setupmodel, setupdata


def setup_batch_gen(val: bool,
                    records_folder: str,
                    fcst_shape: tuple,
                    con_shape: tuple,
                    out_shape: tuple,
                    batch_size: int=64,
                    val_size: int=None,
                    downsample: bool=False,
                    weights=None,
                    val_fixed=True,
                    crop_size: int=None,
                    seed: int=None
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
                           records_folder=records_folder, 
                           crop_size=crop_size,
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
                             load_constants,
                             data_paths,
                             latitude_range,
                             longitude_range,
                             batch_size=1,
                             downsample=False,
                             hour='random',
                             shuffle=True
                             ):

    from dsrnngan.data_generator import DataGenerator as DataGeneratorFull
    from dsrnngan.data import get_obs_dates

    date_range = date_range_from_year_month_range(year_month_range)
    dates = get_obs_dates(date_range[0], date_range[-1], 
                          obs_data_source=obs_data_source, data_paths=data_paths)
    data_full = DataGeneratorFull(dates=dates,
                                  forecast_data_source=fcst_data_source, 
                                  observational_data_source=obs_data_source,
                                  latitude_range=latitude_range,
                                  longitude_range=longitude_range,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  constants=load_constants,
                                  normalise=True,
                                  downsample=downsample,
                                  data_paths=data_paths,
                                  hour=hour)
    return data_full


def setup_data(records_folder,
               fcst_data_source,
               obs_data_source,
               latitude_range,
               longitude_range,
               training_range=None,
               validation_range=None,
               hour='random',
               val_size=None,
               downsample=False,
               fcst_shape=(20, 20, 9),
               con_shape=(200, 200, 1),
               out_shape=(200, 200, 1),
               load_constants=True, # If True will load constants data
               weights=None,
               batch_size=None,
               load_full_image=False,
               seed=None,
               data_paths=DATA_PATHS,
               crop_size=None,
               shuffle=True):

    if load_full_image:
        if training_range is None:
            batch_gen_train = None
        else:
            batch_gen_train = setup_full_image_dataset(training_range,
                                          fcst_data_source=fcst_data_source,
                                          obs_data_source=obs_data_source,
                                          latitude_range=latitude_range,
                                          longitude_range=longitude_range,
                                          batch_size=batch_size,
                                          downsample=downsample,
                                          load_constants=load_constants,
                                          data_paths=data_paths,
                                          hour=hour,
                                          shuffle=shuffle)
        if validation_range is None:
            batch_gen_valid = None
        else:
            
            batch_gen_valid = setup_full_image_dataset(validation_range,
                                          fcst_data_source=fcst_data_source,
                                          obs_data_source=obs_data_source,
                                          latitude_range=latitude_range,
                                          longitude_range=longitude_range,
                                          batch_size=batch_size,
                                          downsample=downsample,
                                          load_constants=load_constants,
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
            val_size=val_size,
            downsample=downsample,
            weights=weights,
            crop_size=crop_size,
            seed=seed)

    gc.collect()
    return batch_gen_train, batch_gen_valid



def load_model_from_folder(model_folder, model_number=None):
    
    model_weights_root = os.path.join(model_folder, "models")
    config_path = os.path.join(model_folder, 'setup_params.yaml')
    

    if model_number is None:
        model_fp = sorted(glob(os.path.join(model_weights_root, '*.h5')))[-1]
    else:
        model_fp = os.path.join(model_weights_root, f'gen_weights-{model_number:07d}.h5')
        
    setup_params = load_yaml_file(config_path)
    
    df_dict = setup_params['DOWNSCALING']
    mode = setup_params["GENERAL"]["mode"]
    architecture = setup_params["MODEL"]["architecture"]
    padding = setup_params["MODEL"]["padding"]
    input_channels = setup_params['DATA']['input_channels']
    constant_fields = setup_params['DATA']['constant_fields']
    filters_gen = setup_params["GENERATOR"]["filters_gen"]
    noise_channels = setup_params["GENERATOR"]["noise_channels"]
    latent_variables = setup_params["GENERATOR"]["latent_variables"]
    filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]

    print('setting up inputs')
    model = setupmodel.setup_model(mode=mode,
                                   architecture=architecture,
                                   downscaling_steps=df_dict["steps"],
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   padding=padding,
                                   constant_fields=constant_fields)

    gen = model.gen
    
    print('loading weights')
    gen.load_weights(model_fp)
    
    return gen

def load_data_from_folder(records_folder, batch_size=1, load_full_image=True):
    """
    Loads daata from folder: if load full image is true then this fetches data directly from source,
    if not it will load from the previously created tfrecords

    Args:
        records_folder (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 1.
        load_full_image (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    setup_params = load_yaml_file(os.path.join(records_folder, 'local_config.yaml'))

    fcst_data_source=setup_params['DATA']['fcst_data_source']
    obs_data_source=setup_params['DATA']['obs_data_source']
    downsample = setup_params['GENERAL']['downsample']

    training_range = setup_params['TRAIN'].get('training_range')
    validation_range = setup_params['VAL'].get('val_range')
    
    min_latitude = setup_params['DATA']['min_latitude']
    max_latitude = setup_params['DATA']['max_latitude']
    latitude_step_size = setup_params['DATA']['latitude_step_size']
    min_longitude = setup_params['DATA']['min_longitude']
    max_longitude = setup_params['DATA']['max_longitude']
    longitude_step_size = setup_params['DATA']['longitude_step_size']
    latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
    longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)
    
    data_gen_train, data_gen_valid = setupdata.setup_data(
        records_folder=records_folder,
        fcst_data_source=fcst_data_source,
        obs_data_source=obs_data_source,
        latitude_range=latitude_range,
        longitude_range=longitude_range,
        load_full_image=load_full_image,
        validation_range=validation_range,
        training_range=training_range,
        batch_size=batch_size,
        downsample=downsample,
        data_paths=DATA_PATHS)
    
    return data_gen_train, data_gen_valid

def generate_prediction(data_iterator, generator, noise_channels, ensemble_size = 1, 
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
    
    from dsrnngan.data import DATA_PATHS, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE, input_field_lookup
    from dsrnngan.data import get_obs_dates
    
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
                                load_constants=True,
                                data_paths=DATA_PATHS,
                                latitude_range=DEFAULT_LATITUDE_RANGE,
                                longitude_range=DEFAULT_LONGITUDE_RANGE,
                                batch_size=1,
                                downsample=False,
                                hour='random',
                                shuffle=False
                                )
        
        tpidx = input_field_lookup[fcst_data_source.lower()].index('tp')
        
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