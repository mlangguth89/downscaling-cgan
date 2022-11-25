import gc
import os
from debugpy import listen

from dsrnngan import tfrecords_generator
from dsrnngan.tfrecords_generator import DataGenerator
from dsrnngan.data import all_ifs_fields
from dsrnngan.utils import date_range_from_year_month_range


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
                             batch_size=1,
                             downsample=False):

    from dsrnngan.data_generator import DataGenerator as DataGeneratorFull
    from dsrnngan.data import get_obs_dates

    date_range = date_range_from_year_month_range(year_month_range)
    dates = get_obs_dates(date_range[0], date_range[-1], 
                          obs_data_source=obs_data_source)
    data_full = DataGeneratorFull(dates=dates,
                                  forecast_data_source=fcst_data_source, 
                                  observational_data_source=obs_data_source,
                                  batch_size=batch_size,
                                  log_precip=True,
                                  shuffle=True,
                                  constants=load_constants,
                                  hour='random',
                                  fcst_norm=True,
                                  downsample=downsample)
    return data_full


def setup_data(records_folder,
               fcst_data_source,
               obs_data_source,
               training_range=None,
               validation_range=None,
               val_size=None,
               downsample=False,
               fcst_shape=(20, 20, 9),
               con_shape=(200, 200, 1),
               out_shape=(200, 200, 1),
               load_constants=True, # If True will load constants data
               weights=None,
               batch_size=None,
               load_full_image=False,
               seed=None):

    if load_full_image:
        batch_gen_train = None if training_range is None \
            else setup_full_image_dataset(training_range,
                                          fcst_data_source=fcst_data_source,
                                          obs_data_source=obs_data_source,
                                          batch_size=batch_size,
                                          downsample=downsample,
                                          load_constants=load_constants)
        batch_gen_valid = None if validation_range is None \
            else setup_full_image_dataset(validation_range,
                                          fcst_data_source=fcst_data_source,
                                          obs_data_source=obs_data_source,
                                          batch_size=batch_size,
                                          downsample=downsample,
                                          load_constants=load_constants)

    else:
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
            seed=seed)

    gc.collect()
    return batch_gen_train, batch_gen_valid
