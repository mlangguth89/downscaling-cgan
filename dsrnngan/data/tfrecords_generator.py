import os
import glob
import random
import time
import numpy as np
import tensorflow as tf
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import git

from dsrnngan.utils import read_config
from dsrnngan.data.data import file_exists, denormalise
from dsrnngan.utils.utils import hash_dict, write_to_yaml, date_range_from_year_month_range

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

return_dic = True

DATA_PATHS = read_config.get_data_paths()
records_folder = DATA_PATHS["TFRecords"]["tfrecords_path"]

#TODO: pass this as function argument instead
model_config = read_config.read_model_config()
ds_fac = model_config.downscaling_factor

# Use autotune to tune the prefetching of records in parrallel to processing to improve performance
AUTOTUNE = tf.data.AUTOTUNE

def DataGenerator(data_label, batch_size, fcst_shape, con_shape, 
                  out_shape, repeat=True, 
                  downsample=False, weights=None, crop_size=None,
                  records_folder=records_folder, seed=None):
    return create_mixed_dataset(data_label, 
                                batch_size,
                                fcst_shape,
                                con_shape,
                                out_shape,
                                repeat=repeat, 
                                downsample=downsample, 
                                weights=weights, 
                                crop_size=crop_size,
                                folder=records_folder, 
                                seed=seed)


def create_mixed_dataset(data_label: str,
                         batch_size: int,
                         fcst_shape: tuple[int, int, int]=(20, 20, 9),
                         con_shape: tuple[int, int, int]=(200, 200, 2),
                         out_shape: tuple[int, int, int]=(200, 200, 1),
                         repeat: bool=True,
                         downsample: bool=False,
                         folder: str=records_folder,
                         shuffle_size: int=1024,
                         weights: list=None,
                         crop_size: int=None,
                         seed: int=None):
    """_summary_

    Args:
        year (int): year of data
        batch_size (int): size of batches
        fcst_shape (tuple[int, int, int], optional): shape of forecast input. Defaults to (20, 20, 9).
        con_shape (tuple[int, int, int], optional): shape of constants input. Defaults to (200, 200, 2).
        out_shape (tuple[int, int, int], optional): shape of output. Defaults to (200, 200, 1).
        repeat (bool, optional): repeat dataset or not. Defaults to True.
        downsample (bool, optional): whether to downsample or not. Defaults to False.
        folder (str, optional): folder containing tf records. Defaults to records_folder.
        shuffle_size (int, optional): buffer size of shuffling. Defaults to 1024.
        weights (list, optional): list of floats, weights of classes when sampling. Defaults to None.
        crop_size (int, optional): Size to crop randomly crop images to.
        seed (int, optional): seed for shuffling and sampling. Defaults to None.

    Returns:
        _type_: _description_
    """    

    classes = 4
    if weights is None:
        weights = [1./classes]*classes
    datasets = [create_dataset(data_label,
                               i,
                               fcst_shape=fcst_shape,
                               con_shape=con_shape,
                               out_shape=out_shape,
                               folder=folder,
                               shuffle_size=shuffle_size,
                               repeat=repeat,
                               crop_size=crop_size,
                               seed=seed)
                for i in range(classes)]
    
    sampled_ds = tf.data.Dataset.sample_from_datasets(datasets,
                                                      weights=weights, seed=seed).batch(batch_size)

    
    if downsample:
        if return_dic:
            sampled_ds = sampled_ds.map(_dataset_downsampler)
        else:
            sampled_ds = sampled_ds.map(_dataset_downsampler_list)
        
    sampled_ds = sampled_ds.prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together

def _dataset_downsampler(inputs, outputs):
    image = outputs['output']
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], padding='VALID',
                         name='conv_debug', data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs, outputs


def _dataset_downsampler_list(inputs, constants, outputs):
    image = outputs
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], padding='VALID', name='conv_debug', data_format='NHWC')
    inputs = image
    return inputs, constants, outputs

def _dataset_cropper_dict(inputs, outputs, crop_size, seed=None):
    '''
    Random crop of inputs and outputs
    
    Note: this currently only works when inputs and outputs are all the same dimensions (i.e not downscaling)
    
    '''
    outputs = outputs['output']
    hires_inputs = inputs['hi_res_inputs']
    lores_inputs = inputs['lo_res_inputs']
    
    cropped_lores_input, cropped_hires_input, cropped_output = _dataset_cropper_list(lores_inputs, hires_inputs, outputs, crop_size, seed)
    
    cropped_outputs = {'output': cropped_output}
    cropped_inputs = {'hi_res_inputs': cropped_hires_input,
                      'lo_res_inputs': cropped_lores_input}
    
    return cropped_inputs, cropped_outputs

def _dataset_cropper_list(lores_inputs, hires_inputs, outputs, crop_size, seed):
    
    (_, _, lores_channels) = lores_inputs.shape
    (_, _, hires_channels) = hires_inputs.shape
    
    if not seed:
        # Choose random seed (to make sure consistent selection)
        seed = (np.random.randint(1e6), np.random.randint(1e6))
    
    cropped_output = tf.image.stateless_random_crop(outputs, size=[crop_size, crop_size, 1], seed=seed)
    cropped_hires_input = tf.image.stateless_random_crop(hires_inputs, size=[crop_size, crop_size, hires_channels] , seed=seed)
    cropped_lores_input = tf.image.stateless_random_crop(lores_inputs, size=[crop_size, crop_size, lores_channels], seed=seed)
    
    return cropped_lores_input, cropped_hires_input, cropped_output


def _parse_batch(record_batch,
                 insize=(20, 20, 9),
                 consize=(200, 200, 2),
                 outsize=(200, 200, 1)):
    """_summary_

    Args:
        record_batch (tf.python.framework.ops.EagerTensor): Single item from a tf Records Dataset
        insize (tuple, optional): shape of the forecast data (lat, lon, n_features). Defaults to (20, 20, 9).
        consize (tuple, optional): shape of the constants data (lat, lon, n_features). Defaults to (200, 200, 2).
        outsize (tuple, optional): shape of the output (lat, lon, n_features). Defaults to (200, 200, 1).

    Returns:
        tuple: tuple of dicts, containing inputs and outputs
    """
    # Create a description of the features
    

    feature_description = {
        'generator_input': tf.io.FixedLenFeature(insize, tf.float32),
        'constants': tf.io.FixedLenFeature(consize, tf.float32),
        'generator_output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    
    if return_dic:
        output = ({'lo_res_inputs': example['generator_input'],
                   'hi_res_inputs': example['constants']},
                  {'output': example['generator_output']})

        return output
    else:
        return example['generator_input'], example['constants'], example['generator_output']


def create_dataset(data_label: str,
                   clss: str,
                   fcst_shape=(20, 20, 9),
                   con_shape=(200, 200, 2),
                   out_shape=(200, 200, 1),
                   folder: str=records_folder,
                   shuffle_size: int=1024,
                   repeat: bool=True,
                   crop_size: int=None,
                   seed: int=None):
    """
    Load tfrecords and parse into appropriate shapes

    Args:
        year (int): year
        clss (str): class (bin to take data from)
        fcst_shape (tuple, optional): shape of the forecast data (lat, lon, n_features). Defaults to (20, 20, 9).
        con_shape (tuple, optional): shape of the constants data (lat, lon, n_features). Defaults to (200, 200, 2).
        out_shape (tuple, optional): shape of the output (lat, lon, n_features). Defaults to (200, 200, 1).
        folder (_type_, optional): folder containing tf records. Defaults to records_folder.
        shuffle_size (int, optional): buffer size for shuffling. Defaults to 1024.
        crop_size (int, optional): Size to crop randomly crop images to.
        repeat (bool, optional): create repeat dataset or not. Defaults to True.

    Returns:
        tf.data.DataSet: _description_
    """
    
    if seed:
        if not isinstance(seed, int):
            int_seed = seed[0]
        else:
            int_seed = seed
    else:
        int_seed = None
    
    files_ds = tf.data.Dataset.list_files(f"{folder}/{data_label}_*.{clss}.*.tfrecords")
     
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=AUTOTUNE)
    
    
    ds = ds.shuffle(shuffle_size, seed=int_seed)

    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
                # num_parallel_calls=AUTOTUNE)
                
    if crop_size:
        if return_dic:
            ds = ds.map(lambda x,y: _dataset_cropper_dict(x, y, crop_size=crop_size, seed=seed), 
                                                          num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda x,y,z: _dataset_cropper_list(x, y, z, crop_size=crop_size, seed=seed), 
                                                            num_parallel_calls=AUTOTUNE)
        
    
    if repeat:
        return ds.repeat()
    else:
        return ds


def create_fixed_dataset(year: int=None,
                         mode: str='validation',
                         batch_size: int=16,
                         downsample: bool=False,
                         fcst_shape: tuple=(20, 20, 9),
                         con_shape: tuple=(200, 200, 2),
                         out_shape: tuple=(200, 200, 1),
                         name: str=None,
                         folder: str=records_folder):
    assert year is not None or name is not None, "Must specify year or file name"
    if name is None:
        name = os.path.join(folder, f"{mode}{year}.tfrecords")
    else:
        name = os.path.join(folder, name)
    fl = glob.glob(name)
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=1)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    ds = ds.batch(batch_size)
    if downsample and return_dic:
        ds = ds.map(_dataset_downsampler)
    elif downsample and not return_dic:
        ds = ds.map(_dataset_downsampler_list)
    return ds


def _float_feature(list_of_floats: list):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def write_data(year_month_range: list,
               data_label: str,
               forecast_data_source: str, 
               observational_data_source: str,
               hours: list,
               num_class: int=4,
               normalise: bool=True,
               data_paths: dict=DATA_PATHS,
               constants: bool=True,
               latitude_range: list=None,
               longitude_range: list=None,
               debug: bool=False,
               data_config: dict=None,
               num_shards: int=1) -> str:
    """
    Function to write training data to TF records

    Args:
        year_month_range (list): List of date strings in YYYYMM format. The code will take all dates between the maximum and minimum year months (inclusive)
        data_label (str): Label to assign to data (e.g. train, validate)
        forecast_data_source (str): Source of forecast data (e.g. ifs)
        observational_data_source (str): Source of observational data (e.g. imerg)
        hours (list): List of hours to include
        num_class (int, optional): Number of classes to split data into. Defaults to 4.
        normalise (bool, optional): Whether or not to normalise the input data. Defaults to True.
        data_paths (dict, optional): Dict of paths to the data sources. Defaults to DATA_PATHS.
        constants (bool, optional): Whether or not to include constants. Defaults to True.
        latitude_range (list, optional): Latitude range to use. Defaults to None.
        longitude_range (list, optional): Longitude range to use. Defaults to None.
        debug (bool, optional): Debug mode. Defaults to False.
        config (dict, optional): Config dict. Defaults to None. If None then will read from default config location
        num_shards (int, optional): Number of shards to split each tfrecord into (to make sure records are not too big)

    Returns:
        str: Name of directory that records have been written to
    """

    from .data_generator import DataGenerator
    logger.info('Start of write data')
    logger.info(locals())     
      
    dates = date_range_from_year_month_range(year_month_range)
    start_date = dates[0]
    dates = [item for item in dates if file_exists(data_source=forecast_data_source, year=item.year,
                                                        month=item.month, day=item.day,
                                                        data_paths=data_paths)]
    if dates:
        if not dates[0] == start_date and not debug:
            # Means there likely isn't forecast data for the day before
            dates = dates[1:]
            
        records_folder = data_paths["TFRecords"]["tfrecords_path"]
        
        if not data_config:
            data_config = read_config.read_config(config_filename='data_config.yaml')
        
        class_bin_boundaries = data_config.get('class_bin_boundaries')
        if class_bin_boundaries is not None:
            print(f'Data will be bundled according to class bin boundaries provided: {class_bin_boundaries}')
            num_class = len(class_bin_boundaries) + 1

        input_image_width = data_config['input_image_width']
        num_samples_per_image = data_config['num_samples_per_image']

        if not os.path.isdir(records_folder):
            os.mkdirs(records_folder, exist_ok=True)
        
        #  Create directory that is hash of setup params, so that we know it's the right data later on
        hash_dir = os.path.join(records_folder, hash_dict(data_config))
        
        if not os.path.isdir(hash_dir):
            os.mkdir(hash_dir)
        
        print(f'Output folder will be {hash_dir}')
            
        # Write params in directory
        write_to_yaml(os.path.join(hash_dir, 'data_config.yaml'), data_config)
        
        with open(os.path.join(hash_dir, 'git_commit.txt'), 'w+') as ofh:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            ofh.write(sha)
        
        if debug:
            dates = dates[:1]

        for hour in hours:
            print('Hour = ', hour)
            start_time = time.time()
            dgc = DataGenerator(dates=[item.strftime('%Y%m%d') for item in dates],
                                forecast_data_source=forecast_data_source, 
                                observational_data_source=observational_data_source,
                                data_paths=data_paths,
                                batch_size=1,
                                shuffle=False,
                                constants=constants,
                                hour=hour,
                                normalise=normalise,
                                longitude_range=longitude_range,
                                latitude_range=latitude_range)
            print(f'Data generator initialization took {time.time() - start_time}')
            
            start_time = time.time()
            
            if dates:
                fle_hdles = {}
                for cl in range(num_class):
                    fle_hdles[cl] = []
                    for shard in range(num_shards):
                        flename = os.path.join(hash_dir, f"{data_label}_{hour}.{cl}.{shard}.tfrecords")
                        fle_hdles[cl].append(tf.io.TFRecordWriter(flename))
            
            print(f'File initialization took {time.time() - start_time}')
            print('starting fetching batches')
            for batch, date in tqdm(enumerate(dates), total=len(dates), position=0, leave=True):
                
                logger.debug(f"hour={hour}, batch={batch}")
                
                try:
                    
                    sample = dgc.__getitem__(batch)
                    (depth, width, height) = sample[1]['output'].shape
                
                    for k in range(depth):
                        
                        # NOTE: This kind of random cropping could also be done using Tensorflow during the data loading process. However, our experiments
                        # found this to be much slower than pre-cropping whilst creating the tensorflow records. The trade-off is that more hard drive space is required to
                        # store these examples, so the number of samples per image has to be chosen as high as possible to still get the benefit of the random cropping.
                        for ii in range(num_samples_per_image):
                            
                            idx = random.randint(0, width-input_image_width)
                            idy = random.randint(0, height-input_image_width)

                            observations = sample[1]['output'][k, 
                                                               idx:(idx+input_image_width), 
                                                               idy:(idy+input_image_width)].flatten()

                            forecast = sample[0]['lo_res_inputs'][k, 
                                                                  idx:(idx+input_image_width), 
                                                                  idy:(idy+input_image_width), :].flatten()
                            
                            const = sample[0]['hi_res_inputs'][k, 
                                                               idx:(idx+input_image_width), 
                                                               idy:(idy+input_image_width), :].flatten()
                            
                            if sample[0]['hi_res_inputs'][k, 
                                                               idx:(idx+input_image_width), 
                                                               idy:(idy+input_image_width), :].shape != (input_image_width, input_image_width, data_config['constant_fields']):
                                raise ValueError(f'Wrong constants dimensions for k == {k}, ii={ii}, date = {str(date)}, idx={idx}, idy={idy}')
       
                            # Check no Null values
                            if np.isnan(observations).any() or np.isnan(forecast).any() or np.isnan(const).any():
                                raise ValueError('Unexpected NaN values in data')
                            
                            # Check for empty data
                            if forecast.sum() == 0 or const.sum() == 0:
                                raise ValueError('one or more of arrays is all zeros')
                            
                            # Check hi res data has same dimensions
                                
                            feature = {
                                'generator_input': _float_feature(forecast),
                                'constants': _float_feature(const),
                                'generator_output': _float_feature(observations)
                            }

                            features = tf.train.Features(feature=feature)
                            example = tf.train.Example(features=features)
                            example_to_string = example.SerializeToString()
                            
                            # If provided, bin data according to bin boundaries (typically quartiles)
                            if class_bin_boundaries:
                                                        
                                threshold = 0.1
                                rainy_pixel_fraction = (denormalise(observations) > threshold).mean()
                                boundary_comparison = [rainy_pixel_fraction < cbb for cbb in class_bin_boundaries]
                                if any(boundary_comparison):
                                    clss = [rainy_pixel_fraction < cbb for cbb in class_bin_boundaries].index(True)
                                else:
                                    clss = len(class_bin_boundaries) 
                            else:
                                clss = random.choice(range(num_class))
                                
                            # Choose random shard
                            fh = random.choice(fle_hdles[clss])

                            fh.write(example_to_string)
                            
                except FileNotFoundError as e:
                    print(f"Error loading hour={hour}, date={date}")
            
            for cl, fhs in fle_hdles.items():
                for fh in fhs:
                    fh.close()
                    
        return hash_dir
    else:
        print('No dates found')

def write_train_test_data(*args, training_range,
                          validation_range=None,
                          test_range=None, **kwargs):
    
    
    write_data(training_range, *args,
               data_label='train', **kwargs)
    
    if validation_range:
        pass # Not using this at the moment
        # print('\n*** Writing validation data')
        # write_data(validation_range, *args,
        #        data_label='validation', **kwargs)
        
    if test_range:
        print('\n*** Writing test data')

        write_data(test_range, *args,
                   data_label='test', **kwargs)


def save_dataset(tfrecords_dataset, flename, max_batches=None):

    assert return_dic, "Only works with return_dic=True"
    flename = os.path.join(records_folder, flename)
    fle_hdle = tf.io.TFRecordWriter(flename)
    for ii, sample in enumerate(tfrecords_dataset):
        logger.info(ii)
        if max_batches is not None:
            if ii == max_batches:
                break
        for k in range(sample[1]['output'].shape[0]):
            feature = {
                'generator_input': _float_feature(sample[0]['lo_res_inputs'][k, ...].numpy().flatten()),
                'constants': _float_feature(sample[0]['hi_res_inputs'][k, ...].numpy().flatten()),
                'generator_output': _float_feature(sample[1]['output'][k, ...].numpy().flatten())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            fle_hdle.write(example_to_string)
    fle_hdle.close()
    return


if __name__ == '__main__':
    
    parser = ArgumentParser(description='Write data to tf records.')

    parser.add_argument('--fcst-hours', nargs='+', default=np.arange(24), type=int, 
                    help='Hour(s) to process (space separated)')
    parser.add_argument('--records-folder', type=str, default=None)
    parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()
    
    # Load relevant parameters from local config
    data_config = read_config.read_data_config()
    model_config = read_config.read_model_config()
    
    val_range = None
    eval_range = None
     
    if args.debug:
        training_range = ['201701']
    else:
        training_range = [str(item) for item in model_config.train.training_range]
        
        if hasattr(model_config.val, 'val_range'):
            val_range = [str(item) for item in model_config.val.val_range]
        
        if hasattr(model_config, 'eval'):
            if hasattr(model_config.eval, 'eval_range'):
                eval_range =  [str(item) for item in model_config.eval.eval_range]
    
    data_paths = DATA_PATHS
    if args.records_folder:
        data_paths['TFRecords']['tfrecords_path'] = args.records_folder
    
    write_train_test_data(training_range=training_range,
                          validation_range=val_range,
                          test_range=eval_range,
                            forecast_data_source=data_config.fcst_data_source, 
                            observational_data_source=data_config.obs_data_source,
                            hours=args.fcst_hours,
                            num_class=data_config.num_classes,
                            normalise=data_config.normalise,
                            data_paths=data_paths,
                            constants=data_config.load_constants,
                            latitude_range=np.arange(data_config.min_latitude, data_config.max_latitude, data_config.latitude_step_size),
                            longitude_range=np.arange(data_config.min_longitude, data_config.max_longitude, data_config.longitude_step_size))
