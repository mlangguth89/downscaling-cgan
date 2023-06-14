import os
import sys
import copy
import tensorflow as tf
import numpy as np
import types
import math
from pathlib import Path

HOME = Path(__file__).parents[2]
CONFIG_FOLDER = HOME / 'config'

from dsrnngan.utils.utils import load_yaml_file

def read_config(config_filename: str=None, config_folder: str=CONFIG_FOLDER) -> dict:
    
    if config_filename is None:
        config_filename = 'local_config.yaml'
        
    config_path = os.path.join(config_folder, config_filename)
    try:
        config_dict = load_yaml_file(config_path)
    except FileNotFoundError as e:
        print(e)
        print(f"You must set {config_filename} in the config folder.")
        sys.exit(1)       
    
    return config_dict

def read_model_config(config_filename: str='model_config.yaml', config_folder: str=CONFIG_FOLDER) -> dict:
        
    model_config_dict = read_config(config_filename=config_filename, config_folder=config_folder)
    
    model_config = copy.deepcopy(model_config_dict)
    for k, v in model_config.items():
        if isinstance(v, dict):
            model_config[k] = types.SimpleNamespace(**v)
    model_config = types.SimpleNamespace(**model_config)
    
        
    model_config.train.num_epochs = model_config_dict["train"].get("num_epochs")
    model_config.train.num_samples = model_config_dict['train'].get('num_samples') # leaving this in while we transition to using epochs    
    model_config.train.crop_size = model_config_dict['train'].get('img_chunk_width')
    model_config.train.kl_weight = float(model_config.train.kl_weight)
    model_config.train.content_loss_weight = float(model_config.train.content_loss_weight)
    
    model_config.val.val_range = model_config_dict['val'].get('val_range')
    model_config.val.val_size = model_config_dict.get("val", {}).get("val_size")
    
    model_config.generator.learning_rate_gen = float(model_config.generator.learning_rate_gen)
    model_config.discriminator.learning_rate_disc = float(model_config.discriminator.learning_rate_disc)
    
    
    if model_config.mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")
    
    if model_config.train.CL_type not in ["CRPS", "CRPS_phys", "ensmeanMSE", "ensmeanMSE_phys"]:
        raise ValueError("Content loss type is restricted to 'CRPS', 'CRPS_phys', 'ensmeanMSE', 'ensmeanMSE_phys'")
    
    if model_config.train.num_samples is None:
        if model_config.train.num_epochs is None:
            raise ValueError('Must specify either num_epochs or num_samples')
        
    assert math.prod(model_config.downscaling_steps) == model_config.downscaling_factor, "downscaling factor steps do not multiply to total downscaling factor!"

    return model_config

def read_data_config(config_filename: str='data_config.yaml', config_folder: str=CONFIG_FOLDER) -> dict:
    
    data_config = read_config(config_filename=config_filename, config_folder=config_folder)
    
    data_config_ns = types.SimpleNamespace(**data_config)
    data_config_ns.load_constants = data_config.get('load_constants', True)
    data_config_ns.input_channels = len(data_config_ns.input_fields)
    
    return data_config_ns

def get_data_paths(config_folder: str=CONFIG_FOLDER):
    data_config = read_data_config(config_folder=config_folder)
    data_paths = data_config.paths[data_config.data_paths]
    return data_paths


def set_gpu_mode(model_config: dict=None):
    
    if model_config is None:
        model_config = read_model_config()
        
    if model_config.use_gpu:
        print('Setting up GPU', flush=True)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # remove environment variable (if it doesn't exist, nothing happens)
        
        # set memory allocation to incremental if desired
        # I think this is to ensure the GPU doesn't run out of memory
        if model_config.gpu_mem_incr:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            # if gpu_mem_incr False, do nothing
            pass
        if model_config.disable_tf32:
            tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        # if use_gpu False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
def get_lat_lon_range_from_config(data_config=None):
    
    if data_config is None:
        data_config = read_data_config()
    
    min_latitude = data_config.min_latitude
    max_latitude = data_config.max_latitude
    latitude_step_size = data_config.latitude_step_size
    min_longitude = data_config.min_longitude
    max_longitude = data_config.max_longitude
    longitude_step_size = data_config.longitude_step_size
    
    latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
    longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)
    
    return latitude_range, longitude_range