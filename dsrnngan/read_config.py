import os
import sys
import tensorflow as tf
import numpy as np
import types
import math

from dsrnngan.utils import load_yaml_file

def read_config(config_filename=None):
    
    if config_filename is None:
        config_filename = 'local_config.yaml'
        
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_filename)
    try:
        localconfig = load_yaml_file(config_path)
    except FileNotFoundError as e:
        print(e)
        print(f"You must set {config_filename} in the main folder. Copy local_config-example.yaml and adjust appropriately.")
        sys.exit(1)       
    
    return localconfig


def get_data_paths():
    data_config_paths = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_paths.yaml')

    try:
        all_data_paths = load_yaml_file(data_config_paths)
    except FileNotFoundError as e:
        print(e)
        print("data_paths.yaml not found. Should exist in main folder")
        sys.exit(1)

    lc = read_config()['LOCAL']
    data_paths = all_data_paths[lc['data_paths']]
    return data_paths


def set_gpu_mode():
    lc = read_config()['LOCAL']
    if lc['use_gpu']:
        print('Setting up GPU')
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # remove environment variable (if it doesn't exist, nothing happens)
        # set memory allocation to incremental if desired
        if lc['gpu_mem_incr']:
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
        if lc['disable_tf32']:
            tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        # if use_gpu False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
def get_lat_lon_range_from_config(config=None):
    
    if config is None:
        config = read_config()
    
    if isinstance(config, str):
        config = load_yaml_file(config)
    
    min_latitude = config['DATA']['min_latitude']
    max_latitude = config['DATA']['max_latitude']
    latitude_step_size = config['DATA']['latitude_step_size']
    min_longitude = config['DATA']['min_longitude']
    max_longitude = config['DATA']['max_longitude']
    longitude_step_size = config['DATA']['longitude_step_size']
    
    latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
    longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)
    
    return latitude_range, longitude_range



def get_config_objects(config):
    
    local_config = types.SimpleNamespace(**config['LOCAL'])
    model_config = types.SimpleNamespace(**config['MODEL'])
    ds_config =  types.SimpleNamespace(**config['DOWNSCALING'])    
    data_config = types.SimpleNamespace(**config['DATA'])
    gen_config = types.SimpleNamespace(**config['GENERATOR'])
    dis_config = types.SimpleNamespace(**config['DISCRIMINATOR'])
    train_config = types.SimpleNamespace(**config['TRAIN'])
    val_config = types.SimpleNamespace(**config['VAL'])
    
    train_config.num_epochs = config["TRAIN"].get("num_epochs")
    train_config.num_samples = config['TRAIN'].get('num_samples') # leaving this in while we transition to using epochs    
    train_config.crop_size = config['TRAIN'].get('img_chunk_width')
    
    val_config.val_range = config['VAL'].get('val_range')
    val_config.val_size = config.get("VAL", {}).get("val_size")
    
    data_config.load_constants = config['DATA'].get('load_constants', True) 
    
    model_config.mode = config["MODEL"].get("mode", False) or config['GENERAL']['mode']
    model_config.downsample = config["MODEL"].get("downsample", False) or config.get('GENERAL', {}).get('downsample', False) 
    
    gen_config.learning_rate_gen = float(gen_config.learning_rate_gen)
    dis_config.learning_rate_disc = float(dis_config.learning_rate_disc)
    train_config.kl_weight = float(train_config.kl_weight)
    train_config.content_loss_weight = float(train_config.content_loss_weight)
    
    assert math.prod(ds_config.steps) == ds_config.downscaling_factor, "downscaling factor steps do not multiply to total downscaling factor!"
    
    return model_config, local_config, ds_config, data_config, gen_config, dis_config, train_config, val_config