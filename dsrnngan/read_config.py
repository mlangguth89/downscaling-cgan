import os
import sys
import tensorflow as tf

from dsrnngan.utils import load_yaml_file

def read_config(config_filename='local_config.yaml'):
    
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
