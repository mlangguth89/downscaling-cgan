import argparse
import json
import os
import git
import wandb
import copy
import logging
from glob import glob
from types import SimpleNamespace
from tensorflow import config as tf_config

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd

from dsrnngan.data import data, setupdata
from dsrnngan.evaluation import evaluation
from dsrnngan.utils import read_config, utils
from dsrnngan.model import setupmodel, train
from unittest import mock


logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

model_config = read_config.read_model_config()
data_config = read_config.read_data_config()


data_config.min_longitude = -180
data_config.max_longitude = 180
data_config.longitude_step_size = 1
data_config.min_latitude = -90
data_config.max_latitude = 90
data_config.latitude_step_size = 1
num_constant_fields = 2
num_input_fields = 10
num_output_fields = 10 # Note: assuming all outputs can be produced simultaneously

latitude_range, longitude_range = utils.get_lat_lon_range_from_config(data_config=data_config)
data_config.input_image_height = len(latitude_range)
data_config.input_image_width = len(longitude_range)
    
model_config.image_chunk_width = None

input_image_shape =  (data_config.input_image_height, data_config.input_image_width, data_config.input_channels)
output_image_shape = (model_config.downscaling_factor * input_image_shape[0], model_config.downscaling_factor * input_image_shape[1])
constants_image_shape = (data_config.input_image_height, data_config.input_image_width, num_constant_fields)

inputs_array = np.random.rand(1, *input_image_shape)
constants_array = np.random.rand(1, *constants_image_shape)
# output_shape = np.random.rand(*output_image_shape)


parser = argparse.ArgumentParser()
parser.add_argument('--model-folder', type=str, default=None,
                    help="Folder in which previous model configs and training results have been stored.")
parser.add_argument('--num-images', type=int, default=20,
                    help="Number of images to evaluate on")
parser.add_argument('--eval-ensemble-size', type=int, default=1,
                    help="Size of ensemble to evaluate on")
parser.add_argument('--debug', action='store_true')                

# def blah():
#     t=1
#     return  [{'lo_res_inputs': np.random.rand(1,  *input_image_shape),
#                     'hi_res_inputs': np.random.rand(1, *constants_image_shape)},
#                     {'output': np.random.rand(*output_image_shape)}]
# @mock.patch.multiple('dsrnngan.data.data_generator.DataGenerator', 
#                     __init__=mock.MagicMock(return_value=None), 
#                     __getitem__=mock.MagicMock(return_value=blah()))
def main():
    
    
    
    # Create dummy model here
    model = setupmodel.setup_model(
            model_config=model_config,
            data_config=data_config)
    gen = model.gen
    # model_weights_root = os.path.join(args.model_folder, "models")
    # os.makedirs(model_weights_root, exist_ok=True)
    
    # model.save(model_weights_root)
    # TODO: wait time for average file load time.
    
    # Mock output
    samples_gen = evaluation.generate_gan_sample(gen=gen, 
                        cond=inputs_array, 
                        const=constants_array, 
                        noise_channels=model_config.generator.noise_channels, 
                        ensemble_size=1, 
                        batch_size=1)
    
    t=1
    

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    print('Checking GPU devices')
    gpu_devices = tf_config.list_physical_devices('GPU')
    
    print(gpu_devices)
    
    if len(gpu_devices) == 0:
        logger.debug('GPU devices are not being seen')
    logger.debug(gpu_devices)
    
    main()
    

    