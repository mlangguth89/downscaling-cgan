import sys, os
import unittest
import tempfile
import yaml
from pathlib import Path
from glob import glob

HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))



# class TestEvaluation(unittest.TestCase):
    
#     def test_quality_metrics_by_time(self):
        
#         # This works currently only after the test_main bits have been run
#         log_folder = str(HOME / 'system_tests' / 'data' / 'tmp')
#         quality_metrics_by_time(mode="GAN",
#                                 arch='forceconv',
#                                 fcst_data_source='era5',
#                                 obs_data_source='imerg',
#                                 load_constants=False,
#                                 val_years=2019,
#                                 log_fname=os.path.join(log_folder, 'qual.txt'),
#                                 weights_dir=os.path.join(log_folder, 'models'),
#                                 downsample=False,
#                                 model_numbers=[4],
#                                 batch_size=1,  # do inference 1 at a time, in case of memory issues
#                                 num_batches=2,
#                                 filters_gen=2,
#                                 filters_disc=2,
#                                 input_channels=5,
#                                 constant_fields=1,
#                                 latent_variables=1,
#                                 noise_channels=4,
#                                 rank_samples=2,
#                                 padding='reflect')
        
#     def test_rank_metrics_by_time(self):
        
#         log_folder = str(HOME / 'system_tests' / 'data' / 'tmp')
#         rank_metrics_by_time(mode="GAN",
#                                 arch='forceconv',
#                                 fcst_data_source='era5',
#                                 obs_data_source='imerg',
#                                 val_years=2019,
#                                 constant_fields=1,
#                                 load_constants=False,
#                                 log_fname=os.path.join(log_folder, 'rank.txt'),
#                                 weights_dir=os.path.join(log_folder, 'models'),
#                                 downsample=False,
#                                 weights=[0.4, 0.3, 0.2, 0.1],
#                                 add_noise=True,
#                                 noise_factor=0.001,
#                                 model_numbers=[4],
#                                 ranks_to_save=[4],
#                                 batch_size=1,  # ditto
#                                 num_batches=2,
#                                 filters_gen=2,
#                                 filters_disc=2,
#                                 input_channels=5,
#                                 latent_variables=1,
#                                 noise_channels=4,
#                                 padding='reflect',
#                                 rank_samples=10,
#                                 max_pooling=True,
#                                 avg_pooling=True)