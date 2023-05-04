import argparse
import json
import os
import math
import git
import logging
from glob import glob
from pathlib import Path

from tensorflow import config as tf_config

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd



from dsrnngan import data
from dsrnngan import evaluation
from dsrnngan import read_config
from dsrnngan import setupdata
from dsrnngan import setupmodel
from dsrnngan import train
from dsrnngan import utils

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


parser = argparse.ArgumentParser()
parser.add_argument('--records-folder', type=str, default=None,
                    help="Folder from which to gather the tensorflow records")
parser.add_argument('--no-train', dest='do_training', action='store_false',
                    help="Do NOT carry out training, only perform eval")
parser.add_argument('--restart', dest='restart', action='store_true',
                    help="Restart training from latest checkpoint")
group = parser.add_mutually_exclusive_group()
group.add_argument('--eval-model-numbers', nargs='+', default=None,
                    help='Model number(s) to evaluate on (space separated)')
group.add_argument('--eval-full', dest='evalnum', action='store_const', const="full")
group.add_argument('--eval-short', dest='evalnum', action='store_const', const="short")
group.add_argument('--eval-blitz', dest='evalnum', action='store_const', const="blitz")
parser.add_argument('--num-samples', type=int,
                    help="Override of num samples")
parser.add_argument('--num-images', type=int, default=20,
                    help="Number of images to evaluate on")
parser.add_argument('--ensemble-size', type=int, default=None,
                    help="Size of ensemble to evaluate on")
parser.add_argument('--noise-factor', type=float, default=1e-3,
                    help="Multiplicative noise factor for rank histogram")
parser.add_argument('--val-ym-start', type=str,
                    help='Validation start in YYYYMM format (defaults to range specified in config)')
parser.add_argument('--val-ym-end', type=str,
                    help='Validation start in YYYYMM format (defaults to the range specified in the config)')
parser.add_argument('--no-shuffle-eval', action='store_true', 
                    help='Boolean, will turn off shuffling at evaluation.')
parser.add_argument('--save-generated-samples', action='store_true',
                    help='Flag to trigger saving of the evaluation arrays')
parser.add_argument('--training-weights', default=None, nargs=4,help='Weighting of classes',
                    type=float
                    )
parser.add_argument('--output-suffix', default=None, 
                    help='Suffix to append to model folder. If none then model folder has same name as TF records folder used as input.')

def main(restart: bool, do_training: bool, num_images: int,
         noise_factor: float, ensemble_size: int, shuffle_eval: bool=True, records_folder: str=None, evalnum: str=None, eval_model_numbers: list=None, 
         seed: int=None, num_samples_override: int=None,
         val_start: str=None, val_end: str=None, save_generated_samples: bool=False, training_weights: list=None, debug: bool=False,
         output_suffix: str=None, log_folder=None
         ):
    """ Function for training and evaluating a cGAN, from a dataset of tf records

    Args:
        restart (bool): If True then will restart training from latest checkpoint       
        do_training (bool): If True then will run training on model 
        num_images (int): Number of images to evaluate on
        noise_factor (float): Noise factor to add to CRPS evaluation
        ensemble_size (int): Size of ensemble to evaluate on
        shuffle_eval (bool, optional): If False then no shuffling of data before evaluation (for cases where you want to assess on a contiguous range of dates). Defaults to True.
        records_folder (str, optional): Location of tfrecords. Defaults to None. If None then will try to infer folder from tehash of the config.
        evalnum (str, optional): String representing type of evaluation: blitz, full, short. Defaults to None.
        eval_model_numbers (list, optional): List of models numbers to evaluate. Defaults to None.
        seed (int, optional): Optional random seed. Defaults to None.
        num_samples_override (int, optional): Total of samples to use in traininig. Defaults to None. If None then will read from config
        val_start (str, optional): Validation start in YYYYMM format. Defaults to None.
        val_end (str, optional): Validation end in YYYYMM format. Defaults to None.
        save_generated_samples (bool, optional): If True then generated samples are stored for further analysis. Defaults to False.
        training_weights (list, optional): List of floats, fraction of samples to take from each class in the training data. Defaults to None. If None then is read from the config
        debug (bool, optional): Whether or not to use in debug mode (stops training after first checkpoint). Defaults to False.
        output_suffix (str, optional): Suffix to append to output folder name. Defaults to None.
        log_folder (str, optional): root folder to store results in. Defaults to None. If None then will be taken from config.

    """
    print('Reading config')
    if records_folder is None:
        
        config = read_config.read_config()
        data_paths = read_config.get_data_paths()
        
        records_folder = os.path.join(data_paths['TFRecords']['tfrecords_path'], utils.hash_dict(config))
        if not os.path.isdir(records_folder):
            raise ValueError('Data has not been prepared that matches this config')
    else:
        config = utils.load_yaml_file(os.path.join(records_folder, 'local_config.yaml'))
        data_paths = utils.load_yaml_file(os.path.join(records_folder, 'data_paths.yaml'))

    log_folder = log_folder or config.get('SETUP', {}).get('log_folder', False) or config["MODEL"]["log_folder"] 
    log_folder = os.path.join(log_folder, records_folder.split('/')[-1])
    
    if output_suffix:
        output_suffix = '_' + output_suffix if not output_suffix.startswith('_') else output_suffix
        log_folder = log_folder + output_suffix
    
    model_config, _, ds_config, data_config, gen_config, dis_config, train_config, val_config = read_config.get_config_objects(config)
    
    input_image_shape = (data_config.input_image_width, data_config.input_image_width, data_config.input_channels)
    output_image_shape = (ds_config.downscaling_factor * input_image_shape[0], ds_config.downscaling_factor * input_image_shape[1], 1)
    constants_image_shape = (data_config.input_image_width, data_config.input_image_width, data_config.constant_fields)
    
    if training_weights is None:
        training_weights = train_config.training_weights

    ensemble_size = ensemble_size or train_config.ensemble_size  

    if val_start:
        val_config.val_range[0] = val_start
    if val_end:
        val_config.val_range[1] = val_end
    
    latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(config)

    noise_factor = float(noise_factor)

    if model_config.mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")

    if ensemble_size is not None:
        if train_config.CL_type not in ["CRPS", "CRPS_phys", "ensmeanMSE", "ensmeanMSE_phys"]:
            raise ValueError("Content loss type is restricted to 'CRPS', 'CRPS_phys', 'ensmeanMSE', 'ensmeanMSE_phys'")

    evaluate = (eval_model_numbers or evalnum)
    
    if evaluate and val_config.val_range is None:
        raise ValueError('Must specify validation range when using --evaluate flag')
    
    # Calculate number of samples from epochs:
    if num_samples_override is not None:
        train_config.num_samples = num_samples_override
    else:
        if train_config.num_samples is None:
            if train_config.num_epochs is None:
                raise ValueError('Must specify either num_epochs or num_samples')
            num_data_points = len(utils.date_range_from_year_month_range(train_config.training_range)) * len(data.all_fcst_hours)
            train_config.num_samples = num_data_points * train_config.num_epochs
        
    num_checkpoints = int(train_config.num_samples/(train_config.steps_per_checkpoint * train_config.batch_size))
    checkpoint = 1
    
    # Get Git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # create root log folder / log folder / models subfolder if they don't exist
    # Path(root_log_folder).mkdir(parents=True, exist_ok=True)
    # log_folder = os.path.join(root_log_folder, sha[:8])
    
    model_weights_root = os.path.join(log_folder, "models")
    Path(model_weights_root).mkdir(parents=True, exist_ok=True)

    # save setup parameters
    utils.write_to_yaml(os.path.join(log_folder, 'setup_params.yaml'), config)
        
    with open(os.path.join(log_folder, 'git_commit.txt'), 'w+') as ofh:
        ofh.write(sha)

    if do_training:
        logger.debug('Starting training')
        # initialize GAN
        model = setupmodel.setup_model(
            mode=model_config.mode,
            architecture=model_config.architecture,
            downscaling_steps=ds_config.steps,
            input_channels=data_config.input_channels,
            constant_fields=data_config.constant_fields,
            latent_variables=gen_config.latent_variables,
            filters_gen=gen_config.filters_gen,
            filters_disc=dis_config.filters_disc,
            noise_channels=gen_config.noise_channels,
            padding=model_config.padding,
            lr_disc=dis_config.learning_rate_disc,
            lr_gen=gen_config.learning_rate_gen,
            kl_weight=train_config.kl_weight,
            ensemble_size=ensemble_size,
            CLtype=train_config.CL_type,
            content_loss_weight=train_config.content_loss_weight)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            training_range=train_config.training_range,
            validation_range=val_config.val_range,
            fcst_data_source=data_config.fcst_data_source,
            obs_data_source=data_config.obs_data_source,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            val_size=val_config.val_size,
            records_folder=records_folder,
            downsample=model_config.downsample,
            fcst_shape=input_image_shape,
            con_shape=constants_image_shape,
            out_shape=output_image_shape,
            weights=training_weights,
            crop_size=train_config.crop_size,
            batch_size=train_config.batch_size,
            load_full_image=False,
            seed=seed)

        if restart: # load weights and run status
            try:
                model.load(model.filenames_from_root(model_weights_root))
                with open(os.path.join(log_folder, "run_status.json"), 'r') as f:
                    run_status = json.load(f)
                training_samples = run_status["training_samples"]
                checkpoint = int(training_samples / (train_config.steps_per_checkpoint * train_config.batch_size)) + 1

                log_file = "{}/log.txt".format(log_folder)
                log = pd.read_csv(log_file)
                log_list = [log]
            except FileNotFoundError:
                # Catch case where folder exists but no model saved yet
                training_samples = 0

                log_file = os.path.join(log_folder, "log.txt")
                log_list = []

        else:  # initialize run status
            training_samples = 0

            log_file = os.path.join(log_folder, "log.txt")
            log_list = []

        plot_fname = os.path.join(log_folder, "progress.pdf")

        while (training_samples < train_config.num_samples):  # main training loop

            logger.debug("Checkpoint {}/{}".format(checkpoint, num_checkpoints))

            # train for some number of batches
            loss_log = train.train_model(model=model,
                                         mode=model_config.mode,
                                         batch_gen_train=batch_gen_train,
                                         batch_gen_valid=batch_gen_valid,
                                         noise_channels=gen_config.noise_channels,
                                         latent_variables=gen_config.latent_variables,
                                         checkpoint=checkpoint,
                                         steps_per_checkpoint=train_config.steps_per_checkpoint,
                                         plot_samples=val_config.val_size,
                                         plot_fn=plot_fname)

            training_samples += train_config.steps_per_checkpoint * train_config.batch_size
            checkpoint += 1


            # save results
            model.save(model_weights_root)
            run_status = {
                "training_samples": training_samples,
            }
            with open(os.path.join(log_folder, "run_status.json"), 'w') as f:
                json.dump(run_status, f)

            log_data = {"training_samples": [training_samples]}
            for foo in loss_log:
                log_data[foo] = loss_log[foo]

            log_list.append(pd.DataFrame(data=log_data))
            log = pd.concat(log_list)
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each checkpoint
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)
            
            if debug:
                break

    else:
        logger.debug("Training skipped...")
        
    # model iterations to save full rank data to disk for during evaluations;
    # necessary for plot rank histograms. these are large files, so small
    # selection used to avoid storing gigabytes of data
    interval = train_config.steps_per_checkpoint * train_config.batch_size
    
    # Get latest model number
    latest_model_fp = sorted(glob(os.path.join(model_weights_root, '*.h5')))[-1]
    finalchkpt = int(int(latest_model_fp.split('/')[-1].split('.')[-2][-7:]) / interval)
    
    # last 4 checkpoints, or all checkpoints if < 4
    ranks_to_save = [(finalchkpt - ii)*interval for ii in range(3, -1, -1)] if finalchkpt >= 4 else [ii*interval for ii in range(1, finalchkpt+1)]

    if eval_model_numbers:
        ranks_to_save = eval_model_numbers
        pass
    elif evalnum == "blitz":
        eval_model_numbers = ranks_to_save.copy()  # should not be modifying list in-place, but just in case!
    elif evalnum == "short":
        # last 1/3rd of checkpoints
        Neval = max(finalchkpt // 3, 1)
        eval_model_numbers = [(finalchkpt - ii)*interval for ii in range((Neval-1), -1, -1)]
    elif evalnum == "full":
        eval_model_numbers = np.arange(0, train_config.num_samples + 1, interval)[1:].tolist()

    # evaluate model performance
    if evaluate:
        logger.debug('Performing evaluation')
        evaluation.evaluate_multiple_checkpoints(mode=model_config.mode,
                                                 arch=model_config.architecture,
                                                 fcst_data_source=data_config.fcst_data_source,
                                                 obs_data_source=data_config.obs_data_source,
                                                 validation_range=val_config.val_range,
                                                 latitude_range=latitude_range,
                                                 longitude_range=longitude_range,
                                                 log_folder=log_folder,
                                                 weights_dir=model_weights_root,
                                                 records_folder=records_folder,
                                                 downsample=model_config.downsample,
                                                 noise_factor=noise_factor,
                                                 model_numbers=eval_model_numbers,
                                                 ranks_to_save=ranks_to_save,
                                                 num_images=num_images,
                                                 filters_gen=gen_config.filters_gen,
                                                 filters_disc=dis_config.filters_disc,
                                                 input_channels=data_config.input_channels,
                                                 latent_variables=gen_config.latent_variables,
                                                 noise_channels=gen_config.noise_channels,
                                                 padding=model_config.padding,
                                                 ensemble_size=ensemble_size,
                                                 constant_fields=data_config.constant_fields,
                                                 data_paths=data_paths,
                                                 shuffle=shuffle_eval,
                                                 save_generated_samples=save_generated_samples)
    
    return log_folder


if __name__ == "__main__":
    
    print('Checking GPU devices')
    gpu_devices = tf_config.list_physical_devices('GPU')
    
    print(gpu_devices)
    
    if len(gpu_devices) == 0:
        logger.debug('GPU devices are not being seen')
    logger.debug(gpu_devices)
    
    print('Setting GPU mode')
    read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
    
    args = parser.parse_args()
    
    if args.eval_model_numbers:
        eval_model_numbers = [args.eval_model_numbers] if not isinstance(args.eval_model_numbers, list) else args.eval_model_numbers
        eval_model_numbers = [int(mn) for mn in eval_model_numbers]
    else:
        eval_model_numbers = None

    main(records_folder=args.records_folder, restart=args.restart, do_training=args.do_training, 
        evalnum=args.evalnum,
        noise_factor=args.noise_factor,
        num_samples_override=args.num_samples,
        num_images=args.num_images,
        eval_model_numbers=eval_model_numbers,
        val_start=args.val_ym_start,
        val_end=args.val_ym_end,
        ensemble_size=args.ensemble_size,
        shuffle_eval=not args.no_shuffle_eval,
        save_generated_samples=args.save_generated_samples,
        training_weights=args.training_weights,
        output_suffix=args.output_suffix)
