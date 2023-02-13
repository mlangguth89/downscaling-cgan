import argparse
import json
import os
import math
import git
from glob import glob
from pathlib import Path
import tensorflow as tf

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd

from dsrnngan import data
from dsrnngan import evaluation
from dsrnngan import plots
from dsrnngan import read_config
from dsrnngan import setupdata
from dsrnngan import setupmodel
from dsrnngan import train
from dsrnngan import utils

parser = argparse.ArgumentParser()
parser.add_argument('--records-folder', type=str, default=None,
                    help="Folder from which to gather the tensorflow records")
parser.add_argument('--no-train', dest='do_training', action='store_false',
                    help="Do NOT carry out training, only perform eval")
parser.add_argument('--restart', dest='restart', action='store_true',
                    help="Restart training from latest checkpoint")
group = parser.add_mutually_exclusive_group()
group.add_argument('--model-numbers', nargs='+', default=None,
                    help='Model number(s) to evaluate on (space separated)')
group.add_argument('--eval-full', dest='evalnum', action='store_const', const="full")
group.add_argument('--eval-short', dest='evalnum', action='store_const', const="short")
group.add_argument('--eval-blitz', dest='evalnum', action='store_const', const="blitz")
parser.add_argument('--evaluate', action='store_true',
                    help="Boolean: if true will run evaluation")
parser.add_argument('--plot-ranks', dest='plot_ranks', action='store_true',
                    help="Plot rank histograms")
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

def main(restart, do_training, evaluate, plot_ranks, num_images,
         noise_factor, ensemble_size, shuffle_eval=True, records_folder=None, evalnum=None, model_numbers=None, 
         seed=None, num_samples_override=None,
         val_start=None, val_end=None,
         ):
    
    if records_folder is None:
        
        config = read_config.read_config()
        data_paths = read_config.get_data_paths()
        
        records_folder = os.path.join(data_paths['TFRecords']['tfrecords_path'], utils.hash_dict(config))
        if not os.path.isdir(records_folder):
            raise ValueError('Data has not been prepared that matches this config')
    else:
        config = utils.load_yaml_file(os.path.join(records_folder, 'local_config.yaml'))
        data_paths = utils.load_yaml_file(os.path.join(records_folder, 'data_paths.yaml'))

    # TODO either change this to use a toml file or e.g. pydantic input validation

    architecture = config["MODEL"]["architecture"]
    padding = config["MODEL"]["padding"]
    log_folder = config['SETUP'].get('log_folder', False) or config["MODEL"]["log_folder"] 
    log_folder = os.path.join(log_folder, utils.hash_dict(config))
    mode = config["MODEL"].get("mode", False) or config['GENERAL']['mode']
    problem_type = config["MODEL"].get("problem_type", False) or config['GENERAL']['problem_type'] ## TODO: check if this is used anywhere
    downsample = config["MODEL"].get("downsample", False) or config['GENERAL']['downsample']
    
    downscaling_steps = config['DOWNSCALING']['steps']
    downscaling_factor = config['DOWNSCALING']['downscaling_factor']
    
    fcst_data_source=config['DATA']['fcst_data_source']
    obs_data_source=config['DATA']['obs_data_source']
    input_channels = config['DATA']['input_channels']
    constant_fields = config['DATA']['constant_fields']
    input_image_width = config['DATA']['input_image_width']
    output_image_width = downscaling_factor * input_image_width
    constants_image_width = input_image_width
    load_constants = config['DATA'].get('load_constants', True)    
    
    filters_gen = config["GENERATOR"]["filters_gen"]
    lr_gen = float(config["GENERATOR"]["learning_rate_gen"])
    noise_channels = config["GENERATOR"]["noise_channels"]
    latent_variables = config["GENERATOR"]["latent_variables"]
    
    filters_disc = config["DISCRIMINATOR"]["filters_disc"]
    lr_disc = config["DISCRIMINATOR"]["learning_rate_disc"]
    
    training_range = config['TRAIN']['training_range']
    training_weights = config["TRAIN"]["training_weights"]
    num_epochs = config["TRAIN"].get("num_epochs")
    num_samples = config['TRAIN'].get('num_samples') # leaving this in while we transition to using epochs
    steps_per_checkpoint = config["TRAIN"]["steps_per_checkpoint"]
    batch_size = config["TRAIN"]["batch_size"]
    kl_weight = config["TRAIN"]["kl_weight"]
    ensemble_size = ensemble_size or config["TRAIN"]["ensemble_size"]
    CL_type = config["TRAIN"]["CL_type"]
    content_loss_weight = config["TRAIN"]["content_loss_weight"]
    crop_size = config['TRAIN'].get('img_chunk_width')
    
    val_range = config['VAL'].get('val_range')
    if val_start:
        val_range[0] = val_start
    if val_end:
        val_range[1] = val_end

    val_size = config.get("VAL", {}).get("val_size")
    
    latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(config)
    
    # otherwise these are of type string, e.g. '1e-5'
    lr_gen = float(lr_gen)
    lr_disc = float(lr_disc)
    kl_weight = float(kl_weight)
    noise_factor = float(noise_factor)
    content_loss_weight = float(content_loss_weight)

    if mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")

    if ensemble_size is not None:
        if CL_type not in ["CRPS", "CRPS_phys", "ensmeanMSE", "ensmeanMSE_phys"]:
            raise ValueError("Content loss type is restricted to 'CRPS', 'CRPS_phys', 'ensmeanMSE', 'ensmeanMSE_phys'")

    if evaluate and val_range is None:
        raise ValueError('Must specify validation range when using --evaluate flag')
    
    assert math.prod(downscaling_steps) == downscaling_factor, "downscaling factor steps do not multiply to total downscaling factor!"
    
    # Calculate number of samples from epochs:
    if num_samples_override is not None:
        num_samples = num_samples_override
    else:
        if num_samples is None:
            if num_epochs is None:
                raise ValueError('Must specify either num_epochs or num_samples')
            num_data_points = len(utils.date_range_from_year_month_range(training_range)) * len(data.all_fcst_hours)
            num_samples = num_data_points * num_epochs
        
    num_checkpoints = int(num_samples/(steps_per_checkpoint * batch_size))
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
        # initialize GAN
        model = setupmodel.setup_model(
            mode=mode,
            architecture=architecture,
            downscaling_steps=downscaling_steps,
            input_channels=input_channels,
            constant_fields=constant_fields,
            latent_variables=latent_variables,
            filters_gen=filters_gen,
            filters_disc=filters_disc,
            noise_channels=noise_channels,
            padding=padding,
            lr_disc=lr_disc,
            lr_gen=lr_gen,
            kl_weight=kl_weight,
            ensemble_size=ensemble_size,
            CLtype=CL_type,
            content_loss_weight=content_loss_weight)
        
        fcst_shape=(input_image_width, input_image_width, input_channels)
        
        con_shape=(constants_image_width, constants_image_width, constant_fields)
        out_shape=(output_image_width, output_image_width, 1)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            training_range=training_range,
            validation_range=val_range,
            fcst_data_source=fcst_data_source,
            obs_data_source=obs_data_source,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            val_size=val_size,
            records_folder=records_folder,
            downsample=downsample,
            fcst_shape=fcst_shape,
            con_shape=con_shape,
            out_shape=out_shape,
            weights=training_weights,
            batch_size=batch_size,
            load_full_image=False,
            crop_size=crop_size,
            seed=seed)

        if restart: # load weights and run status

            model.load(model.filenames_from_root(model_weights_root))
            with open(os.path.join(log_folder, "run_status.json"), 'r') as f:
                run_status = json.load(f)
            training_samples = run_status["training_samples"]
            checkpoint = int(training_samples / (steps_per_checkpoint * batch_size)) + 1

            log_file = "{}/log.txt".format(log_folder)
            log = pd.read_csv(log_file)
            log_list = [log]

        else:  # initialize run status
            training_samples = 0

            log_file = os.path.join(log_folder, "log.txt")
            log_list = []

        plot_fname = os.path.join(log_folder, "progress.pdf")

        while (training_samples < num_samples):  # main training loop

            print("Checkpoint {}/{}".format(checkpoint, num_checkpoints))

            # train for some number of batches
            loss_log = train.train_model(model=model,
                                         mode=mode,
                                         batch_gen_train=batch_gen_train,
                                         batch_gen_valid=batch_gen_valid,
                                         noise_channels=noise_channels,
                                         latent_variables=latent_variables,
                                         checkpoint=checkpoint,
                                         steps_per_checkpoint=steps_per_checkpoint,
                                         plot_samples=val_size,
                                         plot_fn=plot_fname)

            training_samples += steps_per_checkpoint * batch_size
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

    else:
        print("Training skipped...")
        
    # model iterations to save full rank data to disk for during evaluations;
    # necessary for plot rank histograms. these are large files, so small
    # selection used to avoid storing gigabytes of data
    interval = steps_per_checkpoint * batch_size
    
    # Get latest model number
    latest_model_fp = sorted(glob(os.path.join(model_weights_root, '*.h5')))[-1]
    finalchkpt = int(int(latest_model_fp.split('/')[-1].split('.')[-2][-7:]) / interval)
    
    # last 4 checkpoints, or all checkpoints if < 4
    ranks_to_save = [(finalchkpt - ii)*interval for ii in range(3, -1, -1)] if finalchkpt >= 4 else [ii*interval for ii in range(1, finalchkpt+1)]

    if model_numbers:
        ranks_to_save = model_numbers
        pass
    elif evalnum == "blitz":
        model_numbers = ranks_to_save.copy()  # should not be modifying list in-place, but just in case!
    elif evalnum == "short":
        # last 1/3rd of checkpoints
        Neval = max(finalchkpt // 3, 1)
        model_numbers = [(finalchkpt - ii)*interval for ii in range((Neval-1), -1, -1)]
    elif evalnum == "full":
        model_numbers = np.arange(0, num_samples + 1, interval)[1:].tolist()

    # evaluate model performance
    if evaluate:
        evaluation.evaluate_multiple_checkpoints(mode=mode,
                                                 arch=architecture,
                                                 fcst_data_source=fcst_data_source,
                                                 obs_data_source=obs_data_source,
                                                 validation_range=val_range,
                                                 latitude_range=latitude_range,
                                                 longitude_range=longitude_range,
                                                 log_folder=log_folder,
                                                 weights_dir=model_weights_root,
                                                 records_folder=records_folder,
                                                 downsample=downsample,
                                                 noise_factor=noise_factor,
                                                 model_numbers=model_numbers,
                                                 ranks_to_save=ranks_to_save,
                                                 num_images=num_images,
                                                 filters_gen=filters_gen,
                                                 filters_disc=filters_disc,
                                                 input_channels=input_channels,
                                                 latent_variables=latent_variables,
                                                 noise_channels=noise_channels,
                                                 padding=padding,
                                                 ensemble_size=ensemble_size,
                                                 constant_fields=constant_fields,
                                                 data_paths=data_paths,
                                                 shuffle=shuffle_eval,
                                                 save_generated_samples=True)

    if plot_ranks:
        plots.plot_histograms(os.path.join(log_folder, f"n{num_images}_{'-'.join(val_range)}_e{ensemble_size}"), val_range, ranks=ranks_to_save, N_ranks=11)

if __name__ == "__main__":
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    if len(gpu_devices) == 0:
        print('GPU devices are not being seen')
    
    read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

    args = parser.parse_args()
    
    if args.model_numbers:
        model_numbers = [args.model_numbers] if not isinstance(args.model_numbers, list) else args.model_numbers
        model_numbers = [int(mn) for mn in model_numbers]
    else:
        model_numbers = None
    
    if args.evaluate and args.evalnum is None and args.model_numbers is None:
        raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', '--eval_blitz', or '--model-numbers X Y Z to specify length of evaluation")

    main(records_folder=args.records_folder, restart=args.restart, do_training=args.do_training, 
        evalnum=args.evalnum,
        evaluate=args.evaluate,
        plot_ranks=args.plot_ranks,
        noise_factor=args.noise_factor,
        num_samples_override=args.num_samples,
        num_images=args.num_images,
        model_numbers=model_numbers,
        val_start=args.val_ym_start,
        val_end=args.val_ym_end,
        ensemble_size=args.ensemble_size,
        shuffle_eval=not args.no_shuffle_eval)
