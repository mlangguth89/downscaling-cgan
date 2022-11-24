import argparse
import json
import os
import yaml
import math
from pathlib import Path
import tensorflow as tf

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd

from dsrnngan import evaluation
from dsrnngan import plots
from dsrnngan import read_config
from dsrnngan import setupdata
from dsrnngan import setupmodel
from dsrnngan import train
from dsrnngan import utils

parser = argparse.ArgumentParser()
parser.add_argument('--training-records-folder', type=str,
                    help="Folder from which to gather the tensorflow records")
parser.add_argument('--no-train', dest='do_training', action='store_false',
                    help="Do NOT carry out training, only perform eval")
parser.add_argument('--restart', dest='restart', action='store_true',
                    help="Restart training from latest checkpoint")
group = parser.add_mutually_exclusive_group()
group.add_argument('--eval-full', dest='evalnum', action='store_const', const="full")
group.add_argument('--eval-short', dest='evalnum', action='store_const', const="short")
group.add_argument('--eval-blitz', dest='evalnum', action='store_const', const="blitz")
parser.add_argument('--evaluate', action='store_true',
                    help="Boolean: if true will run evaluation")
parser.add_argument('--plot-ranks', dest='plot_ranks', action='store_true',
                    help="Plot rank histograms")

def main(root_records_folder, restart, do_training, evalnum, evaluate, plot_ranks,
         seed=None):
    
    config = read_config.read_config()
    records_folder = os.path.join(root_records_folder, utils.hash_dict(config))
    
    if not os.path.isdir(records_folder):
        raise ValueError('Data has not been prepared that matches this config')

    # TODO either change this to use a toml file or e.g. pydantic input validation
    mode = config["GENERAL"]["mode"]
    arch = config["MODEL"]["architecture"]
    padding = config["MODEL"]["padding"]
    log_folder = config["SETUP"]["log_folder"]
    problem_type = config["GENERAL"]["problem_type"]
    downsample = config["GENERAL"]["downsample"]
    fcst_data_source=config['DATA']['fcst_data_source']
    obs_data_source=config['DATA']['obs_data_source']
    input_channels = config['DATA']['input_channels']
    constant_fields = config['DATA']['constant_fields']
    fcst_image_width = config['DATA']['fcst_image_width']
    output_image_width = config['DATA']['output_image_width']
    constants_image_width = config['DATA']['constants_width']
    load_constants = config['DATA']['load_constants']
    downscaling_steps = config['DOWNSCALING']['steps']
    downscaling_factor = config['DOWNSCALING']['downscaling_factor']
    filters_gen = config["GENERATOR"]["filters_gen"]
    lr_gen = float(config["GENERATOR"]["learning_rate_gen"])
    noise_channels = config["GENERATOR"]["noise_channels"]
    latent_variables = config["GENERATOR"]["latent_variables"]
    filters_disc = config["DISCRIMINATOR"]["filters_disc"]
    lr_disc = config["DISCRIMINATOR"]["learning_rate_disc"]
    train_years = config["TRAIN"]["train_years"]
    training_weights = config["TRAIN"]["training_weights"]
    num_samples = config["TRAIN"]["num_samples"]
    steps_per_checkpoint = config["TRAIN"]["steps_per_checkpoint"]
    batch_size = config["TRAIN"]["batch_size"]
    kl_weight = config["TRAIN"]["kl_weight"]
    ensemble_size = config["TRAIN"]["ensemble_size"]
    CLtype = config["TRAIN"]["CL_type"]
    content_loss_weight = config["TRAIN"]["content_loss_weight"]
    val_years = config.get("VAL", {}).get("val_years")
    val_size = config.get("VAL", {}).get("val_size")
    num_images = config["EVAL"]["num_batches"]
    add_noise = config["EVAL"]["add_postprocessing_noise"]
    noise_factor = config["EVAL"]["postprocessing_noise_factor"]
    max_pooling = config["EVAL"]["max_pooling"]
    avg_pooling = config["EVAL"]["avg_pooling"]
    
    # otherwise these are of type string, e.g. '1e-5'
    lr_gen = float(lr_gen)
    lr_disc = float(lr_disc)
    kl_weight = float(kl_weight)
    noise_factor = float(noise_factor)
    content_loss_weight = float(content_loss_weight)

    if mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")

    if ensemble_size is not None:
        if CLtype not in ["CRPS", "CRPS_phys", "ensmeanMSE", "ensmeanMSE_phys"]:
            raise ValueError("Content loss type is restricted to 'CRPS', 'CRPS_phys', 'ensmeanMSE', 'ensmeanMSE_phys'")

    if evaluate and val_years is None:
        raise ValueError('Must specify at least one validation year when using --qual flag')
    
    assert math.prod(downscaling_steps) == downscaling_factor, "downscaling factor steps do not multiply to total downscaling factor!"
     
    num_checkpoints = int(num_samples/(steps_per_checkpoint * batch_size))
    checkpoint = 1

    # create log folder and model save/load subfolder if they don't exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    model_weights_root = os.path.join(log_folder, "models")
    Path(model_weights_root).mkdir(parents=True, exist_ok=True)

    # save setup parameters
    save_config = os.path.join(log_folder, 'setup_params.yaml')
    with open(save_config, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    if do_training:
        # initialize GAN
        model = setupmodel.setup_model(
            mode=mode,
            arch=arch,
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
            CLtype=CLtype,
            content_loss_weight=content_loss_weight)
        
        fcst_shape=(fcst_image_width, fcst_image_width, input_channels)
        
        con_shape=(constants_image_width, constants_image_width, constant_fields)
        out_shape=(output_image_width, output_image_width, 1)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            train_years=train_years,
            val_years=val_years,
            fcst_data_source=fcst_data_source,
            obs_data_source=obs_data_source,
            val_size=val_size,
            records_folder=records_folder,
            downsample=downsample,
            fcst_shape=fcst_shape,
            con_shape=con_shape,
            out_shape=out_shape,
            weights=training_weights,
            batch_size=batch_size,
            load_full_image=False,
            seed=seed)

        if restart: # load weights and run status

            model.load(model.filenames_from_root(model_weights_root))
            with open(log_folder + "-run_status.json", 'r') as f:
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

            data = {"training_samples": [training_samples]}
            for foo in loss_log:
                data[foo] = loss_log[foo]

            log_list.append(pd.DataFrame(data=data))
            log = pd.concat(log_list)
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each checkpoint
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)

    else:
        print("Training skipped...")

    eval_fname = os.path.join(log_folder, "eval_validation.txt")

    # model iterations to save full rank data to disk for during evaluations;
    # necessary for plot rank histograms. these are large files, so small
    # selection used to avoid storing gigabytes of data
    interval = steps_per_checkpoint * batch_size
    finalchkpt = num_samples // interval
    
    # last 4 checkpoints, or all checkpoints if < 4
    ranks_to_save = [(finalchkpt - ii)*interval for ii in range(3, -1, -1)] if finalchkpt >= 4 else [ii*interval for ii in range(1, finalchkpt+1)]

    if evalnum == "blitz":
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
                                                 arch=arch,
                                                 val_years=val_years,
                                                 log_fname=eval_fname,
                                                 weights_dir=model_weights_root,
                                                 downsample=downsample,
                                                 add_noise=add_noise,
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
                                                 ensemble_size=10)

    if plot_ranks:
        plots.plot_histograms(log_folder, val_years, ranks=ranks_to_save, N_ranks=11)

if __name__ == "__main__":
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    if len(gpu_devices) == 0:
        print('GPU devices are not being seen')
    
    read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

    args = parser.parse_args()

    if args.evalnum is None and (args.rank or args.qual):
        raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', or '--eval_blitz' to specify length of evaluation")

    setup_params = read_config.read_config()

    main(training_records_folder=args.training_records_folder, restart=args.restart, do_training=args.do_training, 
        evalnum=args.evalnum,
        evaluate=args.evaluate,
        plot_ranks=args.plot_ranks,
        setup_params=setup_params)
