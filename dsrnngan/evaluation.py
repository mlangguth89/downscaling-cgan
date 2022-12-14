import gc
import os
import copy
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from properscoring import crps_ensemble
from tensorflow.python.keras.utils import generic_utils

from dsrnngan import data
from dsrnngan import read_config
from dsrnngan import setupdata
from dsrnngan import setupmodel
from dsrnngan.noise import NoiseGenerator
from dsrnngan.pooling import pool
from dsrnngan.rapsd import rapsd
from dsrnngan.scoring import rmse, mse, mae, calculate_pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

path = os.path.dirname(os.path.abspath(__file__))
ds_fac = read_config.read_config()['DOWNSCALING']["downscaling_factor"]

metrics = {'correlation': calculate_pearsonr, 'mae': mae, 'mse': mse,
           }
def setup_inputs(*,
                 mode,
                 arch,
                 records_folder,
                 fcst_data_source,
                 obs_data_source,
                 latitude_range,
                 longitude_range,
                 downscaling_steps,
                 validation_range,
                 downsample,
                 input_channels,
                 filters_gen,
                 filters_disc,
                 noise_channels,
                 latent_variables,
                 padding,
                 constant_fields,
                 data_paths):

    # initialise model
    model = setupmodel.setup_model(mode=mode,
                                   architecture=arch,
                                   downscaling_steps=downscaling_steps,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   padding=padding,
                                   constant_fields=constant_fields)

    gen = model.gen

    # always uses full-sized images
    print('Loading full sized image dataset')
    _, data_gen_valid = setupdata.setup_data(
        records_folder,
        fcst_data_source,
        obs_data_source,
        latitude_range=latitude_range,
        longitude_range=longitude_range,
        load_full_image=True,
        validation_range=validation_range,
        batch_size=1,
        downsample=downsample,
        data_paths=data_paths)
    
    return gen, data_gen_valid


def _init_VAEGAN(gen, data_gen, load_full_image, batch_size, latent_variables):
    if False:
        # this runs the model on one batch, which is what the internet says
        # but this doesn't actually seem to be necessary?!
        data_gen_iter = iter(data_gen)
        if load_full_image:
            inputs, outputs = next(data_gen_iter)
            cond = inputs['lo_res_inputs']
            const = inputs['hi_res_inputs']
        else:
            cond, const, _ = next(data_gen_iter)

        noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        mean, logvar = gen.encoder([cond, const])
        gen.decoder.predict([mean, logvar, noise_gen(), const])
    # even after running the model on one batch, this needs to be set(?!)
    gen.built = True
    return


def eval_one_chkpt(*,
                   mode,
                   gen,
                   fcst_data_source,
                   data_gen,
                   noise_channels,
                   latent_variables,
                   num_images,
                   latitude_range,
                   longitude_range,
                   ensemble_size,
                   noise_factor,
                   denormalise_data=True,
                   normalize_ranks=True,
                   show_progress=True,
                   n_quantiles=50):
    
    if num_images < 5:
        Warning('These scores are best performed with more images')
    
    truth_vals = []
    samples_gen_vals = []
    bias = []
    bias_fcst = []
    ranks = []
    lowress = []
    hiress = []
    crps_scores = {}
    mae_all = []
    mse_all = []
    corr_all = []
    emmse_all = []
    fcst_emmse_all = []
    ralsd_all = []
    ensemble_mean_correlation_all = []
    correlation_fcst_all = []

    tpidx = data.input_field_lookup[fcst_data_source.lower()].index('tp')
    
    batch_size = 1  # do one full-size image at a time

    if mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    if show_progress:
        # Initialize progbar
        progbar = generic_utils.Progbar(num_images,
                                        stateful_metrics=("CRPS", "EM-MSE"))

    CRPS_pooling_methods = ['no_pooling', 'max_4', 'max_16', 'avg_4', 'avg_16']
    rng = np.random.default_rng()

    data_idx = 0
    for kk in range(num_images):
        
        # load truth images
        # Bit of a hack here to deal with missing data
        try:
            inputs, outputs = data_gen[data_idx]
        except FileNotFoundError as e:
            print('Could not load file, attempting retries')
            for retry in range(5):
                data_idx += 1
                print(f'Attempting retry {retry} of 5')
                try:
                    inputs, outputs = data_gen[data_idx]
                    break
                except FileNotFoundError:
                    pass
                    
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        truth = outputs['output'][0, :, :]
        dates = inputs['dates']
        hours = inputs['hours']
        
        # Get observations at time of forecast
        loaddate, loadtime = data.get_ifs_forecast_time(dates[0].year, dates[0].month, dates[0].day, hours[0])
        dt = datetime(loaddate.year, loaddate.month, loaddate.day, int(loadtime))
        imerg_persisted_fcst = data.load_imerg(dt.date(), hour=dt.hour, latitude_vals=latitude_range, 
                                               longitude_vals=longitude_range, log_precip=not denormalise_data)
        
        assert imerg_persisted_fcst.shape == truth.shape, ValueError('Shape mismatch in iMERG persistent and truth')
        assert len(dates) == 1, ValueError('Currently must be run with a batch size of 1')
        assert len(dates) == len(hours), ValueError('This is strange, why are they different sizes?')
        
        if denormalise_data:
            truth = data.denormalise(truth)

        # generate predictions, depending on model type
        samples_gen = []
        if mode == "GAN":
            
            noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            for ii in range(ensemble_size):
                nn = noise_gen()
                sample_gen = gen.predict([cond, const, nn])
                samples_gen.append(sample_gen.astype("float32"))
                
        elif mode == "det":
            
            sample_gen = gen.predict([cond, const])
            samples_gen.append(sample_gen.astype("float32"))
            
        elif mode == 'VAEGAN':
            
            # call encoder once
            mean, logvar = gen.encoder([cond, const])
            noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            
            for ii in range(ensemble_size):
                nn = noise_gen()
                # generate ensemble of preds with decoder
                sample_gen = gen.decoder.predict([mean, logvar, nn, const])
                samples_gen.append(sample_gen.astype("float32"))

        # samples generated, now process them (e.g., undo log transform) and calculate MAE etc
        for ii in range(ensemble_size):
            
            sample_gen = samples_gen[ii][0, :, :, 0]
            
            # sample_gen shape should be [n, h, w, c] e.g. [1, 940, 940, 1]
            if denormalise_data:
                sample_gen = data.denormalise(sample_gen)

            # Calculate MAE, MSE for this sample
            mae_val = mae(truth, sample_gen)
            mse_val = mse(truth, sample_gen)
            corr = calculate_pearsonr(truth, sample_gen)

            mae_all.append(mae_val)
            mse_all.append(mse_val)
            corr_all.append(corr)

            if ii == 0:
                # reset on first ensemble member
                ensmean = np.zeros_like(sample_gen)
            ensmean += sample_gen

            samples_gen[ii] = sample_gen
       
        # Calculate Ensemble Mean MSE
        ensmean /= ensemble_size
        emmse = mse(truth, ensmean)
        emmse_all.append(emmse)
        
        # MSE to forecast
        fcst_emmse = mse(truth, cond[0, :, :, tpidx])
        fcst_emmse_all.append(fcst_emmse)
        
        # Correlation between random member of gan and truth (to compare with IFS)
        corr = calculate_pearsonr(truth, ensmean)
        ensemble_mean_correlation_all.append(corr)
        
        # Correlation between random member of gan and truth (to compare with IFS)
        corr = calculate_pearsonr(truth, ensmean)
        ensemble_mean_correlation_all.append(corr)
        
        corr_fcst = calculate_pearsonr(truth, cond[0, :, :, tpidx])
        correlation_fcst_all.append(corr_fcst)

        # Do all RALSD at once, to avoid re-calculating power spectrum of truth image
        ralsd = calculate_ralsd_rmse(truth, samples_gen)
        ralsd_all.append(ralsd.flatten())
        
        # Grid of biases
        bias.append(samples_gen[0] - truth)
        bias_fcst.append(cond[0, :, :, tpidx] - truth)

        # turn list of predictions into array, for CRPS/rank calculations
        samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]
        
        # Store these values for e.g. correlation on the grid
        truth_vals.append(truth)
        samples_gen_vals.append(samples_gen)

        ####################  CRPS calculation ##########################
        # calculate CRPS scores for different pooling methods
        for method in CRPS_pooling_methods:

            if method == 'no_pooling':
                truth_pooled = truth
                samples_gen_pooled = samples_gen
            else:
                truth_pooled = pool(truth, method)
                samples_gen_pooled = pool(samples_gen, method)
                
            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_truth_input = np.expand_dims(truth, 0)
            crps_gen_input = np.expand_dims(samples_gen, 0)
            crps_score = crps_ensemble(crps_truth_input, crps_gen_input).mean()
            del truth_pooled, samples_gen_pooled
            gc.collect()

            if method not in crps_scores:
                crps_scores[method] = []
            crps_scores[method].append(crps_score)
        
        
        # calculate ranks; only calculated without pooling

        # Add noise to truth and generated samples to make 0-handling fairer
        # NOTE truth and sample_gen are 'polluted' after this, hence we do this last
        truth += rng.random(size=truth.shape, dtype=np.float32)*noise_factor
        samples_gen += rng.random(size=samples_gen.shape, dtype=np.float32)*noise_factor

        truth_flat = truth.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
        samples_gen_ranks = samples_gen.reshape((-1, ensemble_size))  # unknown batch size/img dims, known number of samples
        rank = np.count_nonzero(truth_flat[:, None] >= samples_gen_ranks, axis=-1)  # mask array where truth > samples gen, count
        ranks.append(rank)
        
        # keep track of input and truth rainfall values, to facilitate further ranks processing
        cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx]).astype(np.float32), ds_fac, axis=-1), ds_fac, axis=-2)
        lowress.append(cond_exp.ravel())
        hiress.append(truth.astype(np.float32).ravel())
        del samples_gen_ranks, truth_flat
        gc.collect()

        if show_progress:
            emmse_so_far = np.sqrt(np.mean(emmse_all))
            crps_mean = np.mean(crps_scores['no_pooling'])
            losses = [("EM-MSE", emmse_so_far), ("CRPS", crps_mean)]
            progbar.add(1, values=losses)

    truth_array = np.stack(truth_vals, axis=0)
    samples_gen_array = np.stack(samples_gen_vals, axis=0)
    
    # Calculate quantiles for observed and truth
    quantile_boundaries = np.arange(0, 1, 1/10)
    truth_quantiles = np.quantile(truth_array, quantile_boundaries)
    sample_quantiles = np.quantile(samples_gen_array[:,:,0], quantile_boundaries)
    
    quantiles = {'truth': truth_quantiles,
                 'sample': sample_quantiles}

    # Pixelwise correlation
    pixelwise_corr = np.zeros_like(truth_array[0, :, :])
    for row in range(truth_array.shape[1]):
        for col in range(truth_array.shape[2]):
            pixelwise_corr[row, col] = calculate_pearsonr(truth_array[:, row, col], samples_gen_array[:, row, col, 0]).statistic

    ralsd_all = np.concatenate(ralsd_all)
    
    output = {}

    for method in crps_scores:
        output['CRPS_' + method] = np.asarray(crps_scores[method]).mean()
        
    output['mae'] = np.mean(mae_all)
    output['rmse'] = np.sqrt(np.mean(mse_all))
    output['emmse'] = np.sqrt(np.mean(emmse_all))
    output['emmse_fcst'] = np.sqrt(np.mean(fcst_emmse_all))
    output['ralsd'] = np.nanmean(ralsd_all)
    output['corr'] = np.mean(corr_all)
    output['corr_ensemble'] = np.mean(ensemble_mean_correlation_all)
    output['corr_fcst'] = np.mean(correlation_fcst_all)
    
    grid_output = {}
    grid_output['bias'] = np.mean(np.stack(bias, axis=-1), axis=-1)
    grid_output['bias_fcst'] = np.mean(np.stack(bias_fcst, axis=-1), axis=-1)
    grid_output['bias_std'] = np.std(np.stack(bias, axis=-1), axis=-1)
    grid_output['bias_median'] = np.median(np.stack(bias, axis=-1), axis=-1)
    grid_output['pixelwise_corr'] = pixelwise_corr

    ranks = np.concatenate(ranks)
    lowress = np.concatenate(lowress)
    hiress = np.concatenate(hiress)
    gc.collect()
    if normalize_ranks:
        ranks = (ranks / ensemble_size).astype(np.float32)
        gc.collect()
    rank_arrays = (ranks, lowress, hiress)

    return rank_arrays, output, grid_output, quantiles


def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks == 0) | (norm_ranks == 1)
    )
    op = float(op)/len(norm_ranks)
    return op


def log_line(log_fname, line):
    with open(log_fname, 'a') as f:
        print(line, file=f)


def evaluate_multiple_checkpoints(*,
                                  mode,
                                  arch,
                                  fcst_data_source,
                                  obs_data_source,
                                  latitude_range,
                                  longitude_range,
                                  validation_range,
                                  log_fname,
                                  weights_dir,
                                  records_folder,
                                  downsample,
                                  noise_factor,
                                  model_numbers,
                                  ranks_to_save,
                                  num_images,
                                  filters_gen,
                                  filters_disc,
                                  input_channels,
                                  latent_variables,
                                  noise_channels,
                                  padding,
                                  ensemble_size,
                                  constant_fields,
                                  data_paths):

    df_dict = read_config.read_config()['DOWNSCALING']

    gen, data_gen_valid = setup_inputs(mode=mode,
                                       arch=arch,
                                       records_folder=records_folder,
                                       fcst_data_source=fcst_data_source,
                                       obs_data_source=obs_data_source,
                                       latitude_range=latitude_range,
                                       longitude_range=longitude_range,
                                       downscaling_steps=df_dict["steps"],
                                       validation_range=validation_range,
                                       downsample=downsample,
                                       input_channels=input_channels,
                                       filters_gen=filters_gen,
                                       filters_disc=filters_disc,
                                       noise_channels=noise_channels,
                                       latent_variables=latent_variables,
                                       padding=padding,
                                       constant_fields=constant_fields,
                                       data_paths=data_paths)

    header = True

    for model_number in model_numbers:
        gen_weights_file = os.path.join(weights_dir, f"gen_weights-{model_number:07d}.h5")

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        print(gen_weights_file)
        if mode == "VAEGAN":
            _init_VAEGAN(gen, data_gen_valid, True, 1, latent_variables)
        gen.load_weights(gen_weights_file)
        rank_arrays, agg_metrics, grid_metrics, quantiles = eval_one_chkpt(mode=mode,
                                             gen=gen,
                                             data_gen=data_gen_valid,
                                             fcst_data_source=fcst_data_source,
                                             noise_channels=noise_channels,
                                             latent_variables=latent_variables,
                                             num_images=num_images,
                                             ensemble_size=ensemble_size,
                                             noise_factor=noise_factor,
                                             latitude_range=latitude_range,
                                             longitude_range=longitude_range)
        ranks, lowress, hiress = rank_arrays
        OP = rank_OP(ranks)
        
        # Create a dataframe of all the data (to ensure scores are recorded in the right column)
        df = pd.DataFrame.from_dict(dict(N=model_number, op=OP, **{k: [v] for k, v in agg_metrics.items()}))
        
        df.to_csv(log_fname, header=header, mode='a', float_format='%.6f', index=False)
        header = False

        # save one directory up from model weights, in same dir as logfile
        ranks_folder = os.path.dirname(log_fname)

        if model_number in ranks_to_save:
            fname = f"ranksnew-{'-'.join(validation_range)}_{model_number}.npz"
            np.savez_compressed(os.path.join(ranks_folder, fname), ranks=ranks, lowres=lowress, hires=hiress)
            
            # Save other gridwise metrics
            for k, v in grid_metrics.items():
                fname = f"{k}-{'-'.join(validation_range)}_{model_number}.npz"
                np.savez_compressed(os.path.join(ranks_folder, fname), k=v)
                
            # Save quantiles
            fname = f"quantiles-{'-'.join(validation_range)}_{model_number}.npz"
            np.savez_compressed(os.path.join(ranks_folder, fname), truth=quantiles['truth'], sample=quantiles['sample'])
            


def calculate_ralsd_rmse(truth, samples):
    # check 'batch size' is 1; can rewrite to handle batch size > 1 if necessary
    
    truth = np.copy(truth)
    
    if len(truth.shape) == 2:
        truth = np.expand_dims(truth, (0))
    
    expanded_samples = []
    for sample in samples:
        if len(sample.shape) == 2:
            sample = np.expand_dims(sample, (0))
        expanded_samples.append(sample)
    samples = expanded_samples
    
    assert truth.shape[0] == 1, 'Incorrect shape for truth'
    assert samples[0].shape[0] == 1

    # truth has shape 1 x W x H
    # samples is a list, each of shape 1 x W x H

    # avoid producing infinite or misleading values by not doing RALSD calc
    # for images that are mostly zeroes
    if truth.mean() < 0.002:
        return np.array([np.nan])
    # calculate RAPSD of truth once, not repeatedly!
    fft_freq_truth = rapsd(np.squeeze(truth, axis=0), fft_method=np.fft)
    dBtruth = 10 * np.log10(fft_freq_truth)

    ralsd_all = []
    for pred in samples:
        if pred.mean() < 0.002:
            ralsd_all.append(np.nan)
        else:
            fft_freq_pred = rapsd(np.squeeze(pred, axis=0), fft_method=np.fft)
            dBpred = 10 * np.log10(fft_freq_pred)
            rmse = np.sqrt(np.nanmean((dBtruth-dBpred)**2))
            ralsd_all.append(rmse)
    return np.array(ralsd_all)
