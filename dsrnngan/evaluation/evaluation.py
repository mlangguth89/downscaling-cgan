import gc
import os
import warnings
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from properscoring import crps_ensemble
from tensorflow.python.keras.utils import generic_utils
from datetime import datetime
from typing import Iterable

from dsrnngan.data.data_generator import DataGenerator
from dsrnngan.data import data
from dsrnngan.utils import read_config, utils
from dsrnngan.data import setupdata
from dsrnngan.model import setupmodel
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.model.pooling import pool
from dsrnngan.evaluation.rapsd import rapsd
from dsrnngan.evaluation.scoring import rmse, mse, mae, calculate_pearsonr, fss
from dsrnngan.model import gan

warnings.filterwarnings("ignore", category=RuntimeWarning)


path = os.path.dirname(os.path.abspath(__file__))
model_config = read_config.read_model_config()
ds_fac = model_config.downscaling_factor

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
                 hour,
                 downsample,
                 input_channels,
                 filters_gen,
                 filters_disc,
                 noise_channels,
                 latent_variables,
                 padding,
                 constant_fields,
                 data_paths,
                 shuffle,
                 fcst_fields=None):

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
        fcst_data_source,
        obs_data_source,
        fcst_fields=fcst_fields,
        records_folder=records_folder,
        latitude_range=latitude_range,
        longitude_range=longitude_range,
        load_full_image=True,
        validation_range=validation_range,
        batch_size=1,
        downsample=downsample,
        data_paths=data_paths,
        shuffle=shuffle,
        hour=hour)
    
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

def generate_gan_sample(gen: gan.WGANGP, 
                        cond: np.ndarray, 
                        const: np.ndarray, 
                        noise_channels: int, 
                        ensemble_size: int=1, 
                        batch_size: int=1, 
                        seed: int=None):
    
    samples_gen = []
    noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
    noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size, random_seed=seed)
    for ii in range(ensemble_size):
        nn = noise_gen()
        sample_gen = gen.predict([cond, const, nn])
        samples_gen.append(sample_gen.astype("float32"))
        
    return samples_gen

def create_single_sample(*,
                   data_idx: int,
                   mode: str,
                   batch_size: int,
                   gen: gan.WGANGP,
                   fcst_data_source: str,
                   data_gen: DataGenerator,
                   noise_channels: int,
                   latent_variables: int,
                   latitude_range: Iterable,
                   longitude_range: Iterable,
                   ensemble_size: int,
                   denormalise_data: bool=True,
                   seed: int=None):
    
    tpidx = data.input_fields.index('tp')
    
    batch_size = 1  # do one full-size image at a time

    if mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    # load truth images
    inputs, outputs = data_gen[data_idx]
      
    cond = inputs['lo_res_inputs']
    fcst = cond[0, :, :, tpidx]
    const = inputs['hi_res_inputs']
    obs = outputs['output'][0, :, :]
    date = inputs['dates']
    hour = inputs['hours']           
    
    # Get observations 24hrs before
    imerg_persisted_fcst = data.load_imerg(date[0] - timedelta(days=1), hour=hour[0], latitude_vals=latitude_range, 
                                            longitude_vals=longitude_range, log_precip=not denormalise_data)
    
    assert imerg_persisted_fcst.shape == obs.shape, ValueError('Shape mismatch in iMERG persistent and truth')
    assert len(date) == 1, ValueError('Currently must be run with a batch size of 1')
    assert len(date) == len(hour), ValueError('This is strange, why are they different sizes?')
        
    if denormalise_data:
        obs = data.denormalise(obs)
        fcst = data.denormalise(fcst)
            
    if mode == "GAN":
        
        samples_gen = generate_gan_sample(gen, cond, const, noise_channels, ensemble_size, batch_size, seed)
            
    elif mode == "det":
        samples_gen = []
        sample_gen = gen.predict([cond, const])
        samples_gen.append(sample_gen.astype("float32"))
        
    elif mode == 'VAEGAN':
        samples_gen = []
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

        samples_gen[ii] = sample_gen

    return obs, samples_gen, fcst, imerg_persisted_fcst, cond, const, date, hour

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
                   batch_size: int=1):
    
    if num_images < 5:
        Warning('These scores are best performed with more images')
    
    truth_vals = []
    samples_gen_vals = []
    fcst_vals = []
    persisted_fcst_vals = []
    dates = []
    hours = []
    cond_vals = []
    const_vals = []
    
    ranks = []
    lowress = []
    hiress = []
    crps_scores = {}
    mse_all = []
    emmse_all = []
    fcst_mse_all = []
    ralsd_rmse_all = []
    corr_all = []
    correlation_fcst_all = []

    tpidx = data.input_fields.index('tp')
    
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
    for kk in tqdm(range(num_images)):
        
        try:
            obs, samples_gen, fcst, imerg_persisted_fcst, cond, const, date, hour = create_single_sample(mode=mode,
                    data_idx=data_idx,
                    batch_size=batch_size,
                    gen=gen,
                    fcst_data_source=fcst_data_source,
                    data_gen=data_gen,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    latitude_range=latitude_range,
                    longitude_range=longitude_range,
                    ensemble_size=ensemble_size,
                    denormalise_data=denormalise_data)
        except IndexError:
            # Run out of samples
            break 
               
        except FileNotFoundError:
            print('Could not load file, attempting retries')
            success = False
            for retry in range(5):
                data_idx += 1
                print(f'Attempting retry {retry} of 5')
                try:
                    obs, samples_gen, fcst, imerg_persisted_fcst, cond, const, date, hour = create_single_sample(mode=mode,
                        data_idx=data_idx,
                        batch_size=batch_size,
                        gen=gen,
                        fcst_data_source=fcst_data_source,
                        data_gen=data_gen,
                        noise_channels=noise_channels,
                        latent_variables=latent_variables,
                        latitude_range=latitude_range,
                        longitude_range=longitude_range,
                        ensemble_size=ensemble_size,
                        denormalise_data=denormalise_data)
                    success = True
                    break
                except FileNotFoundError:
                    pass
            if not success:
                raise FileNotFoundError
        
        dates.append(date)
        hours.append(hour)
        cond_vals.append(cond)
        const_vals.append(const)

        # Do all RALSD at once, to avoid re-calculating power spectrum of truth image
    
        ralsd_rmse = calculate_ralsd_rmse(obs, samples_gen)
        ralsd_rmse_all.append(ralsd_rmse.flatten())
               
        # turn list of predictions into array, for CRPS/rank calculations
        samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]
        
        mse_all.append(mse(samples_gen[:,:,0], obs))
        emmse_all.append(mse(np.mean(samples_gen, axis=-1), obs))
        fcst_mse_all.append(mse(obs, fcst))

        corr_all.append(calculate_pearsonr(obs, samples_gen[:, :, 0]))
        correlation_fcst_all.append(calculate_pearsonr(obs, fcst))
        
        # Store these values for e.g. correlation on the grid
        truth_vals.append(obs)
        samples_gen_vals.append(samples_gen)
        fcst_vals.append(fcst)
        persisted_fcst_vals.append(imerg_persisted_fcst)
        
        ####################  CRPS calculation ##########################
        # calculate CRPS scores for different pooling methods
        for method in CRPS_pooling_methods:

            if method == 'no_pooling':
                truth_pooled = obs
                samples_gen_pooled = samples_gen
            else:
                truth_pooled = pool(obs, method)
                samples_gen_pooled = pool(samples_gen, method)
                
            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_truth_input = np.expand_dims(obs, 0)
            crps_gen_input = np.expand_dims(samples_gen, 0)
            crps_score_grid = crps_ensemble(crps_truth_input, crps_gen_input)
            crps_score = crps_score_grid.mean()
            del truth_pooled, samples_gen_pooled
            gc.collect()

            if method not in crps_scores:
                crps_scores[method] = []

            crps_scores[method].append(crps_score)
        
        
        # calculate ranks; only calculated without pooling

        # Add noise to truth and generated samples to make 0-handling fairer
        # NOTE truth and sample_gen are 'polluted' after this, hence we do this last
        obs += rng.random(size=obs.shape, dtype=np.float32)*noise_factor
        samples_gen += rng.random(size=samples_gen.shape, dtype=np.float32)*noise_factor

        truth_flat = obs.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
        samples_gen_ranks = samples_gen.reshape((-1, ensemble_size))  # unknown batch size/img dims, known number of samples
        rank = np.count_nonzero(truth_flat[:, None] >= samples_gen_ranks, axis=-1)  # mask array where truth > samples gen, count
        ranks.append(rank)
        
        # keep track of input and truth rainfall values, to facilitate further ranks processing
        cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx]).astype(np.float32), ds_fac, axis=-1), ds_fac, axis=-2)
        lowress.append(cond_exp.ravel())
        hiress.append(obs.astype(np.float32).ravel())
        del samples_gen_ranks, truth_flat
        gc.collect()

        if show_progress:
            emmse_so_far = np.sqrt(np.mean(emmse_all))
            crps_mean = np.mean(crps_scores['no_pooling'])
            losses = [("EM-MSE", emmse_so_far), ("CRPS", crps_mean)]
            progbar.add(1, values=losses)
            
        # This is crucial to get different data!
        data_idx += 1

    truth_array = np.stack(truth_vals, axis=0)
    samples_gen_array = np.stack(samples_gen_vals, axis=0)
    fcst_array = np.stack(fcst_vals, axis=0)
    persisted_fcst_array = np.stack(persisted_fcst_vals, axis=0)
    cond_array = np.stack(cond_vals, axis=0)
    const_array = np.stack(const_vals, axis=0)
    
    point_metrics = {}
        
    ralsd_rmse_all = np.concatenate(ralsd_rmse_all)
    
    for method in crps_scores:
        point_metrics['CRPS_' + method] = np.asarray(crps_scores[method]).mean()
     
    point_metrics['rmse'] = np.sqrt(np.mean(mse_all))
    point_metrics['emmse'] = np.sqrt(np.mean(emmse_all))
    point_metrics['mse_fcst'] = np.sqrt(np.mean(fcst_mse_all))
    point_metrics['ralsd'] = np.nanmean(ralsd_rmse_all)
    point_metrics['corr'] = np.mean(corr_all)
    point_metrics['corr_fcst'] = np.mean(correlation_fcst_all)
    
    ranks = np.concatenate(ranks)
    lowress = np.concatenate(lowress)
    hiress = np.concatenate(hiress)
    gc.collect()
    if normalize_ranks:
        ranks = (ranks / ensemble_size).astype(np.float32)
        gc.collect()
    rank_arrays = (ranks, lowress, hiress)
    
    arrays = {'truth': truth_array, 'samples_gen': samples_gen_array, 'fcst_array': fcst_array, 
              'persisted_fcst': persisted_fcst_array, 'dates': dates, 'hours': hours}

    return rank_arrays, point_metrics, arrays


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
                                  weights_dir,
                                  records_folder,
                                  downsample,
                                  noise_factor,
                                  model_numbers,
                                  log_folder,
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
                                  data_paths,
                                  shuffle,
                                  fcst_fields=None,
                                  save_generated_samples=False,
                                  batch_size: int=1
                                  ):

    df_dict = read_config.read_config()['DOWNSCALING']

    gen, data_gen_valid = setup_inputs(mode=mode,
                                       arch=arch,
                                       records_folder=records_folder,
                                       fcst_data_source=fcst_data_source,
                                       obs_data_source=obs_data_source,
                                       fcst_fields=fcst_fields,
                                       latitude_range=latitude_range,
                                       longitude_range=longitude_range,
                                       downscaling_steps=df_dict["steps"],
                                       validation_range=validation_range,
                                       hour='random',
                                       downsample=downsample,
                                       input_channels=input_channels,
                                       filters_gen=filters_gen,
                                       filters_disc=filters_disc,
                                       noise_channels=noise_channels,
                                       latent_variables=latent_variables,
                                       padding=padding,
                                       constant_fields=constant_fields,
                                       data_paths=data_paths,
                                       shuffle=shuffle)

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
        rank_arrays, agg_metrics, arrays = eval_one_chkpt(mode=mode,
                                             gen=gen,
                                             data_gen=data_gen_valid,
                                             fcst_data_source=fcst_data_source,
                                             noise_channels=noise_channels,
                                             latent_variables=latent_variables,
                                             num_images=num_images,
                                             ensemble_size=ensemble_size,
                                             noise_factor=noise_factor,
                                             latitude_range=latitude_range,
                                             longitude_range=longitude_range,
                                             batch_size=batch_size)
        ranks, lowress, hiress = rank_arrays
        OP = rank_OP(ranks)
        
        # save one directory up from model weights, in same dir as logfile
        output_folder = os.path.join(log_folder, f"n{num_images}_{'-'.join(validation_range)}_e{ensemble_size}")
        
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        
        # Create a dataframe of all the data (to ensure scores are recorded in the right column)
        df = pd.DataFrame.from_dict(dict(N=model_number, op=OP, **{k: [v] for k, v in agg_metrics.items()}))
        eval_fname = os.path.join(output_folder, f"eval_validation.csv")
        df.to_csv(eval_fname, header=header, mode='a', float_format='%.6f', index=False)
        header = False

        if save_generated_samples:
            with open(os.path.join(output_folder, f"arrays-{model_number}.pkl"), 'wb+') as ofh:
                pickle.dump(arrays, ofh, pickle.HIGHEST_PROTOCOL)
        

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


def get_diurnal_cycle(array: np.ndarray, 
                      dates: list, 
                      hours: list, 
                      longitude_range: Iterable, 
                      latitude_range: Iterable):
    
    hourly_sums = {}
    hourly_counts = {}

    n_samples = array.shape[0]

    for n in range(n_samples):
        
        h = hours[n]
        d = dates[n]

        for l_ix, long in enumerate(longitude_range):
            
            local_hour = utils.get_local_datetime(utc_datetime=datetime(d.year, d.month, d.day, h),
                                                  longitude=long, latitude=np.mean(latitude_range)).hour

            if local_hour not in hourly_counts:
                hourly_counts[local_hour] = 1
                hourly_sums[local_hour] = array[n,:,l_ix].mean()
            else:
                hourly_counts[local_hour] += 1
                hourly_sums[local_hour] += array[n,:,l_ix].mean()
    return hourly_sums, hourly_counts

def get_fss_scores(truth_array, fss_data_dict, hourly_thresholds, window_sizes, n_samples):

    fss_results = {'thresholds': hourly_thresholds, 'window_sizes': window_sizes, 'scores': {}}

    for data_name, data_array in tqdm(fss_data_dict.items()):
        fss_results['scores'][data_name] = []
        for thr in hourly_thresholds:

            tmp_fss = []

            for w in tqdm(window_sizes):
                
                tmp_fss.append(fss(truth_array[:n_samples, :, :], data_array, w, thr, mode='constant'))
            
            fss_results['scores'][data_name].append(tmp_fss)
    return fss_results
