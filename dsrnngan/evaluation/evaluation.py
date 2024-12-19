import gc
import os
import sys
import warnings
from datetime import datetime, timedelta
import pickle
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from properscoring import crps_ensemble
from tensorflow.python.keras.utils import generic_utils
from datetime import datetime
from typing import Iterable
from types import SimpleNamespace
import netCDF4 as nc
import xarray as xr

from dsrnngan.data.data_generator import DataGenerator
from dsrnngan.data import data
from dsrnngan.utils import read_config, utils
from dsrnngan.data import setupdata
from dsrnngan.model import setupmodel
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.model.pooling import pool
from dsrnngan.evaluation.rapsd import rapsd
from dsrnngan.evaluation.scoring import rmse, mse, mae, calculate_pearsonr, fss, critical_success_index
from dsrnngan.model import gan

warnings.filterwarnings("ignore", category=RuntimeWarning)


path = os.path.dirname(os.path.abspath(__file__))
model_config = read_config.read_model_config()
ds_fac = model_config.downscaling_factor

metrics = {'correlation': calculate_pearsonr, 'mae': mae, 'mse': mse,
           }

def setup_inputs(
                 model_config: SimpleNamespace,
                 data_config: SimpleNamespace,
                 records_folder: str,
                 hour: int,
                 shuffle: bool,
                 batch_size: int,
                 use_training_data: bool=False):

    # initialise model
    model = setupmodel.setup_model(model_config=model_config,
                                   data_config=data_config)

    gen = model.gen


    num_constant_fields = len(data_config.constant_fields)
    
    input_image_shape = (model_config.eval.img_height , model_config.eval.img_width, data_config.input_channels)
    output_image_shape = (model_config.downscaling_factor * input_image_shape[0], model_config.downscaling_factor * input_image_shape[1])
    constants_image_shape = (model_config.downscaling_factor * input_image_shape[0] , model_config.downscaling_factor * input_image_shape[1], num_constant_fields)

    print("i: "+str(input_image_shape))
    print("o: "+str(output_image_shape))
    print("c: "+str(constants_image_shape))

    # always uses full-sized images
    print('Loading full sized image dataset')
    data_gen_train, data_gen_valid = setupdata.setup_data(
        data_config,
        model_config,
        fcst_shape=input_image_shape,
        con_shape=constants_image_shape,
        out_shape=output_image_shape,
        records_folder=records_folder,
        load_full_image=True,
        shuffle=shuffle,
        hour=hour,
        full_image_batch_size=batch_size,
        use_training_data=use_training_data,
        data_label="evaluation")
    
    if use_training_data:
        return gen, data_gen_train
    else:
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
                   data_config: SimpleNamespace,
                   model_config: SimpleNamespace,
                   batch_size: int,
                   gen: gan.WGANGP,
                   lo_res_inputs, #data_gen: DataGenerator,
                   hi_res_inputs,
                   output,
                   latitude_range: Iterable,
                   longitude_range: Iterable,
                   ensemble_size: int,
                   output_normalisation: str,
                   input_normalisation: str,
                   seed: int=None
                   ):


    input_field_variables = list(data_config.input_fields.keys())
    if 'tp' in input_field_variables:
        tpidx = input_field_variables.index('tp')
    else:
        # Temporary fix to allow testing of other tp variants
        tpidx = input_field_variables.index('tpq')
    
    batch_size = 1  # do one full-size image at a time

    if model_config.mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    
    cond = lo_res_inputs
    fcst = cond[0, :, :, :]
    const = hi_res_inputs
    obs = output[0, :, :]

    if output_normalisation is not None:
        obs = data.denormalise(obs, normalisation_type=output_normalisation)
    
    if input_normalisation is not None:
        fcst = data.denormalise(fcst, normalisation_type=input_normalisation)

    lo_res_target = fcst[:, :, tpidx]
    
    if model_config.mode == "GAN":
        original_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f  
            samples_gen = generate_gan_sample(gen, cond, const, model_config.generator.noise_channels, ensemble_size, batch_size, seed)
        sys.stdout = original_stdout
        
            
    elif model_config.mode == "det":
        samples_gen = []
        sample_gen = gen.predict([cond, const])
        samples_gen.append(sample_gen.astype("float32"))
        
    elif model_config.mode == 'VAEGAN':
        samples_gen = []
        # call encoder once
        mean, logvar = gen.encoder([cond, const])
        noise_shape = np.array(cond)[0, ..., 0].shape + (model_config.latent_variables,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        
        for ii in range(ensemble_size):
            nn = noise_gen()
            # generate ensemble of preds with decoder
            sample_gen = gen.decoder.predict([mean, logvar, nn, const])
            samples_gen.append(sample_gen.astype("float32"))

    # samples generated, now process them (e.g., undo log transform) and calculate MAE etc
    for ii in range(ensemble_size):
        
        sample_gen = samples_gen[ii][0, :, :, 0]
        
        if output_normalisation is not None:
            sample_gen = data.denormalise(sample_gen, normalisation_type=output_normalisation)

        samples_gen[ii] = sample_gen

    return obs, samples_gen, fcst, lo_res_target, cond, const

def eval_one_chkpt(*,
                   gen,
                   data_config,
                   model_config,
                   data_gen,
                   num_images,
                   noise_factor,
                   ensemble_size,
                   normalize_ranks=True,
                   show_progress=True,
                   batch_size: int=1,
                   output_folder=None,
                   model_number=None,
                   eval_months_idx=None,
                   skip_interval=1,
                   lon_min=None,
                   lon_max=None,
                   lat_min=None,
                   lat_max=None):
    
    latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(data_config)
    output_normalisation = data_config.output_normalisation
    
    if data_config.normalise_inputs:
        input_normalisation = data_config.input_normalisation_strategy['tp']['normalisation']
    else:
        input_normalisation = None


    if num_images is not None and num_images < 5:
        Warning('These scores are best performed with more images')

    if lon_min and lon_max:
        lon_range = [lon_min, lon_max]
    else:
        lon_range = None

    if lat_min and lat_max:
        lat_range = [lat_min, lat_max]
    else:
        lat_range = None

    
    truth_vals = []
    samples_gen_vals = []
    fcst_vals = []
    lo_res_target_vals = []
    time_vals = []
    time_vals_str = []
    time_vals_int = []
    cond_vals = []
    const_vals = []
    ranks = []
    lowress = []
    hiress = []
    crps_scores = {}
    mse_all = []
    emmse_all = []
    ralsd_rmse_all = []
    corr_all = []
    csi_all = []
    max_bias_all = []
    max_quantile_diff = []

    agg_metrics = []

    show_progress_temp = show_progress
    
    input_field_variables = list(data_config.input_fields.keys())
    if 'tp' in input_field_variables:
        tpidx = input_field_variables.index('tp')
    else:
        # Temporary fix to allow testing of other tp variants
        tpidx = input_field_variables.index('tpq')
    
    batch_size = 1  # do one full-size image at a time

    if model_config.mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    if show_progress:
        # Initialize progbar
        progbar = generic_utils.Progbar(target=num_images,
                                        stateful_metrics=("CRPS", "EM-RMSE"))

    CRPS_pooling_methods = ['no_pooling', 'max_4', 'max_16', 'avg_4', 'avg_16']

    rng = np.random.default_rng()


    save_at_end = True
    try:
        total_processed_images=0
        data_gen_iter = data_gen.as_numpy_iterator()
        num_images_left = None
        num_processed_days = 0
        month_idx = 0
        month = 0
        year = 0
        while num_images is None or total_processed_images < num_images:
            data_batch = next(data_gen_iter)
            if len(data_batch) > 0:
                number_of_days_per_batch = len(data_batch[0][3])
            else:
                continue
            for k in range(number_of_days_per_batch):
                if num_images is not None and total_processed_images >= num_images:
                        break
                processed_images = 0
                
                if num_processed_days%skip_interval != 0:
                    num_processed_days += 1
                    continue
                else:
                    num_processed_days += 1
                    
                for j, hour_data in enumerate(data_batch):
                    if num_images is not None and total_processed_images >= num_images:
                        break
                    processed_images += 1
                    total_processed_images += 1
                    lo_res_inputs, hi_res_inputs, output, timestamp = hour_data
                    success = False
                    time_pd = pd.Timestamp.fromtimestamp(timestamp[k][0], tz='UTC')
                    time_np = np.datetime64(time_pd)
                    #print("time_np: "+str(time_np))
                    time_int = timestamp[k][0]
                    time_str = time_pd.strftime('%Y-%m-%d %H:%M:%S')
                    month_old = month
                    year_old = year
                    month = time_pd.month
                    year = time_pd.year

                    save_monthly_data = False
                    if month_old!=0 and month!=month_old:
                        if eval_months_idx is None:
                            save_monthly_data = True
                        elif month_idx in eval_months_idx:
                            save_monthly_data = True
                            
                    if save_monthly_data:
                        save_netCDF(time_vals, time_vals_int, time_vals_str, truth_vals, 
                                    samples_gen_vals, fcst_vals, lo_res_target_vals, 
                                    output_folder, model_number, month_old, 
                                    year_old, lon_range, lat_range, input_field_variables)
                        
                        agg_metrics.append(save_metrics(crps_scores, mse_all, emmse_all, 
                                                        ralsd_rmse_all, corr_all, csi_all, 
                                                        max_bias_all, max_quantile_diff, ranks, 
                                                        lowress, hiress, normalize_ranks, 
                                                        ensemble_size, month_old, year_old, 
                                                        model_number, output_folder))
                        
                        time_vals, time_vals_int, time_vals_str, truth_vals, samples_gen_vals, fcst_vals, cond_vals, const_vals, \
                        lo_res_target_vals, mse_all, emmse_all, ralsd_rmse_all, corr_all, csi_all, max_bias_all, \
                        max_quantile_diff, ranks, lowress, hiress = ([] for _ in range(19))
                        crps_scores = {}
                        gc.collect()

                    if month_old!=0 and month!=month_old:
                        month_idx += 1

                    if eval_months_idx is not None and month_idx not in eval_months_idx:
                        show_progress_temp = False
                        if month_idx > max(eval_months_idx):
                            save_at_end = False
                        continue

                    show_progress_temp = show_progress

                    for n in range(5):
                        if success:
                            continue
                        try:         
                            obs, samples_gen, fcst, lo_res_target, cond, const = create_single_sample(
                                    data_config=data_config,
                                    model_config=model_config,
                                    batch_size=1,
                                    gen=gen,
                                    lo_res_inputs=np.expand_dims(lo_res_inputs[k, ...], axis=0), 
                                    hi_res_inputs=np.expand_dims(hi_res_inputs[k, ...], axis=0),
                                    output=np.expand_dims(output[k, ...], axis=0),
                                    latitude_range=latitude_range,
                                    longitude_range=longitude_range,
                                    ensemble_size=ensemble_size,
                                    output_normalisation=output_normalisation,
                                    input_normalisation=input_normalisation
                                    )
                            success = True
                            
                        except Exception as e:
                            if n < 4:
                                print(f'Could not load file, attempting retry {n+1} of 4')
                                continue
                            else:
                                raise e
                
                    cond_vals.append(cond)
                    const_vals.append(const)
                    time_vals_str.append(time_str)
                    time_vals.append(time_np)
                    time_vals_int.append(time_int)
                    
            
                    # Do all RALSD at once, to avoid re-calculating power spectrum of truth image
                
                    ralsd_rmse = calculate_ralsd_rmse(obs, samples_gen)
                    ralsd_rmse_all.append(ralsd_rmse.flatten())
           
                    # turn list of predictions into array, for CRPS/rank calculations
                    samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]
                    
                    mse_all.append(mse(samples_gen[:,:,0], obs))
                    emmse_all.append(mse(np.mean(samples_gen, axis=-1), obs))
                    
                    corr_all.append(calculate_pearsonr(obs, samples_gen[:, :, 0]))
      
                    # critical success index
                    csi_all.append(critical_success_index(obs, samples_gen[:,:,0], threshold=1.0))
    
                    # Max difference
                    max_bias_all.append((samples_gen - np.stack([obs]*ensemble_size, axis=-1)).max())
    
                    # Max difference at 99.99th quantile
                    max_quantile_diff.append( np.abs(np.quantile(samples_gen[:,:,0], 0.99999) - np.quantile(obs, 0.99999)).max() )
                    
                    # Store these values for e.g. correlation on the grid
                    truth_vals.append(obs)
                    samples_gen_vals.append(samples_gen)
                    fcst_vals.append(fcst)
                    lo_res_target_vals.append(lo_res_target)
    
                    ####################  CRPS calculation ##########################
                    # calculate CRPS scores for different pooling methods
                    
                    for method in CRPS_pooling_methods:
                        truth_pooled = pool(obs, method)
                        samples_gen_pooled = pool(samples_gen, method)
                        # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
                        crps_truth_input = np.squeeze(truth_pooled, axis=-1)
                        crps_gen_input = samples_gen_pooled
                        crps_score_grid = crps_ensemble(crps_truth_input, crps_gen_input)
                        crps_score = crps_score_grid.mean()
                        del truth_pooled, samples_gen_pooled, crps_truth_input, crps_gen_input
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
                    cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx], normalisation_type=input_normalisation).astype(np.float32), ds_fac, axis=-1), ds_fac, axis=-2)
                    lowress.append(cond_exp.ravel())
                    hiress.append(obs.astype(np.float32).ravel())
                    del samples_gen_ranks, truth_flat
                    gc.collect()
                
                if show_progress_temp:
                    emrmse_so_far = np.sqrt(np.mean(emmse_all))
                    crps_mean = np.mean(crps_scores['no_pooling'])
                    losses = [("EM-RMSE", emrmse_so_far), ("CRPS", crps_mean)]
                    progbar.add(processed_images, values=losses)

    except StopIteration:
        # Wenn das Dataset erschöpft ist, verlasse die Schleife
        print("All batches have been processed.")
        print("number of processed images: "+str(total_processed_images))

    if save_at_end:
        agg_metrics.append(save_metrics(crps_scores, mse_all, emmse_all, 
                                        ralsd_rmse_all, corr_all, csi_all, 
                                        max_bias_all, max_quantile_diff, ranks, 
                                        lowress, hiress, normalize_ranks, 
                                        ensemble_size, month_old, year_old, 
                                        model_number, output_folder))
        
        save_netCDF(time_vals, time_vals_int, time_vals_str, truth_vals, samples_gen_vals, 
                    fcst_vals, lo_res_target_vals,  output_folder, model_number, 
                    month, year, lon_range, lat_range, input_field_variables)

    df = pd.concat(agg_metrics, ignore_index=True)
    df = df.drop(columns=['month', 'year'])

    result_df = pd.DataFrame([
        {**{'N': n}, 
         **{metric: weighted_avg(df[df['N'] == n], metric) for metric in df.columns if metric != 'N'}}
        for n in df['N'].unique()
    ])
    result_df['rmse'] = np.sqrt(result_df['mse'])
    result_df['emrmse'] = np.sqrt(result_df['emmse'])

    return result_df


def weighted_avg(df, metric_col):
    avg_values = [x[0] for x in df[metric_col]]
    counts = [x[1] for x in df[metric_col]]
    return np.average(avg_values, weights=counts)

def save_metrics(crps_scores, mse_all, emmse_all, ralsd_rmse_all, 
                 corr_all, csi_all, max_bias_all, max_quantile_diff, 
                 ranks, lowress, hiress, normalize_ranks, 
                 ensemble_size, month, year, model_number, output_folder):
    
    ralsd_rmse_all = np.concatenate(ralsd_rmse_all)

    point_metrics = {}
    point_metrics['month'] = month
    point_metrics['year'] = year
    for method in crps_scores:
        point_metrics['CRPS_' + method] = (np.asarray(crps_scores[method]).mean(), len(np.asarray(crps_scores[method])))
    point_metrics['mse'] = (np.mean(mse_all), len(mse_all))
    point_metrics['emmse'] = (np.mean(emmse_all), len(emmse_all))
    point_metrics['ralsd'] = (np.nanmean(ralsd_rmse_all), np.sum(~np.isnan(ralsd_rmse_all)))
    point_metrics['corr'] = (np.mean(corr_all), len(corr_all))
    point_metrics['csi'] = (np.mean(csi_all), len(csi_all))
    point_metrics['max_bias'] = (np.mean(max_bias_all), len(max_bias_all))
    point_metrics['max_quantile_diff'] = (np.mean(max_quantile_diff), len(max_quantile_diff))
    
    ranks = np.concatenate(ranks)
    #lowress = np.concatenate(lowress)
    #hiress = np.concatenate(hiress)
    if normalize_ranks:
        ranks = (ranks / ensemble_size).astype(np.float32)
    OP = rank_OP(ranks)

    op_tuple= (OP, len(ranks))
    
    # Create a dataframe of all the data (to ensure scores are recorded in the right column)
    df = pd.DataFrame.from_dict(dict(N=model_number, op=[op_tuple], **{k: [v] for k, v in point_metrics.items()}))
    #df = pd.DataFrame.from_dict(dict(N=model_number, **{k: [v] for k, v in point_metrics.items()}))
    eval_fname = os.path.join(output_folder, f"eval_validation_temp.csv")
    write_header = not os.path.exists(eval_fname)
    df.to_csv(eval_fname, header=write_header, mode='a', float_format='%.6f', index=False)

    return df

def save_netCDF(time_vals, time_vals_int, time_vals_str, truth_vals, samples_gen_vals, 
                fcst_vals, lo_res_target_vals, output_folder, model_number, 
                month, year, lon_range, lat_range, input_field_variables):
    
    time_str_array = np.array(time_vals_str, dtype='S19') 
    time_int_array = np.array(time_vals_int) 
    time_array = np.array(time_vals) 
    truth_array = np.stack(truth_vals, axis=0)
    samples_gen_array = np.stack(samples_gen_vals, axis=0)
    samples_gen_mean_array = np.stack(np.mean(samples_gen_vals, axis=-1), axis=0)
    fcst_array = np.stack(fcst_vals, axis=0)
    lo_res_target_array = np.stack(lo_res_target_vals, axis=0)
    print("truth_array.shape: "+str(truth_array.shape)) 
    print("samples_gen_array.shape: "+str(samples_gen_array.shape)) 
    print("samples_gen_mean_array.shape: "+str(samples_gen_mean_array.shape)) 
    print("fcst_array.shape: "+str(fcst_array.shape)) 
    print("lo_res_target_array.shape: "+str(lo_res_target_array.shape)) 
    print("time_array.shape: "+str(time_array.shape)) 
    print("time_str_array.shape: "+str(time_str_array.shape)) 

    arrays = {'truth': truth_array, 'samples_gen': samples_gen_array, 'input': fcst_array, 'input (target)': lo_res_target_array, 'samples_gen_mean': samples_gen_mean_array, 'time': time_array, 'time (str)': time_str_array, 'time (int)': time_int_array}

    filename_nc = os.path.join(output_folder, f'pred_samples_{model_number}_y{year}_m{month}.nc')

    # recover geographical coordinate information and define coordinates
    #Lat 24.9166666666667 - 73.8333333333333
    #Lon -19.0833333333333 - 35.0833333333333
    if lat_range:
        lat = np.linspace(lat_range[0], lat_range[1], arrays['samples_gen'].shape[1])
    else:
        lat = np.arange(0, arrays['samples_gen'].shape[1])
    if lon_range:
        lon = np.linspace(lon_range[0], lon_range[1], arrays['samples_gen'].shape[2])
    else:
        lon = np.arange(0, arrays['samples_gen'].shape[2])
    ens = np.arange(1, arrays['samples_gen'].shape[3]+1)
    #channels = np.arange(0, arrays["input"].shape[3])
    channels = input_field_variables if input_field_variables and len(input_field_variables)== arrays["input"].shape[3] else np.arange(0, arrays["input"].shape[3])
    if not input_field_variables or len(input_field_variables) != arrays["input"].shape[3]:
        print(
            f"Warning: input_field_variables is invalid or mismatched. "
            f"Falling back to numeric indices for channels."
        )
    time = arrays["time"]
    #print(time)
    print("Datatype of time:", time.dtype)

    # Create Dataset
    
    tot_prec_ref = xr.DataArray(
        arrays["truth"],
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="tot_prec_ref",
        attrs={"long_name": "Ground truth precipitation", "units": "kg/m^2", "source": "IMERG"}
    )
    
    tot_prec_pred = xr.DataArray(
        arrays["samples_gen"],
        dims=["time", "lat", "lon", "ens"],
        coords={"time": time, "lat": lat, "lon": lon, "ens": ens},
        name="tot_prec_pred",
        attrs={"long_name": "Generated precipitation samples", "units": "kg/m^2", "source": "HarrisWGAN Model"}
    )
    
    tot_prec_pred_mean = xr.DataArray(
        arrays["samples_gen_mean"],
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="tot_prec_pred_mean",
        attrs={"long_name": "Mean of generated precipitation samples", "units": "kg/m^2", "source": "Model"}
    )
    
    input_tot_prec_data = xr.DataArray(
        np.repeat(np.repeat(arrays['input (target)'], 3, axis=1), 3, axis=2),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="input_tot_prec_data",
        attrs={"long_name": "Low resolution target precipitation", "units": "kg/m^2", "source": "ERA5"}
    )
    
    input_data = xr.DataArray(
        np.repeat(np.repeat(arrays['input'], 3, axis=1), 3, axis=2),
        dims=["time", "lat", "lon", "channel"],
        coords={"time": time, "lat": lat, "lon": lon, "channel": channels},
        name="input_data",
        attrs={"long_name": "Input forecast data", "source": "ERA5"}
    )
    
    time_str = xr.DataArray(
        arrays['time (str)'],
        dims=["time"],
        coords={"time": time},
        name="time_str",
        attrs={"long_name": "Human-readable time strings"}
    )

    time_int = xr.DataArray(
        arrays['time (int)'],
        dims=["time"],
        coords={"time": time},
        name="time_int",
        attrs={"long_name": "seconds since 1970-01-01 00:00:00 UTC"}
    )
    
    
    ds= xr.Dataset(
        {
            "tot_prec_ref": tot_prec_ref,
            "tot_prec_pred": tot_prec_pred,
            "tot_prec_pred_mean": tot_prec_pred_mean,
            "input_tot_prec_data": input_tot_prec_data,
            "input_data": input_data,
            #"time_str": time_str,
            #"time_int": time_int,
        },
        attrs={
            "title": "Total Precipitation Downscaling Dataset",
            "institution": "Forschungszentrum Jülich (FZJ)",
            "source": "IMERG, ERA5, Harris WGAN Downscaling Model",
            "history": f"Created on {pd.Timestamp.now().isoformat(timespec='seconds')}",
        }
    )
    
    # Save the dataset to a NetCDF file
    ds.to_netcdf(filename_nc, format="NETCDF4")

    print(f'Data successfully saved as CF-compliant NetCDF in {filename_nc}!')
   
    return 

def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks == 0) | (norm_ranks == 1)
    )
    op = float(op)/len(norm_ranks)
    return op


def log_line(log_fname, line):
    with open(log_fname, 'a') as f:
        print(line, file=f)


def evaluate_multiple_checkpoints(
                                  model_config,
                                  data_config,
                                  weights_dir,
                                  records_folder,
                                  model_numbers,
                                  log_folder,
                                  noise_factor,
                                  num_images,
                                  ensemble_size,
                                  shuffle,
                                  save_generated_samples=False,
                                  batch_size: int=1,
                                  use_training_data: bool=False,
                                  eval_months_idx=None,
                                  eval_model_idx=None,
                                  skip_interval=1,
                                  lon_min=None,
                                  lon_max=None,
                                  lat_min=None,
                                  lat_max=None
                                  ):

    gen, data_gen = setup_inputs(model_config=model_config,
                                       data_config=data_config,
                                       records_folder=records_folder,
                                       hour='random',
                                       shuffle=shuffle,
                                       batch_size=batch_size,
                                       use_training_data=use_training_data)
    header = True

    for model_number_index, model_number in enumerate(model_numbers):

        if eval_model_idx is not None and model_number_index != eval_model_idx:
            continue
        print(f"model_number: {str(model_number)}")
        gen_weights_file = os.path.join(weights_dir, f"gen_weights-{model_number:07d}.h5")

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        if model_config.mode == "VAEGAN":
            _init_VAEGAN(gen, data_gen, True, 1, model_config.latent_variables)
        gen.load_weights(gen_weights_file)
        

        if use_training_data:
            ym_range = model_config.train.training_range
        else:
            ym_range = model_config.val.val_range
            
        # save one directory up from model weights, in same dir as logfile
        # Assuming that it is very unlikely to have the same start and end of the range and have a collision on this hash
        range_hash = hashlib.sha256(str(ym_range).encode('utf-8')).hexdigest()[:5]
        num_images_index = num_images if num_images is not None else 0
        output_folder = os.path.join(log_folder, f"n{num_images_index}_{ym_range[0][0]}-{ym_range[-1][-1]}_{range_hash}_e{ensemble_size}_complete")
        
        os.makedirs(output_folder, exist_ok=True)

        df = eval_one_chkpt(
                 gen=gen,
                 data_gen=data_gen,
                 data_config=data_config,
                 model_config=model_config,
                 num_images=num_images,
                 ensemble_size=ensemble_size,
                 noise_factor=noise_factor,
                 batch_size=batch_size,
                 output_folder=output_folder,
                 model_number=model_number,
                 eval_months_idx=eval_months_idx,
                 skip_interval=skip_interval,
                 lon_min=lon_min,
                 lon_max=lon_max,
                 lat_min=lat_min,
                 lat_max=lat_max)

        # Save exact validation range
        with open(os.path.join(output_folder, 'val_range.pkl'), 'wb+') as ofh:
            pickle.dump(ym_range, ofh)
        
        # Save a dataframe of all the data (to ensure scores are recorded in the right column)
        eval_fname = os.path.join(output_folder, f"eval_validation.csv")
        write_header = not os.path.exists(eval_fname)
        df.to_csv(eval_fname, header=write_header, mode='a', float_format='%.6f', index=False)

        

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
