import gc
import os
from glob import glob
from tensorflow.keras.optimizers import Adam

from dsrnngan.model import deterministic, setupmodel
from dsrnngan.model import gan
from dsrnngan.model import models
from dsrnngan.model.vaegantrain import VAE
from dsrnngan.utils import read_config
from dsrnngan.utils.utils import load_yaml_file

            # lr_disc=model_config.discriminator.learning_rate_disc,
            # lr_gen=model_config.generator.learning_rate_gen,
            # kl_weight=model_config.train.kl_weight,
            # ensemble_size=model_config.train.ensemble_size,
            # CLtype=model_config.train.CL_type,
            # content_loss_weight=model_config.train.content_loss_weight

def setup_model(*,
                model_config,
                data_config):
                                   
    if model_config.mode in ("GAN", "VAEGAN"):
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator,
                      "forceconv-long": models.generator}[model_config.architecture]
        disc_to_use = {"normal": models.discriminator,
                       "forceconv": models.discriminator,
                       "forceconv-long": models.discriminator}[model_config.architecture]
    elif model_config.mode == "det":
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator}[model_config.architecture]

    if model_config.mode == 'GAN':
        gen = gen_to_use(mode=model_config.mode,
                         arch=model_config.architecture,
                         downscaling_steps=model_config.downscaling_steps,
                         input_channels=data_config.input_channels,
                         num_constant_fields=len(data_config.constant_fields),
                         noise_channels=model_config.generator.noise_channels,
                         filters_gen=model_config.generator.filters_gen,
                         padding=model_config.padding,
                         output_activation=model_config.generator.output_activation,
                         norm=model_config.generator.normalisation)
        disc = disc_to_use(arch=model_config.architecture,
                           downscaling_steps=model_config.downscaling_steps,
                           input_channels=data_config.input_channels,
                           num_constant_fields=len(data_config.constant_fields),
                           filters_disc=model_config.discriminator.filters_disc,
                           padding=model_config.padding,
                           norm=model_config.discriminator.normalisation)
        model = gan.WGANGP(gen, disc, model_config.mode, lr_disc=model_config.discriminator.learning_rate_disc, lr_gen=model_config.generator.learning_rate_gen,
                           ensemble_size=model_config.train.ensemble_size,
                           CLtype=model_config.train.CL_type,
                           content_loss_weight=model_config.train.content_loss_weight)
    elif model_config.mode == 'VAEGAN':
        (encoder, decoder) = gen_to_use(mode=model_config.mode,
                                        arch=model_config.architecture,
                                        downscaling_steps=model_config.downscaling_steps,
                                        input_channels=data_config.input_channels,
                                        latent_variables=model_config.generator.latent_variables,
                                        filters_gen=model_config.generator.filters_gen,
                                        padding=model_config.padding)
        disc = disc_to_use(arch=model_config.architecture,
                           downscaling_steps=model_config.downscaling_steps,
                           input_channels=data_config.input_channels,
                           filters_disc=model_config.discriminator.filters_disc,
                           padding=model_config.padding)
        gen = VAE(encoder, decoder)
        model = gan.WGANGP(gen, disc, model_config.mode, lr_disc=model_config.discriminator.learning_rate_disc,
                           lr_gen=model_config.generator.learning_rate_gen, kl_weight=model_config.train.kl_weight,
                           ensemble_size=model_config.train.ensemble_size,
                           CLtype=model_config.train.CL_type,
                           content_loss_weight=model_config.train.content_loss_weight)
    elif model_config.mode == 'det':
        gen = gen_to_use(mode=model_config.mode,
                         arch=model_config.architecture,
                         downscaling_steps=model_config.downscaling_steps,
                         input_channels=data_config.input_channels,
                         filters_gen=model_config.generator.filters_gen,
                         padding=model_config.padding)
        model = deterministic.Deterministic(gen,
                                            lr=model_config.generator.learning_rate_gen,
                                            loss='mse',
                                            optimizer=Adam)

    gc.collect()
    return model


def load_model_from_folder(model_folder, model_number=None):

    model_weights_root = os.path.join(model_folder, "models")
    config_path = os.path.join(model_folder, 'setup_params.yaml')

    if model_number is None:
        model_fp = sorted(glob(os.path.join(model_weights_root, '*.h5')))[-1]
    else:
        model_fp = os.path.join(model_weights_root, f'gen_weights-{model_number:07d}.h5')

    setup_params = load_yaml_file(config_path)
    model_config, data_config = read_config.get_config_objects(setup_params)

    print('setting up inputs')
    model = setupmodel.setup_model(model_config=model_config, data_config=data_config)

    gen = model.gen

    print('loading weights')
    gen.load_weights(model_fp)

    return gen

