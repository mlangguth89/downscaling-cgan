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


def setup_model(*,
                mode,
                architecture,
                downscaling_steps,
                input_channels,
                filters_gen,
                filters_disc,
                noise_channels,
                num_constant_fields,
                latent_variables=None,
                padding=None,
                kl_weight=None,
                ensemble_size=None,
                CLtype=None,
                content_loss_weight=None,
                lr_disc=None,
                lr_gen=None,
                rotate=False):

    if mode in ("GAN", "VAEGAN"):
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator,
                      "forceconv-long": models.generator}[architecture]
        disc_to_use = {"normal": models.discriminator,
                       "forceconv": models.discriminator,
                       "forceconv-long": models.discriminator}[architecture]
    elif mode == "det":
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator}[architecture]

    if mode == 'GAN':
        gen = gen_to_use(mode=mode,
                         arch=architecture,
                         downscaling_steps=downscaling_steps,
                         input_channels=input_channels,
                         num_constant_fields=num_constant_fields,
                         noise_channels=noise_channels,
                         filters_gen=filters_gen,
                         padding=padding,
                         rotate=rotate)
        disc = disc_to_use(arch=architecture,
                           downscaling_steps=downscaling_steps,
                           input_channels=input_channels,
                           num_constant_fields=num_constant_fields,
                           filters_disc=filters_disc,
                           padding=padding,
                           rotate=rotate)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc, lr_gen=lr_gen,
                           ensemble_size=ensemble_size,
                           CLtype=CLtype,
                           content_loss_weight=content_loss_weight)
    elif mode == 'VAEGAN':
        (encoder, decoder) = gen_to_use(mode=mode,
                                        arch=architecture,
                                        downscaling_steps=downscaling_steps,
                                        input_channels=input_channels,
                                        latent_variables=latent_variables,
                                        filters_gen=filters_gen,
                                        padding=padding)
        disc = disc_to_use(arch=architecture,
                           downscaling_steps=downscaling_steps,
                           input_channels=input_channels,
                           filters_disc=filters_disc,
                           padding=padding)
        gen = VAE(encoder, decoder)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc,
                           lr_gen=lr_gen, kl_weight=kl_weight,
                           ensemble_size=ensemble_size,
                           CLtype=CLtype,
                           content_loss_weight=content_loss_weight)
    elif mode == 'det':
        gen = gen_to_use(mode=mode,
                         arch=architecture,
                         downscaling_steps=downscaling_steps,
                         input_channels=input_channels,
                         filters_gen=filters_gen,
                         padding=padding)
        model = deterministic.Deterministic(gen,
                                            lr=lr_gen,
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
    model_config, _, ds_config, data_config, gen_config, dis_config, train_config, val_config = read_config.get_config_objects(setup_params)

    print('setting up inputs')
    model = setupmodel.setup_model(mode=model_config.mode,
                                   architecture=model_config.architecture,
                                   downscaling_steps=ds_config.steps,
                                   input_channels=data_config.input_channels,
                                   filters_gen=gen_config.filters_gen,
                                   filters_disc=dis_config.filters_disc,
                                   noise_channels=gen_config.noise_channels,
                                   latent_variables=gen_config.latent_variables,
                                   padding=model_config.padding,
                                   num_constant_fields=len(data_config.constant_fields),
                                   rotate=model_config.train.rotate)

    gen = model.gen

    print('loading weights')
    gen.load_weights(model_fp)

    return gen

