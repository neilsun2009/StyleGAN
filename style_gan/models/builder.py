from mmcv.utils import Registry, build_from_cfg
from torch import nn


GENERATORS = Registry('generator')
DISCRIMINATORS = Registry('discriminator')
GANS = Registry('gan')
GEN_DOWNSAMPLINGS = Registry('gen_downsampling')
GEN_UPSAMPLINGS = Registry('gen_upsampling')
GEN_MAPPINGS = Registry('gen_mapping')
GEN_SYNTHESISES = Registry('gen_synthesis')


def build(cfg, registry, default_args=None):
    """Build a module

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_generator(cfg, **default_args):
    return build(cfg, GENERATORS, default_args)


def build_discriminator(cfg, **default_args):
    return build(cfg, DISCRIMINATORS, default_args)

def build_gan(cfg, **default_args):
    return build(cfg, GANS, default_args)

def build_gen_downsampling(cfg, **default_args):
    return build(cfg, GEN_DOWNSAMPLINGS, default_args)

def build_gen_upsampling(cfg, **default_args):
    return build(cfg, GEN_UPSAMPLINGS, default_args)

def build_gen_mapping(cfg, **default_args):
    return build(cfg, GEN_MAPPINGS, default_args)

def build_gen_synthesis(cfg, **default_args):
    return build(cfg, GEN_SYNTHESISES, default_args)