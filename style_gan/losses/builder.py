from mmcv.utils import Registry, build_from_cfg

LOSSES = Registry('loss')

def build_loss(cfg, default_args=None):
    loss = build_from_cfg(cfg, LOSSES, default_args)
    return loss