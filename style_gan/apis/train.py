import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ..utils.logger import get_root_logger
from ..datasets.builder import build_dataloader
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)
from ..runners import ProGANRunner
                         
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_gan(model,
              dataset,
              cfg,
              distributed=False,
              validate=False,
              timestamp=None,
              meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    print('dataset length', len(dataset[0]))

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = {
        'opt_gen': build_optimizer(model.module.generator, cfg.optimizer_gen),
        'opt_disc': build_optimizer(model.module.discriminator, cfg.optimizer_disc)
    }
    runner = ProGANRunner(
        model=model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(
    #         **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # elif distributed and 'type' not in cfg.optimizer_config:
    #     optimizer_config = OptimizerHook(**cfg.optimizer_config)
    # else:
    # optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(checkpoint_config=cfg.checkpoint_config, 
                                   log_config=cfg.log_config)
    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    runner.register_hook_from_cfg(cfg.save_image_config)
    # register eval hooks
    # if validate:
    #     val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    #     val_dataloader = build_dataloader(
    #         val_dataset,
    #         samples_per_gpu=1,
    #         workers_per_gpu=cfg.data.workers_per_gpu,
    #         dist=distributed,
    #         shuffle=False)
    #     eval_cfg = cfg.get('evaluation', {})
    #     eval_hook = DistEvalHook if distributed else EvalHook
    #     runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.stage_epochs, cfg.fade_in_percentages)
