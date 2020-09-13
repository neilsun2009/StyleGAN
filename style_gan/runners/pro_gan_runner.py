from mmcv.runner import IterBasedRunner, IterLoader
import mmcv
from mmcv.runner.utils import get_host_info
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.hooks import HOOKS, Hook, IterTimerHook
from mmcv.runner.priority import get_priority
from mmcv.runner.dist_utils import get_dist_info
from mmcv.runner.log_buffer import LogBuffer
import time
from os import path as osp
import numpy as np
import torch
import math
from ..datasets.builder import build_dataloader

class ProGANRunner(IterBasedRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._depth = 0
        self._alpha = 0
        self._total_depth = self.model.module.total_depth
        self._ticker = 1


    @property
    def total_depth(self):
        return self._total_depth

    @property
    def depth(self):
        return self._depth
    
    @property
    def alpha(self):
        return self._alpha

    @property
    def ticker(self):
        return self._ticker

    
    
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_iter')
        data_batch = next(data_loader)
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._iter += 1
        self._inner_iter += 1

    def run(self, datasets, samples_per_gpus, workflow, stage_epochs, fade_in_percentages, distributed=False,
        cfg=None, **kwargs):

        assert isinstance(datasets, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(datasets) == len(workflow)
        assert self.total_depth <= len(stage_epochs)
        assert self.total_depth <= len(fade_in_percentages)

        stage_epochs = stage_epochs[:self.total_depth]
        fade_in_percentages = fade_in_percentages[:self.total_depth]
        self._max_epochs = sum(stage_epochs)
        cum_stage_epochs = np.cumsum(stage_epochs)
        cum_epoch_iters = list()
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            dataset_length = len(datasets[i])
            if mode == 'train':
                assert len(samples_per_gpus) == len(stage_epochs)
                self._max_iters = 0
                for samples_per_gpu, epochs in zip(samples_per_gpus, stage_epochs):
                    loader_length = int(math.ceil(dataset_length / samples_per_gpu))
                    self._max_iters += epochs * loader_length
                    cum_epoch_iters += [loader_length] * epochs
                break
        cum_epoch_iters = np.cumsum(cum_epoch_iters)
        # print(stage_epochs, cum_stage_epochs, cum_epoch_iters)

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, self.max_epochs)
        self.call_hook('before_run')

        # iter_loaders = [[IterLoader(x) for x in inner_loaders] for inner_loaders in data_loaders]

        print(self.depth, self.total_depth, self.epoch, self.iter, self.inner_iter)
        print(self.max_epochs, self.max_iters)

        while self.depth < self.total_depth:
            data_loaders = [build_dataloader(
                ds,
                samples_per_gpus[self.depth],
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed) for ds in datasets]
            iter_loaders = [IterLoader(x) for x in data_loaders]
            while self.epoch < cum_stage_epochs[self.depth]:
                self.call_hook('before_train_epoch')

                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    iters = len(iter_loaders[i])
                    fade_point = int((fade_in_percentages[self.depth] / 100)
                                    * stage_epochs[self.depth] * iters)
                    print('fade point: ', fade_point)
                    if isinstance(mode, str):  # self.train()
                        if not hasattr(self, mode):
                            raise ValueError(
                                f'runner has no method named "{mode}" to run an '
                                'epoch')
                        iter_runner = getattr(self, mode)
                    else:
                        raise TypeError(
                            'mode in workflow must be a str, but got {}'.format(
                                type(mode)))

                    for _ in range(epochs):
                        if mode == 'train' and (self.epoch >= cum_stage_epochs[self.depth]
                                or self.iter >= cum_epoch_iters[self.epoch]):
                            break

                        for __ in range(iters):
                            if mode == 'train' and (self.iter >= self.max_iters or
                                    self.inner_iter >= len(data_loaders[i])):
                                break
                            self._alpha = self.ticker / fade_point if self.ticker <= fade_point else 1
                            self._ticker += 1
                            if self.iter % 100 == 0:
                                print('depth: {}, alpha: {}'.format(self.depth, self.alpha))
                                print('iter: {}, inner_iter: {}'.format(self.iter, self.inner_iter))
                            iter_runner(iter_loaders[i], depth=self.depth, alpha=self.alpha, **kwargs)
                self._epoch += 1
                self._inner_iter = 0
                self.call_hook('after_train_epoch')

            del data_loaders
            del iter_loaders
            self._depth += 1
            self._ticker = 1
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
    
    def resume(self,
               checkpoint,
               **kwargs):
        super().resume(checkpoint, **kwargs)
        checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        self._depth = checkpoint['meta']['depth']
        self._ticker = checkpoint['meta']['ticker']
        self._inner_iter = checkpoint['meta']['inner_iter']
    
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.
        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch, iter=self.iter + 1, 
                inner_iter=self.inner_iter + 1, depth=self.depth,
                ticker=self.ticker)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch, iter=self.iter + 1, 
                inner_iter=self.inner_iter + 1, depth=self.depth,
                ticker=self.ticker)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    # def register_lr_hook(self, lr_config):
    #     if lr_config is None:
    #         return
    #     super().register_lr_hook(lr_config)


    def register_training_hooks(self,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.
        Default hooks include:
        - LrUpdaterHook
        - MomentumUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        # self.register_lr_hook(lr_config)
        # self.register_momentum_hook(momentum_config)
        # self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)