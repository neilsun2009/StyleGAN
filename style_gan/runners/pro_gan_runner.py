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

class ProGANRunner:

    def __init__(self, model, optimizer=None, logger=None,
            meta=None, work_dir='', **kwargs):
        self._depth = 0
        self._alpha = 0
        self.model = model
        self._total_depth = self.model.module.total_depth
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.meta = meta
        self._rank, self._world_size = get_dist_info()
        self.log_buffer = LogBuffer()

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

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
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters
    
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

    def run(self, data_loaders, workflow, stage_epochs, fade_in_percentages, **kwargs):

        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        assert self.total_depth <= len(stage_epochs)
        assert self.total_depth <= len(fade_in_percentages)

        stage_epochs = stage_epochs[:self.total_depth]
        fade_in_percentages = fade_in_percentages[:self.total_depth]
        stage_epochs = np.cumsum(stage_epochs)
        self._max_epochs = sum(stage_epochs)
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, self.max_epochs)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        print(self.depth, self.total_depth, self.epoch, self.iter, self.inner_iter)

        while self.depth < self.total_depth:
            while self.epoch < stage_epochs[self.depth]:
                self.call_hook('before_train_epoch')

                self._inner_iter = 0
                for i, flow in enumerate(workflow):
                    ticker = 1
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
                        if mode == 'train' and self.epoch >= stage_epochs[self.depth]:
                            break

                        for __ in range(iters):
                            if mode == 'train' and (self.iter >= self.max_iters or
                                    self.inner_iter >= len(data_loaders[i])):
                                break
                            self._alpha = ticker / fade_point if ticker <= fade_point else 1
                            ticker += 1
                            if self.iter % 100 == 0:
                                print('depth: {}, alpha: {}'.format(self.depth, self.alpha))
                                print('iter: {}, inner_iter: {}'.format(self.iter, self.inner_iter))
                            iter_runner(iter_loaders[i], depth=self.depth, alpha=self.alpha, **kwargs)
                self._epoch += 1
                self.call_hook('after_train_epoch')

            self._depth += 1

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
    
    def resume(self,
               checkpoint,
               **kwargs):
        super().resume(checkpoint, **kwargs)
        self._depth = checkpoint['meta']['depth']
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
            meta = dict(epoch=self.epoch, iter=self.iter + 1, inner_iter=self.inner_iter + 1, depth=self.depth)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch, iter=self.iter + 1, inner_iter=self.inner_iter + 1, depth=self.depth)
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

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg):
        """Register a hook from its cfg.
        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.
        Notes:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        if log_config is None:
            return
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

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