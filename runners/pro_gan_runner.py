from mmcv.runner import IterBasedRunner
import mmcv
from mmcv.runner.utils import get_host_info
from mmcv.runner.checkpoint import save_checkpoint
import time
from os import path as osp

class ProGANRunner(IterBasedRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total_depth = self.model.total_depth
        self._depth = 0

    @property
    def total_depth(self):
        return self._total_depth

    @property
    def depth(self):
        return self._depth

    def run(self, data_loaders, workflow, stage_epochs, fade_in_percentages, **kwargs):

        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        assert self.total_depth <= len(stage_epochs)
        assert self.total_depth <= len(fade_in_percentages)

        stage_epochs = stage_epochs[:self.total_depth]
        fade_in_percentages = fade_in_percentages[:self.total_depth]
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

        while self.depth < self.total_depth:
            self._inner_iter = 0
            while self.epoch < stage_epochs[self.depth]:
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    iters = len(data_loaders[i])
                    fade_point = int((fade_in_percentages[self.depth] / 100)
                                    * stage_epochs[self.depth] * iters)
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

                        for _ in range(iters):
                            if mode == 'train' and (self.iter >= self.max_iters or
                                    self.inner_iter >= len(data_loaders[i])):
                                break
                            ticker = self.inner_iter + 1
                            alpha = ticker / fade_point if ticker <= fade_point else 1
                            iter_runner(data_loaders[i], depth=self.depth, alpha=alpha, **kwargs)
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