# Copyright (c) Open-MMLab. All rights reserved.
import os

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
import time
import numpy as np
from mmcv.image import imwrite

@HOOKS.register_module()
class SaveImageHook(Hook):
    """Save images periodically.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 out_dir='./output_images',
                 save_num=10,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.out_dir = os.path.join(out_dir, str(int(time.time())))
        os.makedirs(self.out_dir, exist_ok=True)
        self.save_num = save_num
        self.args = kwargs

    def _save_images(self, runner):
        pseudo_input = np.array([[]]*self.save_num)
        result = runner.model(pseudo_input, runner.depth, runner.alpha, return_loss=False).cpu().detach()
        for i in range(self.save_num):
            imgname = '{:03d}/{:03d}_{:03d}.jpg'.format(runner.epoch, runner.inner_iter, i)
            save_path = os.path.join(self.out_dir, imgname)
            os.makedirs(os.path.dirname(save_path) , exist_ok=True)
            imwrite(result[i].numpy().transpose(1, 2, 0), save_path)
        del result

    @master_only
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        runner.logger.info(f'Saving images at {runner.epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self._save_images(runner)

    @master_only
    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        runner.logger.info(
            f'Saving images at {runner.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        
        self._save_images(runner)