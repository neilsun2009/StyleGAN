# Copyright (c) Open-MMLab. All rights reserved.
import os

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
import time
import numpy as np
from mmcv.image import imwrite
import torch

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
        self.fixed_input = torch.randn(save_num, 512).to(torch.cuda.current_device())

    def _save_images(self, runner):
        with torch.no_grad():
            # pseudo_input = np.array([[]]*self.save_num)
            result = runner.model(self.fixed_input, runner.depth, runner.alpha, return_loss=False).cpu().detach()
            print(result.shape, torch.max(result), torch.min(result), flush=True)
            from torchvision.utils import save_image
            from torch.nn.functional import interpolate
            result = interpolate(result, scale_factor=1024//result.shape[1])
            imgname = '{:03d}/{:03d}.jpg'.format(runner.epoch, runner.inner_iter)
            save_path = os.path.join(self.out_dir, imgname)
            os.makedirs(os.path.dirname(save_path) , exist_ok=True)
            save_image(result, save_path, nrow=int(np.sqrt(self.save_num)),
                normalize=True, scale_each=True, pad_value=128, padding=1)
        # for i in range(self.save_num):
        #     os.makedirs(os.path.dirname(save_path) , exist_ok=True)
        #     img = result[i].numpy().transpose(1, 2, 0)
        #     img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #     imwrite(img * 255, save_path)
        del result

    @master_only
    def after_train_epoch(self, runner):
        if (not self.by_epoch) or ((not self.every_n_epochs(runner, self.interval)) and (runner.epoch != 0)):
            return

        runner.logger.info(f'Saving images at {runner.epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self._save_images(runner)

    @master_only
    def after_train_iter(self, runner):
        if self.by_epoch or ((not self.every_n_iters(runner, self.interval)) and (runner.iter != 0)):
            return

        runner.logger.info(
            f'Saving images at {runner.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        
        self._save_images(runner)
    
    # @master_only
    # def before_epoch(self, runner):
    #     runner.logger.info(
    #         f'Images prior to Epoch {runner.epoch + 1}')
    #     if not self.out_dir:
    #         self.out_dir = runner.work_dir
        
    #     self._save_images(runner)