import inspect

import mmcv
import numpy as np
from numpy import random

from ..builder import PIPELINES
from torchvision.transforms import ToTensor, Normalize as V_Normalize, Compose, Resize, RandomHorizontalFlip


@PIPELINES.register_module()
class EasyResize(object):
    """Resize images to square images.

    """

    def __init__(self, img_scale):
        self.img_scale = img_scale

    
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results[key], (self.img_scale, self.img_scale), return_scale=True)
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        # if 'flip' not in results:
        #     flip = True if np.random.rand() < self.flip_ratio else False
        #     results['flip'] = flip
        # if 'flip_direction' not in results:
        #     results['flip_direction'] = self.direction
        # if results['flip']:
        #     # flip image
        for key in results.get('img_fields', ['img']):
            # results[key] = mmcv.imflip(
            #     results[key], direction=results['flip_direction'])
            results[key] = RandomHorizontalFlip()(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'

@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            # print('normalize input {} ~ {}'.format(np.min(results[key]), np.max(results[key])))
            # results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
            #                                 self.to_rgb)
            results[key] = V_Normalize(mean=self.mean, std=self.std)(results[key])
            # print('normalize output {} ~ {}'.format(np.min(results[key]), np.max(results[key])))
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
