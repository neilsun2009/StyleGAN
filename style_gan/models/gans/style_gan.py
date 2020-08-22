from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
from mmcv.utils import print_log
from mmcv.runner import load_checkpoint
from ...utils.logger import get_root_logger
from ..builder import (build_generator, build_discriminator,
    GANS)
from ...losses.builder import build_loss

@GANS.register_module()
class StyleGAN(nn.Module):
    # largely ref mmdetection

    def __init__(self, generator, discriminator, loss_gen, loss_disc,
            resolution=1024, latent_channels=512, train_cfg=None, test_cfg=None,
            pretrained=None):
        super().__init__()
        resolution_log2 = int(math.log(resolution, 2))
        assert resolution == math.pow(2, resolution_log2)
        self.resolution = resolution
        self.total_depth = resolution_log2 - 1
        self.latent_channels = latent_channels
        # network
        self.generator = build_generator(generator, resolution=self.resolution)
        self.discriminator = build_discriminator(discriminator, resolution=self.resolution)
        self.init_weights(pretrained)
        # loss
        self.loss_gen = build_loss(loss_gen)
        self.loss_disc = build_loss(loss_disc)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        # init each part's weight is initialized
        # in their own layers

    def gradient_penalty(self, real_input, fake_input, depth, alpha):
        mix = torch.rand(real_input.size(0),1,1,1).to(real_input.device)
        interpolates = mix * real_input.detach() + (1 - mix) * fake_input.detach()
        interpolates.requires_grad_(True)
        disc_interpolates = self.discriminator(interpolates, depth, alpha)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(real_input.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = gradients.norm(2, dim=1).sub(1.).pow(2.).mean()
        return gradient_penalty
    
    def progressive_down_sampling(self, imgs, depth, alpha):
        down_sample_factor = int(np.power(2, self.total_depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.total_depth - depth)), 0)
        ds_real_samples = nn.AvgPool2d(down_sample_factor)(imgs)

        if depth > 0:
            prior_ds_real_samples = F.interpolate(nn.AvgPool2d(prior_down_sample_factor)(imgs), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def forward_train(self, imgs, depth, alpha, **kwargs):
        losses = dict()
        batch_size = imgs.shape[0]
        imgs = self.progressive_down_sampling(imgs, depth, alpha)
        latent_input = torch.randn(batch_size, self.latent_channels) \
            .to(torch.cuda.current_device())
        gen_img_output = self.generator(latent_input, depth, alpha)
        gen_score_output = self.discriminator(gen_img_output, depth, alpha)

        disc_real_output = self.discriminator(imgs, depth, alpha)
        disc_fake_input = self.generator(latent_input, depth, alpha).detach()
        disc_fake_output = self.discriminator(disc_fake_input, depth, alpha)
        # losses
        gradient_penalty = self.gradient_penalty(imgs, disc_fake_input, depth, alpha)
        losses['gen_loss'] = self.loss_gen(gen_score_output)
        losses['disc_loss'] = self.loss_disc(disc_real_output, disc_fake_output, gradient_penalty=gradient_penalty)
        return losses

    def forward_test(self, imgs, depth, alpha, **kwargs):
        batch_size = imgs.shape[0]
        latent_input = torch.randn(batch_size, self.latent_channels) \
            .to(torch.cuda.current_device())
        gen_output = self.generator(latent_input, depth, alpha)
        return gen_output

    def forward(self, img, depth, alpha, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, depth, alpha, **kwargs)
        else:
            return self.forward_test(img, depth, alpha, **kwargs)

    def train_step(self, data, optimizer, depth, alpha):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. 

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data, depth=depth, alpha=alpha)
        # step
        optimizer['opt_gen'].zero_grad()
        losses['gen_loss'].backward()
        optimizer['opt_gen'].step()
        optimizer['opt_disc'].zero_grad()
        losses['disc_loss'].backward()
        optimizer['opt_disc'].step()

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, depth, alpha, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data, depth=depth, alpha=alpha)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self, img, result):
        pass