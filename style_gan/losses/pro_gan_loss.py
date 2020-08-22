import torch
import torch.nn as nn
import numpy as np
from .builder import LOSSES

@LOSSES.register_module()
class ProGANLossGen(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disc_fake_output):
        return disc_fake_output.mean().mul(-1)


@LOSSES.register_module()
class ProGANLossDisc(nn.Module):

    def __init__(self, lambda_drift=0.001, lambda_gp=10):
        super().__init__()
        self.lambda_drift = lambda_drift
        self.lambda_gp = lambda_gp

    def forward(self, disc_real_output, disc_fake_output, gradient_penalty=None):
        loss_real = disc_real_output.mean().mul(-1)
        loss_fake = disc_fake_output.mean()
        loss_real_drift = disc_real_output.pow(2).mean()
        general_loss = loss_real + loss_fake + loss_real_drift * self.lambda_drift
        if gradient_penalty is not None:
            general_loss += gradient_penalty * self.lambda_gp
        return  general_loss
            
