import torch
import torch.nn as nn
import numpy as np
from .builder import LOSSES

@LOSSES.register_module()
class LogisticLossGen(nn.Module):

    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, disc_fake_output):
        return self.softplus(-disc_fake_output).mean()

@LOSSES.register_module()
class LogisticLossDisc(nn.Module):

    def __init__(self, lambda_r1=10):
        super().__init__()
        self.lambda_r1 = lambda_r1
        self.softplus = nn.Softplus()

    def forward(self, disc_real_output, disc_fake_output, gradient_penalty=None):
        loss_real = self.softplus(disc_real_output).mean().mul(-1)
        loss_fake = self.softplus(disc_fake_output).mean()
        # loss_real_drift = disc_real_output.pow(2).mean()
        general_loss = loss_real + loss_fake # + loss_real_drift * self.lambda_drift
        if gradient_penalty is not None:
            general_loss += gradient_penalty * self.lambda_r1 * 0.5
        return  general_loss
    