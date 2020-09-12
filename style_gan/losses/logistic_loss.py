import torch
import torch.nn as nn
import numpy as np
from .builder import LOSSES

@LOSSES.register_module()
class LogisticLossGen(nn.Module):

    def __init__(self):
        super().__init__()
        # self.softplus = nn.Softplus()

    def forward(self, disc_fake_output):
        print('gen loss {}'.format(nn.Softplus()(-disc_fake_output).mean()))
        return torch.mean(nn.Softplus()(-disc_fake_output))

@LOSSES.register_module()
class LogisticLossDisc(nn.Module):

    def __init__(self, lambda_r1=10.):
        super().__init__()
        self.lambda_r1 = lambda_r1

    def forward(self, disc_real_output, disc_fake_output, r1_penalty=None):
        print('real output {} ~ {}'.format(torch.min(disc_real_output), torch.max(disc_real_output)))
        print('fake output {} ~ {}'.format(torch.min(disc_fake_output), torch.max(disc_fake_output)))
        loss_real = torch.mean(nn.Softplus()(-disc_real_output))
        loss_fake = torch.mean(nn.Softplus()(disc_fake_output))
        # loss_real_drift = disc_real_output.pow(2).mean()
        general_loss = loss_real + loss_fake # + loss_real_drift * self.lambda_drift
        if r1_penalty is not None:
            general_loss += r1_penalty * self.lambda_r1 * 0.5
        print('real loss {}, fake loss {}, r1 loss {}'.format(loss_real,
            loss_fake, r1_penalty))
        return  general_loss
    