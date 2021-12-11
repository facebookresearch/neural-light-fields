import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class MAELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class ComplexMSELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(torch.real(inputs), torch.real(targets))
        loss += self.loss(torch.imag(inputs), torch.imag(targets))
        return loss

class ComplexMAELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(torch.real(inputs), torch.real(targets))
        loss += self.loss(torch.imag(inputs), torch.imag(targets))
        return loss

class MSETopN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.frac = cfg.frac
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        n = int(self.frac * targets.shape[0])

        idx = torch.argsort(diff, dim=0)

        targets_sorted = torch.gather(targets, 0, idx)
        targets_sorted = targets_sorted[:n]

        inputs_sorted = torch.gather(inputs, 0, idx)
        inputs_sorted = inputs_sorted[:n]

        loss = self.loss(inputs_sorted, targets_sorted)
        return loss

class MAETopN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.frac = cfg.frac
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        n = int(self.frac * targets.shape[0])

        idx = torch.argsort(diff, dim=0)

        targets_sorted = torch.gather(targets, 0, idx)
        targets_sorted = targets_sorted[:n]

        inputs_sorted = torch.gather(inputs, 0, idx)
        inputs_sorted = inputs_sorted[:n]

        loss = self.loss(inputs_sorted, targets_sorted)
        return loss


loss_dict = {
    'mse': MSELoss,
    'mae': MAELoss,
    'complex_mse': ComplexMSELoss,
    'complex_mae': ComplexMAELoss,
    'mse_top_n': MSETopN,
    'mae_top_n': MAETopN,
}
