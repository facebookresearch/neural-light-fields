import numpy as np
import torch
from torch import nn


class LeakyReLU(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'a' in cfg and not isinstance(cfg, str):
            self.a = cfg.a
        else:
            self.a = 0.01

        self.act = nn.LeakyReLU(self.a, True)

    def forward(self, x):
        return self.act(x)


class ReLU(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(x)


class Sigmoid(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x)


class Softmax(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Softmax()

    def forward(self, x):
        return self.act(x)


class Tanh(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(x)


class Identity(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Identity()

    def forward(self, x):
        return self.act(x)


class L1Norm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=1, dim=-1) * x.shape[-1]


class Probs(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(torch.abs(x), p=1, dim=-1)

class MatrixNorm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * np.sqrt(4)


class L2Norm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * np.sqrt(x.shape[-1])

class Zero(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class Alpha(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return 1.0 - torch.exp(-torch.relu(x))


class Gaussian(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'sigma' in cfg and not isinstance(cfg, str):
            self.sigma = cfg.sigma
        else:
            self.sigma = 0.05

    def forward(self, x):
        return torch.exp(-0.5 * torch.square(x / self.sigma))


activation_map = {
    'alpha': Alpha,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': Tanh,
    'identity': Identity,
    'probs': Probs,
    'l1_norm': L1Norm,
    'l2_norm': L2Norm,
    'matrix_norm': MatrixNorm,
    'zero': Zero,
    'gaussian': Gaussian,
    'leaky_relu': LeakyReLU,
    'relu': ReLU,
}

def get_activation(cfg):
    if isinstance(cfg, str):
        return activation_map[cfg](cfg)
    else:
        return activation_map[cfg.type](cfg)
