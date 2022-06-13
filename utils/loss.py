import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

def compute_l1_loss(model):
    L1_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    return L1_reg


class Distance_Relu_Loss(nn.Module):

    """Zhao et al., Longitudinal self - supervised learning, Medical Image Analysis (2021)
    https://github.com/ZucksLiu/LSSL"""

    def __init__(self, z_dim, requires_grad=True, name="relu_loss", device='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.weights = torch.nn.Parameter(torch.randn(1, z_dim, dtype=torch.float, requires_grad=requires_grad))
        self.weights.data = self.weights.data.detach() / (self.weights.data.detach().norm() + 1e-10)
        nn.Module.register_parameter(self, 'd_weights', self.weights)

    def forward(self, z1, z2, margin):
        zn12 = (z1 - z2).norm(dim=1)

        self.weights.data = self.weights.data.detach() / (self.weights.data.detach().norm() + 1e-10)

        h1 = F.linear(z1, self.weights)
        h2 = F.linear(z2, self.weights)
        h1 = torch.squeeze(h1, dim=1)
        h2 = torch.squeeze(h2, dim=1)
        ncos12 = (h1 - h2 + margin).squeeze() / (zn12 + 1e-7)

        return ncos12


