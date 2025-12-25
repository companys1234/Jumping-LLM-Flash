import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.scale * x / (norm + self.eps)