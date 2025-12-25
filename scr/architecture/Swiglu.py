import torch
import torch.nn as nn
import torch.nn.functional as F


class SwigLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2       # Swish(x1) * x2

