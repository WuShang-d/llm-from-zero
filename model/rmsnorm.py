import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        """Normalize the final dimension of x: [B, T, D]."""
        # [B, T, D] -> [B, T, 1]
        rms = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(rms + self.eps)

        # [B, T, D] / [B, T, 1] -> [B, T, D]
        # [B, T, D] * [D]       -> [B, T, D]
        return (x / rms) * self.weight
