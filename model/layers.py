"""
    linear:
    input shape : [B, T, d_in]
    output shape: [B, T, d_out]
"""

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        # weight: [d_in, d_out]
        self.weight = nn.Parameter(
            torch.randn(d_in, d_out) * (d_in ** -0.5)
        )

        # bias: [d_out]
        self.bias = nn.Parameter(
            torch.zeros(d_out)
        )

    def forward(self, x):
        """
        x:      [B, T, d_in]
        weight: [d_in, d_out]
        bias:   [d_out]
        return: [B, T, d_out]
        """
        return x @ self.weight + self.bias
