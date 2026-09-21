"""
    linear:
    input shape : [B, T, Din]
    output shape: [B, T, Dout]
"""

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, Din, Dout):
        super().__init__()

        # weight: [Din, Dout]
        self.weight = nn.Parameter(
            torch.randn(Din, Dout) * (Din ** -0.5)
        )

        # bias: [Dout]
        self.bias = nn.Parameter(
            torch.zeros(Dout)
        )

    def forward(self, x):
        """
        x:      [B, T, Din]
        weight: [Din, Dout]
        bias:   [Dout]
        return: [B, T, Dout]
        """
        return x @ self.weight + self.bias
