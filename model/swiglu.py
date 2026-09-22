# SwiGLU
# SwiGLU(x) = (SiLU(xW1)·xW3)W2

import torch
import torch.nn as nn

def silu(x):
    # input: [B, T, H]
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = 8 * d_model // 3
        
        self.gate_weight = nn.Parameter(
            torch.randn(self.d_model, self.hidden_dim) * (self.d_model ** -0.5)
        )
        self.up_weight = nn.Parameter(
            torch.randn(self.d_model, self.hidden_dim) * (self.d_model ** -0.5)
        )
        self.down_weight = nn.Parameter(
            torch.randn(self.hidden_dim, self.d_model) * (self.hidden_dim ** -0.5)
        )
        
    def forward(self, x):
        gate = silu(x @ self.gate_weight)
        value = x @ self.up_weight
        
        hidden = gate * value
        output = hidden @ self.down_weight
        return output
