import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import Attention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model=32):
        super().__init__()
        
        assert d_model % num_heads == 0

        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attention = Attention(num_heads, d_model)
        self.feed_forward = SwiGLU(d_model)
        
    def forward(self, x):
        # Pre-Norm
        hidden = x + self.attention(self.attention_norm(x))
        output = hidden + self.feed_forward(self.ffn_norm(hidden))
        
        return output
