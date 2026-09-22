import torch
import torch.nn as nn
from rope import apply_rope

class Attention(nn.Module):
    def __init__(self, heads, d_model=32):
        super().__init__()
        
        assert d_model % heads == 0
        
        self.D = d_model
        self.H = heads
        self.Dh = d_model // heads
        
        assert self.Dh % 2 == 0
        
        # [Q, K, V] = X @ W_QKV
        self.W_QKV = nn.Parameter(
            torch.randn(d_model, d_model*3)
        )
        
        self.W_O = nn.Parameter(
            torch.randn(d_model, d_model)
        )
    
    def forward(self, X: torch.Tensor):
        QKV = X @ self.W_QKV # shape: [B, T, 3*D]
        
        Q, K, V = QKV.split(self.D, dim=-1)
        
        B, T, _ = Q.shape
        Q = Q.reshape(B, T, self.H, self.Dh)
        K = K.reshape(B, T, self.H, self.Dh)
        V = V.reshape(B, T, self.H, self.Dh)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # shape: [B, H, T, Dh]

        Q, K = apply_rope(Q, K) # RoPE
        S = Q @ K.transpose(-2, -1) # [B, H, T, T]
        S = S / (self.Dh ** 0.5)
        
        M = torch.triu(
            torch.full(
                (T, T),
                float("-inf"),
                device=S.device,
                dtype=S.dtype
            ),
            diagonal=1,
        )
        S = S + M
        
        row_max = S.max(dim=-1, keepdim=True).values
        S = S - row_max
        S = torch.exp(S)
        denominator = S.sum(dim=-1, keepdim=True)
        A = S / denominator
        
        O = A @ V # [B, H, T, Dh]
        O = O.transpose(1, 2) # [B, T, H, Dh]
        O = O.reshape(B, T, self.D)
        Y = O @ self.W_O # [B, T, D]
        return Y
        