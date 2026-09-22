import torch
import torch.nn as nn
from .rope import apply_rope

class Attention(nn.Module):
    def __init__(self, num_heads, d_model=32):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim % 2 == 0
        
        # [Q, K, V] = X @ W_QKV
        self.qkv_weight = nn.Parameter(
            torch.randn(d_model, d_model * 3) * (d_model ** -0.5)
        )
        
        self.output_weight = nn.Parameter(
            torch.randn(d_model, d_model) * (d_model ** -0.5)
        )
    
    def forward(self, x: torch.Tensor):
        qkv = x @ self.qkv_weight  # shape: [B, T, 3*D]
        
        query, key, value = qkv.split(self.d_model, dim=-1)
        
        batch_size, sequence_length, _ = query.shape
        query = query.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = key.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        value = value.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # shape: [B, H, T, Dh]

        query, key = apply_rope(query, key)
        scores = query @ key.transpose(-2, -1)  # [B, H, T, T]
        scores = scores / (self.head_dim ** 0.5)
        
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=scores.device,
                dtype=scores.dtype,
            ),
            diagonal=1,
        )
        scores = scores + causal_mask
        
        row_max = scores.max(dim=-1, keepdim=True).values
        shifted_scores = scores - row_max
        exp_scores = torch.exp(shifted_scores)
        denominator = exp_scores.sum(dim=-1, keepdim=True)
        attention_weights = exp_scores / denominator
        
        output = attention_weights @ value  # [B, H, T, Dh]
        output = output.transpose(1, 2)  # [B, T, H, Dh]
        output = output.reshape(batch_size, sequence_length, self.d_model)
        output = output @ self.output_weight  # [B, T, D]
        return output
        
