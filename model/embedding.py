"""
    embedding:
    input shape :  [B, T]
    output shape:  [B, T, D]
"""

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()

        # weight: [V, D]
        self.weight = nn.Parameter(
            torch.randn(vocab_size, d_model) * 0.02
        )

    def forward(self, token_ids):
        """
        token_ids: [B, T]
        return:    [B, T, D]
        """
        return self.weight[token_ids]
