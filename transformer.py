import torch
import torch.nn as nn
from model.embedding import TokenEmbedding
from model.block import TransformerBlock
from model.rmsnorm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_heads, d_model, num_layers):
        """
        num_layers 代表 Transformer blocks 的数量。
        """
        super().__init__()
        
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(num_heads, d_model)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Parameter(
            torch.randn(d_model, vocab_size) * (d_model ** -0.5)
        )
        
    def forward(self, token_ids):
        # [B, T]
        hidden = self.embedding(token_ids)
        # [B, T, D]
        for block in self.blocks:
            hidden = block(hidden)
        # [B, T, D]
        hidden = self.final_norm(hidden)
        logits = hidden @ self.lm_head
        # [B, T, V]
        return logits
        
