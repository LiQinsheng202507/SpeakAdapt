import torch
import torch.nn as nn

class ContextAwareAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(ContextAwareAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        """
        x: (batch, time, dim)
        context: (batch, time, dim) or None (self-attention)
        """
        query = key = value = x if context is None else context
        attn_output, _ = self.attn(x, key, value)
        x = x + self.dropout(attn_output)
        return self.norm(x)
