import torch
import torch.nn as nn

import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, embed_dim, max_len=80):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        po = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.) / embed_dim))
        pe[:, 0::2] = torch.sin(po * div_term)
        pe[:, 1::2] = torch.cos(po * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        l, _, _ = x.shape
        pos = self.pe[:l, :].unsqueeze(1)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, embed_dim, max_len=80, padding_idx=0):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim, padding_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        l, _, _ = x.shape
        idx = torch.arange(l, device=x.device)
        pos = self.pos_embed(idx).unsqueeze(1)
        return pos

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)