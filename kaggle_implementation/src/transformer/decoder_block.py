import torch.nn as nn

from kaggle_implementation.src.self_attention.multi_head_attention import MultiHeadAttention
from kaggle_implementation.src.transformer.transformer_block import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention
        Returns:
           out: output of transformer block

        """

        # we need to pass mask only to fst attention
        attention = self.attention(x, x, x, mask=mask)  # 32x10x512
        value = self.dropout(self.norm(attention + x))

        out = self.transformer_block(key, query, value)

        return out
