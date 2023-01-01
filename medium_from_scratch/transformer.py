import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as functional


def scaled_dot_product_attention(query: Tensor,  # (batch_size, sequence_length, num_features)
                                 key: Tensor,  # (batch_size, sequence_length, num_features)
                                 value: Tensor  # (batch_size, sequence_length, num_features)
                                 ) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))  # (batch_size, sequence_length, sequence_length)
    scale = query.size(-1) ** 0.5
    softmax = functional.softmax(temp / scale, dim=-1)  # (batch_size, sequence_length, sequence_length)
    return softmax.bmm(value)  # (batch_size, sequence_length, num_features)


class AttentionHead(nn.Module):
    def __init__(self,
                 dim_in: int,  # representation dimension (input)
                 dim_q: int,
                 dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)  # from representation to q (query)
        self.k = nn.Linear(dim_in, dim_k)  # from representation to k (key)
        self.v = nn.Linear(dim_in, dim_k)  # from representation to q (query)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_in: int,
                 dim_q: int,
                 dim_k: int):
        super().__init__()
        # Creating the heads
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        # Linear to go from all heads to dim_in
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


def position_encoding(
        seq_len: int,
        dim_model: int,
        device: torch.device) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class Residual(nn.Module):
    def __init__(self,
                 sublayer: nn.Module,
                 dimension: int,
                 dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src
