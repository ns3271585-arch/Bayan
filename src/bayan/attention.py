"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math

import torch
from torch import nn


def attention(q, k, v, mask=None):
    """Compute scaled dot-product attention."""

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) / math.sqrt(q.size(-1))

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, v)

    return output


class MultiHeadAttention(nn.Module):
    """Simple multi-head self-attention implementation."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, and V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Apply multi-head self-attention.

        x shape:
            [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = x.shape

        # Project input into Q, K, and V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split d_model into multiple heads
        # [B, T, D] -> [B, H, T, head_dim]
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention for all heads
        attended = attention(
            q,
            k,
            v,
            mask=mask
        )

        # Merge the heads back together
        # [B, H, T, head_dim] -> [B, T, D]
        attended = (
            attended
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.out_proj(attended)

        return output