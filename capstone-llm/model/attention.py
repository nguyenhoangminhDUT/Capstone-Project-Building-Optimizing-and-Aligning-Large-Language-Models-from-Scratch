"""Causal Multi-Head Self-Attention with Rotary Position Embedding (RoPE)."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.rope import RoPE, apply_rope
except ImportError:
    from rope import RoPE, apply_rope


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention supporting RoPE and causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask tam giác dưới (True cho các vị trí được phép attend)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RoPE] = None,
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Chiếu ma trận và tách head: (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope(x, start_pos=start_pos)
            q, k = apply_rope(q, k, cos, sin)

        # Cập nhật và truy xuất toàn bộ lịch sử Key/Value nếu sử dụng KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx=layer_idx, k_val=k, v_val=v, start_pos=start_pos)

        # Tính scaled dot-product attention scores: (B, n_heads, T, seq_len_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # CRITICAL: Khi T > 1 (prefill), cần áp dụng causal mask tam giác để không nhìn trước tương lai.
        # Khi T == 1 (autoregressive decode), token mới chỉ attend về quá khứ đã được lưu trong cache,
        # do đó không cần mask.
        if T > 1:
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Nhân ma trận với Value: (B, n_heads, T, head_dim)
        out = torch.matmul(attn_weights, v)

        # CRITICAL: 3. Phải gọi contiguous() trước .view(). Phép transpose(1, 2) làm thay đổi
        # stride của tensor khiến dữ liệu không còn nằm liên tục (contiguous) trên bộ nhớ RAM/VRAM.
        # Nếu gọi .view() trực tiếp sẽ phát sinh lỗi RuntimeError.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


if __name__ == "__main__":
    d_model, n_heads, max_len = 384, 8, 512
    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_len=max_len)
    rope = RoPE(head_dim=d_model // n_heads, max_len=max_len)

    x = torch.randn(2, 32, d_model)
    out = attn(x, rope=rope)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"CausalSelfAttention smoke test passed! Output shape: {out.shape}")

