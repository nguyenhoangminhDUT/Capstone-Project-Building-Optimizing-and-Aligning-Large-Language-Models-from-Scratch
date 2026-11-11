"""Causal Multi-Head Self-Attention with Rotary Position Embedding (RoPE) and KV Cache."""

import math
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.rope import RoPE, apply_rope
except ImportError:
    from rope import RoPE, apply_rope


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention supporting RoPE, causal masking, and KV Cache."""

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
        kv_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Chiếu ma trận cho token đầu vào: (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            # CRITICAL: RoPE offset — vị trí RoPE phải là vị trí tuyệt đối trong chuỗi
            # (offset = kv_cache.current_len), không phải vị trí cục bộ trong token mới.
            # Đây là lỗi phổ biến nhất khiến model sinh sai lệch khi dùng KV Cache.
            offset = kv_cache.current_len
            if rope is not None:
                cos, sin = rope(x, start_pos=offset)
                q, k = apply_rope(q, k, cos, sin)

            # Ghi k, v mới vào cache và lấy toàn bộ k_full, v_full tính tới hiện tại
            k, v = kv_cache.append(k, v)

            # Tính scaled dot-product attention: (B, n_heads, T, total_len)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # CRITICAL: Khi T == 1 (bước decode autoregressive), token hiện tại ở vị trí offset
            # được phép attend đến toàn bộ quá khứ trong cache (các vị trí 0..offset). Do đó không cần mask.
            # Khi T > 1 (prefill cùng cache), chỉ áp dụng mask cho phần tương lai cục bộ.
            if T > 1:
                mask = self.causal_mask[:T, :T]
                if offset > 0:
                    past_mask = torch.ones(T, offset, dtype=torch.bool, device=x.device)
                    full_mask = torch.cat((past_mask, mask), dim=-1)
                else:
                    full_mask = mask
                scores = scores.masked_fill(~full_mask, float("-inf"))
        else:
            # Không dùng KV Cache: Tính attention đầy đủ trên toàn bộ chuỗi T
            if rope is not None:
                cos, sin = rope(x, start_pos=0)
                q, k = apply_rope(q, k, cos, sin)

            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # CRITICAL: 1. Cắt causal_mask theo độ dài T thực tế
            # CRITICAL: 2. Dùng float("-inf") thay vì -1e9 để tránh tràn số fp16/bf16
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Nhân trọng số attention với Value: (B, n_heads, T, head_dim)
        out = torch.matmul(attn_weights, v)

        # CRITICAL: 3. Phải gọi contiguous() trước .view() sau phép transpose
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
