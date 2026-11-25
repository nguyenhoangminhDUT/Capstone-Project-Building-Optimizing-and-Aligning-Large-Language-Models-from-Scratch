"""Autoregressive Transformer Decoder-Only LLM (~100M Ultimate Scale-Up)."""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.rmsnorm import RMSNorm
    from model.rope import RoPE
    from model.attention import CausalSelfAttention
    from model.swiglu import SwiGLU
except ImportError:
    from rmsnorm import RMSNorm
    from rope import RoPE
    from attention import CausalSelfAttention
    from swiglu import SwiGLU


class Block(nn.Module):
    """Transformer decoder block with Pre-LayerNorm architecture."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        max_len: int = 1024,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model=d_model, eps=eps)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(d_model=d_model, eps=eps)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        kv_cache: Optional[object] = None,
    ) -> torch.Tensor:
        # Pre-norm: x = x + attn(norm1(x), rope, kv_cache); x = x + ffn(norm2(x))
        x = x + self.attn(self.norm1(x), rope=rope, kv_cache=kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    Autoregressive Transformer Decoder-Only Language Model.
    Cấu hình Ultimate Scale-Up ~100M tham số (Tương đương GPT-2 Small):
    - vocab_size = 32000
    - d_model = 768, n_layers = 12, n_heads = 12 (head_dim = 64)
    - d_ff = 2048 (SwiGLU tự động tính theo tỷ lệ 8/3 d_model)
    - max_seq_len = 1024
    - Weight Tying giữa Embedding và LM Head
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 1024,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Tự động tính d_ff cho SwiGLU: khoảng 8/3 d_model (khoảng 2048 với d_model=768)
        if d_ff is None:
            d_ff = int(2 * 4 * d_model / 3)
            # Làm tròn về bội số của 64 để tối ưu hóa Tensor Cores
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        self.embedding = nn.Embedding(vocab_size, d_model)

        # RoPE được tạo 1 lần trong __init__ và truyền xuống từng Block (hỗ trợ tối thiểu 2200 vị trí cho benchmark L=2048)
        head_dim = d_model // n_heads
        self.rope = RoPE(head_dim=head_dim, max_len=max(2200, max_seq_len))

        self.layers = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    max_len=max_seq_len,
                    eps=eps,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model=d_model, eps=eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Khởi tạo trọng số chuẩn GPT-2 cho Linear và Embedding
        self.apply(self._init_weights)

        # Weight tying giữa token embedding và lm_head
        self.lm_head.weight = self.embedding.weight

        # In thông tin cấu hình và số lượng tham số chính xác
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Transformer] Khởi tạo mô hình 100M (GPT-2 Small Scale):")
        print(f"  - Vocab size  : {self.vocab_size:,}")
        print(f"  - d_model     : {self.d_model} | n_layers: {len(self.layers)} | n_heads: {n_heads} (head_dim: {head_dim})")
        print(f"  - d_ff        : {self.d_ff} (SwiGLU)")
        print(f"  - max_seq_len : {self.max_seq_len}")
        print(f"  - Tổng số tham số (Weight-Tied): {total_params:,} (~{total_params / 1e6:.1f}M)")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with GPT-2 standard normal distribution (std=0.02) and zero bias."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[object]] = None,
    ) -> torch.Tensor:
        """Forward pass mapping input token IDs to next-token logits (B, T, vocab_size)."""
        x = self.embedding(input_ids)  # (B, T, d_model)

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, rope=self.rope, kv_cache=layer_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    # Khởi tạo mô hình 100M chuẩn
    model = Transformer()

    # Smoke test forward pass
    dummy_x = torch.randint(0, model.vocab_size, (2, 16))
    dummy_logits = model(dummy_x)
    assert dummy_logits.shape == (2, 16, model.vocab_size), f"Wrong shape: {dummy_logits.shape}"
    print(f"Forward output shape: {dummy_logits.shape} (OK)")
