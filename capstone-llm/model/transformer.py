"""Autoregressive Transformer Decoder-Only LLM."""

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
        d_ff: int = 1024,
        max_len: int = 512,
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
    """Autoregressive Transformer Decoder-Only Language Model."""

    def __init__(
        self,
        vocab_size: int = 12000,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)

        # RoPE được tạo 1 lần trong __init__ và truyền xuống từng Block
        head_dim = d_model // n_heads
        self.rope = RoPE(head_dim=head_dim, max_len=max_seq_len)

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
    # Khởi tạo mô hình theo đúng cấu hình spec
    model = Transformer(
        vocab_size=12000,
        d_model=384,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=512,
        eps=1e-6,
        dropout=0.0,
    )

    # Input_ids random shape (2, 32)
    input_ids = torch.randint(0, 12000, (2, 32))
    logits = model(input_ids)

    print(f"logits.shape: {logits.shape}")
    print(f"logits.mean(): {logits.mean().item():.6f}")
    print(f"logits.std(): {logits.std().item():.6f}")
    print(f"logits.max(): {logits.max().item():.6f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {total_params:,}")

    # Tính cross-entropy với labels = input_ids dịch phải 1 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, model.vocab_size),
        shift_labels.view(-1),
        reduction="mean",
    )

    loss_val = loss.item()
    expected_loss = math.log(model.vocab_size)
    print(f"Cross-entropy loss: {loss_val:.4f} (Kỳ vọng ~ {expected_loss:.4f})")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert abs(loss_val - expected_loss) < 0.5, f"Loss lệch quá xa kỳ vọng: {loss_val}"
    print("OK")
