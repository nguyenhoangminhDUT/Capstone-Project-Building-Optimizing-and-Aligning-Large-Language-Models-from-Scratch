"""Static Pre-allocated Key-Value Cache for Autoregressive Generation."""

import sys
from pathlib import Path
from typing import Tuple
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.attention import CausalSelfAttention
from model.rope import RoPE


class KVCache:
    """Pre-allocated static Key-Value Cache for efficient single-layer inference."""

    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.device = device
        self.dtype = dtype

        # CRITICAL: Tiền cấp phát tensor cố định max_len thay vì dùng torch.cat() sau mỗi token.
        # Nếu dùng torch.cat(), mỗi bước sinh token sẽ phải cấp phát một vùng nhớ mới và sao chép
        # toàn bộ lịch sử KV trước đó, dẫn đến độ phức tạp O(n^2) bộ nhớ và thời gian (memory allocation overhead).
        cache_shape = (batch_size, n_heads, max_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.current_len: int = 0

    def append(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new key and value representations into cache at current_len and return active window."""
        seq_len = new_k.shape[-2]
        end_pos = self.current_len + seq_len

        if end_pos > self.max_len:
            raise ValueError(
                f"Độ dài vượt quá dung lượng cache tối đa: {end_pos} > {self.max_len}"
            )

        # Ghi new_k, new_v vào vị trí current_len
        self.k_cache[:, :, self.current_len : end_pos, :] = new_k
        self.v_cache[:, :, self.current_len : end_pos, :] = new_v

        # Tăng current_len
        self.current_len = end_pos

        # Trả về k_full và v_full từ 0 đến current_len
        k_full = self.k_cache[:, :, : self.current_len, :]
        v_full = self.v_cache[:, :, : self.current_len, :]
        return k_full, v_full

    def reset(self) -> None:
        """Reset cache tensors to zeros and reset sequence length pointer."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0


if __name__ == "__main__":
    # Smoke test: So sánh output giữa không cache và có cache theo từng bước
    torch.manual_seed(42)
    device = torch.device("cpu")
    d_model, n_heads, head_dim, max_len = 64, 4, 16, 64

    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_len=max_len).to(device)
    attn.eval()
    rope = RoPE(head_dim=head_dim, max_len=max_len).to(device)

    # 1. Forward toàn bộ chuỗi x1 shape (1, 5, 64) không có cache
    x1 = torch.randn(1, 5, d_model, device=device)
    with torch.no_grad():
        out1 = attn(x1, rope=rope)

    # 2. Chạy lại với KV Cache cho từng token từ 0..4
    cache = KVCache(
        batch_size=1,
        n_heads=n_heads,
        head_dim=head_dim,
        max_len=max_len,
        device=device,
        dtype=x1.dtype,
    )
    cache.reset()

    outs = []
    with torch.no_grad():
        for t in range(5):
            x_t = x1[:, t : t + 1, :]  # shape (1, 1, 64)
            out_t = attn(x_t, rope=rope, kv_cache=cache)
            outs.append(out_t)

    out_cached = torch.cat(outs, dim=1)  # shape (1, 5, 64)

    # 3. So sánh out1 với out_cached
    max_diff = (out1 - out_cached).abs().max().item()
    print(f"Max diff vs non-cached output: {max_diff:.8e}")
    assert max_diff < 1e-5, f"Diff quá lớn: {max_diff}"
    print("KV Cache OK")
