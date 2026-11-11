"""Static Pre-allocated Key-Value Cache for Fast Autoregressive Decoding."""

from typing import Tuple
import torch
import torch.nn as nn


class KVCache:
    """Pre-allocated static KV cache storing key and value tensors across all decoder layers."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # CRITICAL: Cấp phát trước bộ đệm tĩnh cho toàn bộ n_layers để tránh re-allocate bộ nhớ GPU
        # mỗi bước sinh token, giúp giảm tối đa độ trễ (latency) khi autoregressive decoding.
        cache_shape = (batch_size, n_heads, max_seq_len, head_dim)
        self.k_cache = [torch.zeros(cache_shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(cache_shape, dtype=dtype, device=device) for _ in range(n_layers)]

    def update(
        self,
        layer_idx: int,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Insert new key/value representations and return the active sequence up to current position."""
        seq_len = k_val.shape[-2]
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(f"Tổng số token ({end_pos}) vượt quá dung lượng max_seq_len ({self.max_seq_len})")

        # CRITICAL: Ghi đè trực tiếp token mới vào slice [start_pos : end_pos] theo chiều thời gian.
        # Thao tác in-place này không phát sinh chi phí cấp phát bộ nhớ mới.
        self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = k_val
        self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = v_val

        # Trả về toàn bộ key/value từ 0 đến end_pos để phục vụ tính attention
        active_k = self.k_cache[layer_idx][:, :, :end_pos, :]
        active_v = self.v_cache[layer_idx][:, :, :end_pos, :]
        return active_k, active_v

    def reset(self) -> None:
        """Reset cache contents to zeros."""
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_()
            v.zero_()


if __name__ == "__main__":
    n_layers, batch_size, n_heads, head_dim, max_seq_len = 6, 2, 8, 48, 512
    cache = KVCache(
        n_layers=n_layers,
        batch_size=batch_size,
        n_heads=n_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    )

    # 1. Prefill step: nạp prompt 10 tokens
    prompt_len = 10
    k_prompt = torch.randn(batch_size, n_heads, prompt_len, head_dim)
    v_prompt = torch.randn(batch_size, n_heads, prompt_len, head_dim)

    k_out, v_out = cache.update(layer_idx=0, k_val=k_prompt, v_val=v_prompt, start_pos=0)
    assert k_out.shape == (batch_size, n_heads, prompt_len, head_dim)
    assert torch.equal(k_out, k_prompt), "Lỗi nạp prefill vào KVCache!"

    # 2. Decode step: nạp 1 token tiếp theo tại vị trí start_pos=10
    k_next = torch.randn(batch_size, n_heads, 1, head_dim)
    v_next = torch.randn(batch_size, n_heads, 1, head_dim)

    k_out, v_out = cache.update(layer_idx=0, k_val=k_next, v_val=v_next, start_pos=10)
    assert k_out.shape == (batch_size, n_heads, 11, head_dim)
    assert torch.equal(k_out[:, :, 10:11, :], k_next), "Lỗi cập nhật decode token tại index 10!"

    print("KVCache smoke test passed successfully!")
