"""Rotary Position Embedding (RoPE) with half-split style."""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    d_2 = x.shape[-1] // 2
    x1 = x[..., :d_2]
    x2 = x[..., d_2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors."""
    # CRITICAL: cos và sin có shape (seq_len, head_dim).
    # Trong mô hình Transformer, q và k có shape chuẩn là (batch_size, n_heads, seq_len, head_dim).
    # Để nhân element-wise chính xác, ta phải broadcast cos và sin thành
    # (1, 1, seq_len, head_dim) bằng cách thêm 2 chiều unsqueeze ở đầu.
    # Với q và k có số chiều khác (2D hoặc 3D khi test), vòng lặp unsqueeze đảm bảo
    # cos và sin khớp số chiều với q và k trước khi nhân.
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RoPE(nn.Module):
    """Rotary Position Embedding generator using half-split frequency layout."""

    def __init__(
        self,
        head_dim: int,
        max_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        # CRITICAL: Đăng ký inv_freq làm buffer (không có gradient, không phải nn.Parameter)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin positional tensors for sequence length of x starting from start_pos."""
        seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[0]
        # CRITICAL: Khi giải mã autoregressive cùng KV Cache, token mới sinh tại vị trí t cần được
        # áp dụng RoPE với vị trí t (start_pos = t), thay vì luôn tính từ vị trí 0.
        t = torch.arange(start_pos, start_pos + seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs có shape (seq_len, head_dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Concat 2 lần theo chiều cuối để khớp half-split: shape (seq_len, head_dim)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


if __name__ == "__main__":
    head_dim = 48
    rope = RoPE(head_dim=head_dim, max_len=2048, base=10000.0)

    # a) apply_rope tại vị trí 0 phải trả về đúng q ban đầu (diff < 1e-6)
    q0 = torch.randn(1, 1, 1, head_dim)
    k0 = torch.randn(1, 1, 1, head_dim)
    cos0, sin0 = rope(q0)  # seq_len = 1 (vị trí 0)
    q_rot0, k_rot0 = apply_rope(q0, k0, cos0, sin0)
    diff_0 = (q_rot0 - q0).abs().max().item()
    print(f"a) Diff at position 0: {diff_0:.8e}")
    assert diff_0 < 1e-6, f"Position 0 test failed with diff: {diff_0}"

    # b) Kiểm tra tính chất tương đối: <RoPE(q, i), RoPE(k, j)> với (5, 3) và (6, 4)
    # Vì khoảng cách tương đối i - j = 2 là như nhau, tích vô hướng phải bằng nhau (diff < 1e-5)
    dummy_seq = torch.empty(16, head_dim)
    cos_table, sin_table = rope(dummy_seq)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Vị trí (i, j) = (5, 3)
    q5, _ = apply_rope(q, q, cos_table[5:6], sin_table[5:6])
    _, k3 = apply_rope(k, k, cos_table[3:4], sin_table[3:4])
    dot_53 = (q5 * k3).sum().item()

    # Vị trí (i, j) = (6, 4)
    q6, _ = apply_rope(q, q, cos_table[6:7], sin_table[6:7])
    _, k4 = apply_rope(k, k, cos_table[4:5], sin_table[4:5])
    dot_64 = (q6 * k4).sum().item()

    diff_shift = abs(dot_53 - dot_64)
    print(f"b) Dot(5, 3) = {dot_53:.6f}, Dot(6, 4) = {dot_64:.6f}, Diff = {diff_shift:.8e}")
    assert diff_shift < 1e-5, f"Relative shift test failed with diff: {diff_shift}"
    print("RoPE smoke test passed successfully!")

