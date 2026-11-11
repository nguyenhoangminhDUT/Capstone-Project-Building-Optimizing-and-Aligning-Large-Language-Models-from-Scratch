"""Root Mean Square Layer Normalization (RMSNorm)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scale parameter."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CRITICAL: Tính RMS trên float32 để tránh underflow/overflow khi input là fp16/bf16,
        # sau đó ép kiểu trở lại dtype ban đầu của x.
        # Dùng torch.rsqrt() kết hợp phép nhân x thay vì x / torch.sqrt(...) để tối ưu tốc độ.
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normed).to(input_dtype)


if __name__ == "__main__":
    d_model = 384
    custom_norm = RMSNorm(d_model=d_model, eps=1e-6)
    ref_norm = torch.nn.RMSNorm(d_model, eps=1e-6)

    # Đồng bộ trọng số để so sánh độ chính xác
    with torch.no_grad():
        custom_norm.weight.copy_(ref_norm.weight)

    x = torch.randn(2, 16, d_model)
    out_custom = custom_norm(x)
    out_ref = ref_norm(x)

    max_diff = (out_custom - out_ref).abs().max().item()
    print(f"Max diff vs torch.nn.RMSNorm: {max_diff:.8e}")
    assert max_diff < 1e-5, f"Diff quá lớn: {max_diff}"
    print("RMSNorm smoke test passed successfully!")
