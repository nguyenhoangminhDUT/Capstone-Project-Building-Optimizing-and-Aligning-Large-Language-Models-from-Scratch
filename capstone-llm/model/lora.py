"""Low-Rank Adaptation (LoRA) for Linear layers."""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer with LoRA low-rank adaptation."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Đóng băng toàn bộ trọng số của base layer
        for param in self.base.parameters():
            param.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features

        # lora_A: chiếu từ in_features xuống rank r
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)

        # CRITICAL: B khởi tạo 0 để identity init. Tại thời điểm khởi tạo, tích lora_A * lora_B = 0,
        # đảm bảo output của LoRALinear ban đầu đồng nhất 100% với mô hình gốc.
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


if __name__ == "__main__":
    base_linear = nn.Linear(384, 384)
    lora_linear = LoRALinear(base_linear, r=8, alpha=16.0, dropout=0.0)

    x = torch.randn(2, 16, 384)
    out_lora = lora_linear(x)
    out_base = base_linear(x)

    diff = (out_lora - out_base).abs().max().item()
    print(f"LoRALinear vs base max diff: {diff:.8e}")
    assert diff < 1e-7, f"Diff quá lớn: {diff}"
    print("LoRALinear smoke test passed!")

