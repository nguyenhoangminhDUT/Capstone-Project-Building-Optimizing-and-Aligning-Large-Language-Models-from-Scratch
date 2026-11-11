"""Low-Rank Adaptation (LoRA) for Linear layers and model-wide adapter utilities."""

from typing import Dict, List
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


def apply_lora(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj"],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> List[LoRALinear]:
    """Recursively replace specified nn.Linear layers in model with LoRALinear adapters."""
    lora_layers: List[LoRALinear] = []
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            lora_layer = LoRALinear(base=module, r=r, alpha=alpha, dropout=dropout)
            setattr(model, name, lora_layer)
            lora_layers.append(lora_layer)
        else:
            lora_layers.extend(
                apply_lora(
                    module,
                    target_modules=target_modules,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
            )
    return lora_layers


def merge_lora(model: nn.Module) -> None:
    """Merge trained LoRA delta weights directly into base Linear layers for zero-overhead inference."""
    for name, module in model.named_children():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                # delta_W = (alpha / r) * (B @ A) có shape (out_features, in_features)
                delta_w = module.scaling * (module.lora_B @ module.lora_A)
                module.base.weight.add_(delta_w)
            # Khôi phục module về nn.Linear gốc
            setattr(model, name, module.base)
        else:
            merge_lora(module)


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only trainable LoRA adapter weights (lora_A, lora_B) for compact checkpoint storage."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


if __name__ == "__main__":
    base_linear = nn.Linear(384, 384)
    lora_linear = LoRALinear(base_linear, r=8, alpha=16.0, dropout=0.0)

    x = torch.randn(2, 16, 384)
    out_lora = lora_linear(x)
    out_base = base_linear(x)

    diff = (out_lora - out_base).abs().max().item()
    print(f"LoRALinear vs base max diff: {diff:.8e}")
    assert diff < 1e-7, f"Diff quá lớn: {diff}"

    # Test merge_lora
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)

    toy = ToyModel()
    applied = apply_lora(toy, target_modules=["q_proj", "v_proj"], r=4)
    assert len(applied) == 2
    assert isinstance(toy.q_proj, LoRALinear)

    merge_lora(toy)
    assert isinstance(toy.q_proj, nn.Linear)
    print("LoRA utilities smoke test passed successfully!")
