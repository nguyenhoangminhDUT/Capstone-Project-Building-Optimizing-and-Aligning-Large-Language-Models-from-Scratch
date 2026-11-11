"""Low-Rank Adaptation (LoRA) for Linear layers and model-wide adapter utilities."""

from typing import Dict, List, Optional
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
        device = base.weight.device
        dtype = base.weight.dtype

        # lora_A: chiếu từ in_features xuống rank r (khớp device và dtype của base layer)
        self.lora_A = nn.Parameter(torch.randn(r, in_features, device=device, dtype=dtype) * 0.01)

        # CRITICAL: B khởi tạo 0 để identity init. Tại thời điểm khởi tạo, tích lora_A * lora_B = 0,
        # đảm bảo output của LoRALinear ban đầu đồng nhất 100% với mô hình gốc.
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> List[LoRALinear]:
    """
    Áp dụng LoRA vào mô hình:
    1. Đóng băng toàn bộ tham số của model hiện tại (requires_grad = False).
    2. Duyệt qua tất cả các module con, tìm các layer 'q_proj' và 'v_proj' (trong attention)
       và thay thế bằng LoRALinear (bọc lấy layer cũ).
    3. In ra tổng số tham số, số tham số được huấn luyện (chỉ của LoRA), và phần trăm tham số được train.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # 1. Đóng băng toàn bộ tham số của model hiện tại
    for param in model.parameters():
        param.requires_grad = False

    # 2. Duyệt qua tất cả các module con và thay thế q_proj, v_proj bằng LoRALinear
    lora_layers: List[LoRALinear] = []

    def _replace_layers(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
                lora_layer = LoRALinear(base=child, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, name, lora_layer)
                lora_layers.append(lora_layer)
            else:
                _replace_layers(child)

    _replace_layers(model)

    # 3. Tính toán và in thống kê tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    print("=" * 65)
    print("LoRA Applied Successfully:")
    print(f"  - Tổng số tham số (Total params)          : {total_params:,}")
    print(f"  - Số tham số được huấn luyện (Trainable LoRA): {trainable_params:,}")
    print(f"  - Phần trăm tham số được train            : {trainable_pct:.2f}%")
    print("=" * 65)

    return lora_layers


def merge_lora(model: nn.Module) -> None:
    """Merge trained LoRA delta weights directly into base Linear layers for zero-overhead inference."""
    for name, module in list(model.named_children()):
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
    import os
    import sys

    # Thiết lập đường dẫn import
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        from model.transformer import Transformer
    except ImportError:
        from transformer import Transformer

    # 1. Smoke test cơ bản: LoRALinear vs base linear (identity initialization)
    print("--- Test 1: LoRALinear identity initialization ---")
    base_linear = nn.Linear(384, 384)
    lora_linear = LoRALinear(base_linear, r=8, alpha=16.0, dropout=0.0)

    x = torch.randn(2, 16, 384)
    out_lora = lora_linear(x)
    out_base = base_linear(x)

    diff = (out_lora - out_base).abs().max().item()
    print(f"LoRALinear vs base max diff: {diff:.8e}")
    assert diff < 1e-7, f"Diff quá lớn: {diff}"

    # 2. Smoke test apply_lora trên Transformer nhỏ
    print("\n--- Test 2: Smoke test apply_lora trên Transformer nhỏ ---")
    small_transformer = Transformer(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
    )

    # Gọi apply_lora với r=8, alpha=16
    lora_adapters = apply_lora(small_transformer, r=8, alpha=16)

    # Kiểm tra các layer q_proj và v_proj trong CausalSelfAttention
    assert len(lora_adapters) == 4, f"Mong đợi 4 LoRA adapters (2 layers x 2 projs), nhận {len(lora_adapters)}"
    for i, block in enumerate(small_transformer.layers):
        assert isinstance(block.attn.q_proj, LoRALinear), f"Layer {i} q_proj không phải LoRALinear"
        assert isinstance(block.attn.v_proj, LoRALinear), f"Layer {i} v_proj không phải LoRALinear"
        assert isinstance(block.attn.k_proj, nn.Linear) and not isinstance(block.attn.k_proj, LoRALinear), f"Layer {i} k_proj không được là LoRA"
        assert isinstance(block.attn.o_proj, nn.Linear) and not isinstance(block.attn.o_proj, LoRALinear), f"Layer {i} o_proj không được là LoRA"

    # Kiểm tra chỉ các tham số LoRA (lora_A, lora_B) mới có requires_grad == True
    for name, param in small_transformer.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad is True, f"Tham số LoRA {name} phải có requires_grad=True"
        else:
            assert param.requires_grad is False, f"Tham số base {name} phải có requires_grad=False"

    # Kiểm tra forward pass
    dummy_input = torch.randint(0, 1000, (2, 16))
    logits = small_transformer(dummy_input)
    assert logits.shape == (2, 16, 1000), f"Output shape không đúng: {logits.shape}"
    print(f"Forward pass output shape: {logits.shape} (OK)")

    # 3. Test merge_lora
    print("\n--- Test 3: Test merge_lora ---")
    merge_lora(small_transformer)
    for i, block in enumerate(small_transformer.layers):
        assert isinstance(block.attn.q_proj, nn.Linear) and not isinstance(block.attn.q_proj, LoRALinear)
        assert isinstance(block.attn.v_proj, nn.Linear) and not isinstance(block.attn.v_proj, LoRALinear)
    print("Merge LoRA thành công!")

    print("\nToàn bộ smoke test LoRA đã hoàn thành thành công!")
