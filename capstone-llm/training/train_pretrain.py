"""Pre-training loop with AdamW, Cosine LR scheduling, mixed precision, and gradient clipping."""

import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Đảm bảo import được các module từ thư mục capstone-llm
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import Transformer
from training.dataset import PretrainDataset


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """Configure AdamW with parameter grouping to exclude 1D tensors from weight decay."""
    # CRITICAL: Trọng số 1D (RMSNorm scale weight) không áp dụng weight decay
    # để tránh làm suy giảm khả năng chuẩn hóa phương sai của các tầng.
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=1e-8,
        fused=use_fused,
    )
    return optimizer


def train(
    model: Transformer,
    dataset: PretrainDataset,
    batch_size: int = 4,
    max_steps: int = 10,
    warmup_steps: int = 2,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    grad_clip: float = 1.0,
    device: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
) -> list[float]:
    """Run pre-training loop with mixed precision, cosine LR decay, and gradient clipping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    device_type = "cuda" if "cuda" in device else "cpu"
    model = model.to(device)
    model.train()

    optimizer = configure_optimizers(model, learning_rate=max_lr, device_type=device_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(dataloader)

    # Mixed precision setup: bfloat16 cho RTX 3060, float16 cho Colab T4
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device_type == "cuda")
    scaler = torch.amp.GradScaler(device=device_type, enabled=(use_amp and amp_dtype == torch.float16))

    losses = []
    for step in range(max_steps):
        t0 = time.time()
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Cập nhật learning rate theo cosine schedule
        lr = get_lr(step, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            # CRITICAL: PretrainDataset đã chuẩn bị sẵn target y là chuỗi dịch phải 1 token
            # nên loss được tính trực tiếp mà không cần shift thêm một lần nữa.
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step_loss = loss.item()
        losses.append(step_loss)
        dt = time.time() - t0
        tok_per_sec = (batch_size * dataset.seq_len) / dt if dt > 0 else 0

        print(f"Step {step + 1:2d}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.1f} tok/s")

    if save_checkpoint_path:
        out_dir = Path(save_checkpoint_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_checkpoint_path)
        print(f"Model checkpoint saved to {save_checkpoint_path}")

    return losses


if __name__ == "__main__":
    # Smoke test: Khởi tạo mô hình nhỏ và chạy 3 step huấn luyện thử
    seq_len = 64
    vocab_size = 1000

    tiny_model = Transformer(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=seq_len,
    )

    # Tạo dữ liệu ngẫu nhiên gồm 1000 tokens
    dummy_tokens = np.random.randint(0, vocab_size, size=1000, dtype=np.int64)
    dummy_dataset = PretrainDataset(tokens=dummy_tokens, seq_len=seq_len)

    print("Running train_pretrain smoke test (3 steps)...")
    losses = train(
        model=tiny_model,
        dataset=dummy_dataset,
        batch_size=2,
        max_steps=3,
        warmup_steps=1,
    )

    assert len(losses) == 3, "Số bước train không khớp!"
    assert all(not math.isnan(l) and not math.isinf(l) for l in losses), "Loss bị NaN hoặc Inf!"
    print("train_pretrain smoke test passed successfully!")

