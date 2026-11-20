"""Pre-training script for 100M LLM on Kaggle (T4 x2 DataParallel, FP16 AMP, Resume Training)."""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import Transformer
from training.dataset import TokenDataset, load_and_tokenize


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:
    """Compute learning rate using linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pre-training."""
    parser = argparse.ArgumentParser(description="Pre-train 100M PyTorch LLM on Kaggle Dual T4 GPUs")
    parser.add_argument("--data", type=str, default="data/tinystories_full.txt", help="Path to text or .bin corpus")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/kaggle/working/capstone_checkpoints/pretrain",
        help="Thư mục lưu trữ checkpoint mặc định trên Kaggle",
    )
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn tới checkpoint .pt để train nối tiếp")
    parser.add_argument("--max_steps", type=int, default=50000, help="Tổng số bước huấn luyện")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate tối đa")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Số bước warmup tuyến tính")
    parser.add_argument("--grad_accum", type=int, default=4, help="Số bước gradient accumulation")
    parser.add_argument("--save_every", type=int, default=1000, help="Số bước lưu checkpoint định kỳ")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Thiết bị huấn luyện (cuda, mps, cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Main pre-training loop for 100M LLM supporting Kaggle T4 x2 Multi-GPU and Resume."""
    args = parse_args()

    # 1. Cấu hình thư mục lưu checkpoint trên Kaggle (tự động fallback nếu chạy máy local)
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        out_dir = Path("checkpoints/pretrain")
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Storage] Thư mục lưu checkpoint: {out_dir}")

    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    num_gpus = torch.cuda.device_count() if device_type == "cuda" else 0

    block_size = 1024  # Độ dài ngữ cảnh nâng cấp 1024 tokens cho model 100M

    print(f"Device: {device} | Số GPU khả dụng: {num_gpus}")
    print(f"Micro Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Effective Batch Size: {args.batch_size * max(1, num_gpus) * args.grad_accum} | Max Steps: {args.max_steps}")

    # 2. Nạp và tokenize dữ liệu (ưu tiên memmap .bin)
    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)

    if not data_path.exists() and (REPO_ROOT / data_path).exists():
        data_path = REPO_ROOT / data_path
    if not tokenizer_path.exists() and (REPO_ROOT / tokenizer_path).exists():
        tokenizer_path = REPO_ROOT / tokenizer_path

    if str(data_path).endswith(".bin") and data_path.exists():
        print(f"[Dataset] Nạp trực tiếp file nhị phân {data_path} qua memmap...")
        token_ids = np.memmap(data_path, dtype=np.uint16, mode="r")
    elif data_path.exists() and tokenizer_path.exists():
        token_ids = load_and_tokenize(str(data_path), str(tokenizer_path))
    else:
        print("[Dataset] Chưa tìm thấy file dữ liệu thật, tạo mảng token giả lập cho kiểm thử...")
        token_ids = np.random.randint(0, 32000, size=50000, dtype=np.uint16)

    if len(token_ids) <= block_size + 1:
        multiplier = ((block_size + 1) * 10 // max(1, len(token_ids))) + 1
        token_ids = np.tile(np.array(token_ids), multiplier)

    dataset = TokenDataset(tokens=token_ids, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size * max(1, num_gpus), shuffle=True, drop_last=True)
    data_iter = iter(dataloader)

    # 3. Khởi tạo mô hình Transformer 100M
    base_model = Transformer(
        vocab_size=32000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=2048,
        max_seq_len=block_size,
        dropout=0.0,
    ).to(device)

    # Multi-GPU DataParallel cho Kaggle T4 x2
    if device_type == "cuda" and num_gpus > 1:
        print(f"[Multi-GPU] Bọc mô hình bằng torch.nn.DataParallel qua {num_gpus} GPUs.")
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # 4. Khởi tạo Optimizer và Mixed Precision FP16
    optimizer = configure_optimizers(
        raw_model,
        weight_decay=0.1,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    use_amp = (device_type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5. Tính năng Resume Training
    global_step = 0
    if args.resume:
        resume_file = Path(args.resume)
        if not resume_file.exists() and (REPO_ROOT / resume_file).exists():
            resume_file = REPO_ROOT / resume_file

        if resume_file.exists():
            print(f"[Resume] Đang nạp checkpoint từ {resume_file}...")
            checkpoint = torch.load(resume_file, map_location=device, weights_only=False)
            sd = checkpoint.get("model_state_dict", checkpoint)
            # Làm sạch tiền tố module. nếu lưu từ DataParallel
            clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            raw_model.load_state_dict(clean_sd)

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            global_step = int(checkpoint.get("step", 0))
            print(f"[Resume] Nạp thành công! Tiếp tục huấn luyện từ step {global_step + 1}...")
        else:
            raise FileNotFoundError(f"Không tìm thấy file checkpoint để resume: {args.resume}")

    # 6. Vòng lặp huấn luyện Pre-training
    print("\n" + "=" * 65)
    print(f"BẮT ĐẦU HUẤN LUYỆN PRE-TRAINING 100M (TARGET: {args.max_steps:,} STEPS)")
    print("=" * 65)

    model.train()
    running_tokens = 0
    t0 = time.time()

    for step in range(global_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        current_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr=args.lr * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Mixed Precision FP16 cho GPU T4
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                # Ép kiểu float32 trước khi tính cross_entropy để tránh tràn số sinh NaN trong FP16
                logits_f32 = logits.float()
                loss = F.cross_entropy(logits_f32.view(-1, raw_model.vocab_size), y.view(-1))
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / args.grad_accum
            running_tokens += x.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Log tiến trình mỗi 50 bước
        if step % 50 == 0 or step == 1 or step == args.max_steps:
            elapsed = time.time() - t0
            tokens_per_sec = running_tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"Step {step:6d}/{args.max_steps} | Loss: {accum_loss:.4f} | "
                f"LR: {current_lr:.2e} | Speed: {tokens_per_sec:,.0f} tok/s"
            )
            running_tokens = 0
            t0 = time.time()

        # Lưu checkpoint định kỳ cho Kaggle (giới hạn 12h)
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            latest_path = out_dir / "latest_checkpoint.pt"
            checkpoint_data = {
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "loss": accum_loss,
            }
            torch.save(checkpoint_data, ckpt_path)
            torch.save(checkpoint_data, latest_path)
            print(f"[Checkpoint] Đã lưu checkpoint tại step {step} ra {ckpt_path}")

    print("\n" + "=" * 65)
    print("HOÀN THÀNH HUẤN LUYỆN PRE-TRAINING 100M!")
    print(f"Checkpoint cuối cùng được lưu tại: {out_dir / 'latest_checkpoint.pt'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
