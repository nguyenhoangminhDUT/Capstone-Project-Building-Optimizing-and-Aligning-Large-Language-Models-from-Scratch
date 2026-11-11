"""Pre-training script for pure PyTorch LLM with AdamW, Cosine LR, Mixed Precision, and Grad Accumulation."""

import argparse
import math
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pre-training."""
    parser = argparse.ArgumentParser(description="Pre-train custom PyTorch LLM")
    parser.add_argument("--data", type=str, default="data/tinystories_sample.txt", help="Path to text corpus")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to use for training (cuda, mps, cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function for pre-training loop."""
    args = parse_args()

    # Tạo thư mục lưu checkpoint
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    block_size = 512

    print(f"Device: {device} | Micro Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Effective Batch Size: {args.batch_size * args.grad_accum} | Max Steps: {args.max_steps}")

    # 1. Nạp và tokenize dữ liệu
    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)

    # Dự phòng đường dẫn trong thư mục capstone-llm nếu chạy từ root
    if not data_path.exists() and (REPO_ROOT / data_path).exists():
        data_path = REPO_ROOT / data_path
    if not tokenizer_path.exists() and (REPO_ROOT / tokenizer_path).exists():
        tokenizer_path = REPO_ROOT / tokenizer_path

    if not data_path.exists() or not tokenizer_path.exists():
        print("Tạo dữ liệu ngẫu nhiên cho smoke test do chưa tìm thấy file dữ liệu thật...")
        token_ids = np.random.randint(0, 12000, size=20000, dtype=np.int64)
    else:
        print(f"Tokenizing dữ liệu từ {data_path}...")
        token_ids = load_and_tokenize(str(data_path), str(tokenizer_path))
        if len(token_ids) <= block_size:
            print("Cảnh báo: Dữ liệu quá ngắn, tự động nhân bản dữ liệu để đủ huấn luyện...")
            multiplier = (block_size * 10 // max(1, len(token_ids))) + 1
            token_ids = np.tile(token_ids, multiplier)

    dataset = TokenDataset(tokens=token_ids, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_iter = iter(dataloader)

    # 2. Khởi tạo mô hình Transformer theo đúng spec
    model = Transformer(
        vocab_size=12000,
        d_model=384,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=block_size,
        dropout=0.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Mô hình đã khởi tạo: {total_params:,} tham số")

    # 3. Khởi tạo Optimizer và Mixed Precision
    optimizer = configure_optimizers(
        model,
        weight_decay=0.1,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    # Thiết lập mixed precision: bfloat16 cho GPU Ampere+ (RTX 3060), float16 cho Colab T4
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device_type == "cuda")
    scaler = torch.amp.GradScaler(device=device_type, enabled=(use_amp and amp_dtype == torch.float16))

    try:
        model.train()
        step = 0
        running_tokens = 0
        t0 = time.time()

        while step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Cập nhật learning rate theo cosine schedule
            current_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr=args.lr * 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Gradient Accumulation Loop
            for _ in range(args.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    x, y = next(data_iter)

                x, y = x.to(device), y.to(device)

                # CRITICAL: Về việc tính cross_entropy trong hoặc ngoài autocast:
                # Với fp16, dải biểu diễn số mũ hạn chế (tối đa ~65504). Nếu logits lớn, hàm Softmax
                # trong cross_entropy dễ bị tràn số (overflow) sinh ra NaN loss.
                # Vì vậy, khi chạy fp16, ta ép kiểu logits về float32 (logits.float()) trước khi tính loss
                # để đảm bảo ổn định số học (numerical stability).
                # Với bf16 (trên RTX 3060+), dải số mũ tương đương float32 (8-bit exponent) nên hoàn toàn
                # không lo tràn số.
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().view(-1, model.vocab_size), y.view(-1))
                    # Chia loss cho grad_accum để scale gradient trung bình chính xác
                    loss = loss / args.grad_accum

                accum_loss += loss.item() * args.grad_accum
                scaler.scale(loss).backward()
                running_tokens += args.batch_size * block_size

            # Gradient clipping 1.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1

            # Log mỗi 50 step (hoặc step đầu tiên và step cuối cùng)
            if step % 50 == 0 or step == 1 or step == args.max_steps:
                elapsed = time.time() - t0
                tok_per_sec = running_tokens / elapsed if elapsed > 0 else 0
                print(
                    f"Step {step:5d}/{args.max_steps} | Loss: {accum_loss:.4f} | "
                    f"LR: {current_lr:.2e} | Speed: {tok_per_sec:.1f} tok/s"
                )
                running_tokens = 0
                t0 = time.time()

            # Lưu checkpoint mỗi 1000 step
            if step % 1000 == 0 or step == args.max_steps:
                ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": accum_loss,
                        "lr": current_lr,
                    },
                    ckpt_path,
                )
                print(f"[Checkpoint] Đã lưu mô hình tại {ckpt_path}")

        print("Huấn luyện hoàn thành xuất sắc!")

    except torch.cuda.OutOfMemoryError as oom_err:
        print("\n" + "=" * 70)
        print("LỖI: torch.cuda.OutOfMemoryError - GPU bị tràn bộ nhớ VRAM!")
        print(f"GỢI Ý KHẮC PHỤC:")
        print(f"1. Giảm micro batch size: hạ --batch_size (ví dụ từ {args.batch_size} xuống {max(1, args.batch_size // 2)}).")
        print(f"2. Tăng --grad_accum (ví dụ từ {args.grad_accum} lên {args.grad_accum * 2}) để giữ nguyên effective batch size.")
        print(f"3. Đảm bảo đang sử dụng mixed precision (bf16/fp16).")
        print("=" * 70 + "\n")
        raise oom_err


if __name__ == "__main__":
    main()
