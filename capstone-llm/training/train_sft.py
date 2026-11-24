"""Supervised Fine-Tuning (SFT) with 100M LLM, LoRA (r=32), Math Mix, and Kaggle Multi-GPU."""

import os
# CRITICAL: Tránh phân mảnh bộ nhớ VRAM trên GPU T4 x2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.lora import apply_lora, get_lora_state_dict, merge_lora
from model.transformer import Transformer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def generate_math_samples(num_samples: int = 10000) -> List[Dict[str, str]]:
    """Tự động sinh 10,000 mẫu bài toán số học cộng/trừ 1-2 chữ số để rèn luyện tư duy logic toán học."""
    random.seed(42)
    samples = []
    templates = [
        ("What is {a} + {b}?", "{ans}"),
        ("Calculate {a} plus {b}.", "{ans}"),
        ("Compute {a} + {b}.", "{ans}"),
        ("What is the sum of {a} and {b}?", "The sum of {a} and {b} is {ans}."),
        ("What is {a} - {b}?", "{ans}"),
        ("Calculate {a} minus {b}.", "{ans}"),
        ("Compute {a} - {b}.", "{ans}"),
        ("What is {a} subtracted by {b}?", "{ans}"),
        ("If you have {a} apples and buy {b} more, how many apples do you have?", "You have {ans} apples in total."),
        ("There are {a} students in a classroom. If {b} students leave, how many are left?", "There are {ans} students remaining."),
    ]
    for _ in range(num_samples):
        op = random.choice(["+", "-"])
        if op == "+":
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            ans = a + b
            valid_tmpls = [t for t in templates if "+" in t[0] or "plus" in t[0] or "sum" in t[0] or "more" in t[0]]
            q_tmpl, a_tmpl = random.choice(valid_tmpls)
        else:
            a = random.randint(1, 99)
            b = random.randint(1, a)
            ans = a - b
            valid_tmpls = [t for t in templates if "-" in t[0] or "minus" in t[0] or "subtracted" in t[0] or "leave" in t[0]]
            q_tmpl, a_tmpl = random.choice(valid_tmpls)

        question = q_tmpl.format(a=a, b=b)
        answer = a_tmpl.format(a=a, b=b, ans=ans)
        samples.append({
            "instruction": question,
            "input": "",
            "output": answer,
        })
    return samples


class SFTDataset(Dataset):
    """Instruction dataset with prompt-loss masking (label = -100 on prompt tokens)."""

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    def __init__(
        self,
        samples: Union[List[Dict[str, str]], object],
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
        pad_token_id: int = 1,
        eos_token_id: int = 3,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        if hasattr(sample, "get"):
            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            output = sample.get("output", "")
        else:
            instruction = sample["instruction"]
            inp = sample.get("input", "") if "input" in sample else ""
            output = sample["output"]

        instruction = str(instruction).strip()
        inp = str(inp).strip()
        output = str(output).strip()

        if inp:
            instruction = f"{instruction}\n{inp}"

        prompt_text = self.PROMPT_TEMPLATE.format(instruction=instruction)
        response_text = f"{output}<|eos|>"

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response_text).ids

        # Giới hạn prompt để không chiếm hết context
        max_prompt_len = max(16, self.max_seq_len - 64)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        if len(response_ids) == 0:
            response_ids = [self.eos_token_id]

        # CRITICAL: Gán nhãn -100 cho toàn bộ token thuộc prompt và template
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        target_len = self.max_seq_len + 1
        if len(input_ids) > target_len:
            input_ids = input_ids[:target_len]
            labels = labels[:target_len]
        else:
            pad_len = target_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def find_base_checkpoint(
    explicit_path: Optional[str] = None,
    checkpoint_dirs: Optional[List[str]] = None,
) -> Optional[Path]:
    """Tìm checkpoint pre-train phù hợp nhất (ưu tiên explicit -> best_model.pt -> latest_checkpoint.pt)."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        if (REPO_ROOT / p).exists():
            return REPO_ROOT / p

    if checkpoint_dirs is None:
        checkpoint_dirs = [
            "/kaggle/working/capstone_checkpoints/pretrain",
            "/kaggle/working/capstone_checkpoints",
            "checkpoints/pretrain",
            "checkpoints",
            str(REPO_ROOT / "checkpoints/pretrain"),
            str(REPO_ROOT / "checkpoints"),
        ]

    for d in checkpoint_dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            continue

        best = dir_path / "best_model.pt"
        if best.exists():
            return best

        latest = dir_path / "latest_checkpoint.pt"
        if latest.exists():
            return latest

        step_files = list(dir_path.glob("checkpoint_step_*.pt"))
        if step_files:
            def extract_step(path: Path) -> int:
                try:
                    return int(path.stem.split("_")[-1])
                except ValueError:
                    return 0
            step_files.sort(key=extract_step, reverse=True)
            return step_files[0]

    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SFT training."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for 100M LLM with LoRA r=32")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Đường dẫn tới pretrain checkpoint 100M (best_model.pt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Đường dẫn tới tokenizer.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/kaggle/working/capstone_checkpoints/sft",
        help="Thư mục lưu SFT checkpoints trên Kaggle",
    )
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn tới SFT checkpoint để train nối tiếp")
    parser.add_argument(
        "--full_ft",
        action="store_true",
        help="Kích hoạt chế độ Full Fine-Tuning (mở khóa toàn bộ 100%% tham số mô hình, không sử dụng LoRA)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện SFT (mặc định 3 epochs)")
    parser.add_argument("--max_samples", type=int, default=20000, help="Số lượng mẫu tải từ yahma/alpaca-cleaned")
    parser.add_argument("--math_samples", type=int, default=10000, help="Số lượng mẫu toán học tự sinh (1-2 chữ số)")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size per GPU (mặc định 4)")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps (mặc định 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate cho các adapter LoRA (mặc định 2e-4)")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank r (mặc định 32 cho model 100M)")
    parser.add_argument("--lora_alpha", type=float, default=64.0, help="LoRA alpha (mặc định 64)")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Độ dài ngữ cảnh tối đa")
    parser.add_argument("--save_every", type=int, default=500, help="Số bước lưu checkpoint định kỳ")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Thiết bị huấn luyện (cuda, mps, cpu)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Chạy thử nghiệm nhanh với dữ liệu giả lập")
    return parser.parse_args()


def main() -> None:
    """Main execution function for SFT training with LoRA r=32 on Kaggle."""
    args = parse_args()

    # 1. Cấu hình thư mục lưu checkpoint trên Kaggle
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        out_dir = Path("checkpoints/sft")
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Checkpoints] Thư mục lưu kết quả: {out_dir}")

    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    num_gpus = torch.cuda.device_count() if device_type == "cuda" else 0

    print(f"Device: {device} | Số GPU khả dụng: {num_gpus}")
    print(f"Micro Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Effective Batch Size: {args.batch_size * max(1, num_gpus) * args.grad_accum} | Epochs: {args.epochs}")

    # 2. Nạp Tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists() and (REPO_ROOT / tokenizer_path).exists():
        tokenizer_path = REPO_ROOT / tokenizer_path
    if not tokenizer_path.exists():
        for cand in [
            REPO_ROOT / "tokenizer" / "tokenizer.json",
            Path("tokenizer/tokenizer.json"),
            Path("/kaggle/working/tokenizer.json"),
        ]:
            if cand.exists():
                tokenizer_path = cand
                break
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tokenizer tại: {args.tokenizer}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"[Tokenizer] Nạp thành công từ {tokenizer_path} (vocab_size={tokenizer.get_vocab_size():,})")

    # 3. Chuẩn bị tập dữ liệu: Alpaca (20,000) + Toán học tự sinh (10,000)
    if args.smoke_test or load_dataset is None:
        print("[Dataset] Chạy chế độ smoke test với dữ liệu mẫu...")
        raw_samples = [
            {"instruction": "What is 47 + 28?", "input": "", "output": "75"},
            {"instruction": "Calculate 83 - 29.", "input": "", "output": "54"},
            {"instruction": "What is the capital of Vietnam?", "input": "", "output": "The capital of Vietnam is Hanoi."},
            {"instruction": "Explain gravity in one sentence.", "input": "", "output": "Gravity is the force attracting masses."},
        ]
    else:
        all_samples = []
        # a. Tải yahma/alpaca-cleaned
        print(f"[Dataset] Đang nạp {args.max_samples:,} mẫu từ 'yahma/alpaca-cleaned'...")
        try:
            ds_alpaca = load_dataset("yahma/alpaca-cleaned", split=f"train[:{args.max_samples}]")
            for item in ds_alpaca:
                all_samples.append({
                    "instruction": str(item.get("instruction", "")).strip(),
                    "input": str(item.get("input", "")).strip(),
                    "output": str(item.get("output", "")).strip(),
                })
            print(f"[Dataset] Đã thêm {len(all_samples):,} mẫu Alpaca.")
        except Exception as e:
            print(f"[Dataset] Cảnh báo khi tải Alpaca ({e}), tiếp tục với dữ liệu toán học...")

        # b. Sinh thêm 10,000 mẫu bài toán số học (cộng/trừ 1-2 chữ số)
        if args.math_samples > 0:
            print(f"[Dataset] Đang tự động sinh {args.math_samples:,} mẫu dữ liệu toán học (cộng/trừ 1-2 chữ số)...")
            math_samples = generate_math_samples(num_samples=args.math_samples)
            all_samples.extend(math_samples)
            print(f"[Dataset] Đã trộn thành công {len(math_samples):,} mẫu toán học vào tập huấn luyện SFT.")

        random.seed(42)
        random.shuffle(all_samples)
        raw_samples = all_samples
        print(f"[Dataset] Tổng số mẫu SFT kết hợp: {len(raw_samples):,}")

    # Tách tập Validation cố định (1,000 mẫu với seed=42)
    random.seed(42)
    shuffled_samples = list(raw_samples)
    random.shuffle(shuffled_samples)

    if args.smoke_test:
        val_samples = shuffled_samples[:2]
        train_samples = shuffled_samples
        print(f"[Dataset Split] Smoke test: {len(train_samples)} mẫu train, {len(val_samples)} mẫu validation.")
    elif len(shuffled_samples) > 1000:
        val_samples = shuffled_samples[:1000]
        train_samples = shuffled_samples[1000:]
        print(f"[Dataset Split] Đã tách 1,000 mẫu validation (seed=42). Train: {len(train_samples):,} mẫu, Val: {len(val_samples):,} mẫu.")
    else:
        split_idx = max(1, len(shuffled_samples) // 5)
        val_samples = shuffled_samples[:split_idx]
        train_samples = shuffled_samples[split_idx:]
        print(f"[Dataset Split] Train: {len(train_samples):,} mẫu, Val: {len(val_samples):,} mẫu.")

    train_dataset = SFTDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size * max(1, num_gpus), shuffle=True, drop_last=True)

    val_dataset = SFTDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * max(1, num_gpus), shuffle=False)

    # 4. Khởi tạo mô hình Transformer 100M
    base_model = Transformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=2048,
        max_seq_len=args.max_seq_len,
        dropout=0.0,
    ).to(device)

    # Nạp pre-train base checkpoint nếu có
    base_ckpt = find_base_checkpoint(args.base_ckpt)
    if base_ckpt:
        print(f"[Base Model] Đang nạp trọng số pre-train từ {base_ckpt}...")
        ckpt_data = torch.load(base_ckpt, map_location=device, weights_only=False)
        sd = ckpt_data.get("model_state_dict", ckpt_data)
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        base_model.load_state_dict(clean_sd)
        print("[Base Model] Nạp checkpoint base thành công!")
    else:
        print("[Base Model] Khởi tạo trọng số ngẫu nhiên cho mô hình 100M.")

    # 5. Cấu hình chế độ: Full Fine-Tuning vs LoRA
    if args.full_ft:
        print("[SFT Training] CHẾ ĐỘ FULL FINE-TUNING ĐƯỢC BẬT: Mở khóa 100% tham số (requires_grad=True), KHÔNG dùng LoRA.")
        for p in base_model.parameters():
            p.requires_grad = True
    else:
        print(f"[SFT Training] CHẾ ĐỘ LORA ĐƯỢC BẬT: Gắn LoRA r={args.lora_r}, alpha={args.lora_alpha}.")
        apply_lora(base_model, r=args.lora_r, alpha=args.lora_alpha)
        # Mở khóa lớp Embedding/LM Head trong giai đoạn SFT (Weight Tying tự động mở khóa cả lm_head.weight)
        base_model.embedding.weight.requires_grad_(True)
        print(f"[SFT Training] Đã mở khóa Embedding/LM Head: requires_grad={base_model.embedding.weight.requires_grad}")

    # Multi-GPU DataParallel cho Kaggle T4 x2
    if device_type == "cuda" and num_gpus > 1:
        print(f"[Multi-GPU] Bọc mô hình bằng torch.nn.DataParallel qua {num_gpus} GPUs.")
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # 6. Khởi tạo Optimizer: tách 2 nhóm tham số cho LoRA hoặc 1 nhóm cho Full FT
    if args.full_ft:
        trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        print(
            f"[SFT Optimizer - Full FT] Mở khóa 100% tham số: {sum(p.numel() for p in trainable_params):,} tham số "
            f"(lr={args.lr:.2e}, weight_decay=0.01, Cosine Annealing decay to {args.lr * 0.1:.2e})"
        )
    else:
        # Nhóm 1: các tham số chứa "lora_" dùng lr=args.lr (weight_decay=0.01)
        # Nhóm 2: [raw_model.embedding.weight] dùng lr=args.lr * 0.25 (weight_decay=0.0) để tránh xáo trộn biểu diễn từ vựng nền tảng
        lora_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and "lora_" in n]
        embed_params = [raw_model.embedding.weight]

        optim_groups = [
            {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": embed_params, "lr": args.lr * 0.25, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95))

        trainable_params = lora_params + embed_params
        total_trainable = sum(p.numel() for p in trainable_params)
        print(
            f"[SFT Optimizer - LoRA] Tổng tham số train: {total_trainable:,} "
            f"(LoRA: {sum(p.numel() for p in lora_params):,}, Embedding: {raw_model.embedding.weight.numel():,})"
        )

    use_amp = (device_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 7. Tính năng Tự động Resume Training SFT
    start_epoch = 1
    global_step = 0
    resume_file: Optional[Path] = None

    if args.resume:
        rf = Path(args.resume)
        if not rf.exists() and (REPO_ROOT / rf).exists():
            rf = REPO_ROOT / rf
        if rf.exists():
            resume_file = rf
        else:
            raise FileNotFoundError(f"Không tìm thấy SFT checkpoint để resume: {args.resume}")
    else:
        # Tự động tìm checkpoint trong out_dir
        latest_candidate = out_dir / "latest_sft_checkpoint.pt"
        if latest_candidate.exists():
            resume_file = latest_candidate
        else:
            step_files = list(out_dir.glob("sft_step_*.pt"))
            if step_files:
                def extract_step(path: Path) -> int:
                    try:
                        return int(path.stem.split("_")[-1])
                    except ValueError:
                        return 0
                step_files.sort(key=extract_step, reverse=True)
                resume_file = step_files[0]

    if resume_file:
        print(f"[Auto-Resume SFT] Đang nạp SFT checkpoint từ: {resume_file}")
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        if "lora_state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Auto-Resume SFT] Cảnh báo không nạp được optimizer state: {e}")

        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and scaler.is_enabled():
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                print(f"[Auto-Resume SFT] Cảnh báo không nạp được scaler state: {e}")

        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 1))
        print(f"[Auto-Resume SFT] Nạp thành công! Tiếp tục từ Epoch {start_epoch}, Step {global_step + 1}...")

    # 8. Vòng lặp huấn luyện SFT
    mode_str = "FULL FINE-TUNING (100% PARAMS)" if args.full_ft else f"LORA (r={args.lora_r}, a={args.lora_alpha})"
    print("\n" + "=" * 65)
    print(f"BẮT ĐẦU HUẤN LUYỆN SFT 100M - CHẾ ĐỘ: {mode_str}")
    print(f"TỔNG SỐ BATCH: {len(dataloader)} BATCHES/EPOCH | EPOCHS: {args.epochs}")
    print("=" * 65)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_start_time = time.time()
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    model.train()
    total_steps_in_epoch = len(dataloader) // args.grad_accum
    total_training_steps = max(1, total_steps_in_epoch * args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        running_loss = 0.0
        accum_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                # Ép kiểu float32 trước cross_entropy để chống tràn số trong fp16
                logits_f32 = logits.float()
                loss = F.cross_entropy(
                    logits_f32.view(-1, raw_model.vocab_size),
                    y.view(-1),
                    ignore_index=-100,
                )
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += accum_loss

                # Cosine Annealing LR decay cho chế độ Full Fine-Tuning (decay về 10% LR ban đầu)
                if args.full_ft:
                    decay_ratio = min(1.0, global_step / total_training_steps)
                    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                    min_lr = args.lr * 0.1
                    curr_lr = min_lr + coeff * (args.lr - min_lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr

                if global_step % 20 == 0 or args.smoke_test:
                    if args.full_ft:
                        print(
                            f"Epoch {epoch:2d}/{args.epochs} | Step {global_step:5d} | "
                            f"Loss: {accum_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                        )
                    else:
                        print(
                            f"Epoch {epoch:2d}/{args.epochs} | Step {global_step:5d} | "
                            f"Loss: {accum_loss:.4f} | LR LoRA: {optimizer.param_groups[0]['lr']:.2e} | LR Embed: {optimizer.param_groups[1]['lr']:.2e}"
                        )

                if global_step % args.save_every == 0 and not args.smoke_test:
                    ckpt_save_path = out_dir / f"sft_step_{global_step}.pt"
                    save_payload = {
                        "epoch": epoch,
                        "step": global_step,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                        "loss": accum_loss,
                        "mode": "full_ft" if args.full_ft else "lora",
                    }
                    if args.full_ft:
                        save_payload["model_state_dict"] = raw_model.state_dict()
                        torch.save(save_payload, ckpt_save_path)
                        torch.save(save_payload, out_dir / "latest_sft_checkpoint.pt")
                        torch.save({"model_state_dict": raw_model.state_dict(), "mode": "full_ft"}, out_dir / "full_sft_model.pt")
                    else:
                        save_payload["lora_state_dict"] = get_lora_state_dict(raw_model)
                        torch.save(save_payload, ckpt_save_path)
                        torch.save(save_payload, out_dir / "latest_sft_checkpoint.pt")
                        torch.save(get_lora_state_dict(raw_model), out_dir / "lora_sft_adapter.pt")
                    print(f"[Checkpoint] Đã lưu SFT checkpoint tại step {global_step} vào {ckpt_save_path}")

                accum_loss = 0.0

            if args.smoke_test and global_step >= 3:
                break

        # Đánh giá Val Loss cuối mỗi epoch (áp dụng cho cả LoRA và Full FT)
        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        with torch.no_grad():
            for vx, vy in val_dataloader:
                vx, vy = vx.to(device), vy.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    v_logits = model(vx).float()
                    v_loss = F.cross_entropy(
                        v_logits.view(-1, raw_model.vocab_size),
                        vy.view(-1),
                        ignore_index=-100,
                    )
                val_loss_accum += v_loss.item()
                val_steps += 1
                if args.smoke_test and val_steps >= 2:
                    break
        val_loss = val_loss_accum / max(1, val_steps)
        model.train()

        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / max(1, total_steps_in_epoch)
        train_loss_history.append(round(avg_loss, 4))
        val_loss_history.append(round(val_loss, 4))

        print(
            f"\n--- Kết thúc Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Thời gian: {epoch_time:.2f}s ---"
        )

        epoch_ckpt_path = out_dir / f"sft_epoch_{epoch}.pt"
        save_epoch_payload = {
            "epoch": epoch,
            "step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "loss": avg_loss,
            "val_loss": val_loss,
            "mode": "full_ft" if args.full_ft else "lora",
        }
        if args.full_ft:
            save_epoch_payload["model_state_dict"] = raw_model.state_dict()
            torch.save(save_epoch_payload, epoch_ckpt_path)
            torch.save(save_epoch_payload, out_dir / "latest_sft_checkpoint.pt")
            torch.save({"model_state_dict": raw_model.state_dict(), "mode": "full_ft"}, out_dir / "full_sft_model.pt")
            print(f"[Checkpoint] Đã lưu Full SFT Model ra {out_dir / 'full_sft_model.pt'}")
        else:
            save_epoch_payload["lora_state_dict"] = get_lora_state_dict(raw_model)
            torch.save(save_epoch_payload, epoch_ckpt_path)
            torch.save(save_epoch_payload, out_dir / "latest_sft_checkpoint.pt")
            torch.save(get_lora_state_dict(raw_model), out_dir / "lora_sft_adapter.pt")
            print(f"[Checkpoint] Đã lưu SFT Adapter ra {out_dir / 'lora_sft_adapter.pt'}")

        if args.smoke_test:
            break

    # Lưu lại weights cuối cùng
    if args.full_ft:
        torch.save({"model_state_dict": raw_model.state_dict(), "mode": "full_ft"}, out_dir / "full_sft_model.pt")
    else:
        torch.save(get_lora_state_dict(raw_model), out_dir / "lora_sft_adapter.pt")

    # Đo đạc và ghi nhận các chỉ số thực nghiệm ra file sft_metrics.json
    elapsed_sec = round(time.time() - training_start_time, 2)
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

    metrics_payload = {
        "mode": "full_ft" if args.full_ft else "lora",
        "total_params": total_params,
        "trainable_params": trainable_params_count,
        "peak_vram_gb": peak_vram_gb,
        "elapsed_sec": elapsed_sec,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    metrics_path = out_dir / "sft_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("HOÀN THÀNH HUẤN LUYỆN SFT 100M!")
    if args.full_ft:
        print(f"Full SFT Model được lưu tại: {out_dir / 'full_sft_model.pt'}")
    else:
        print(f"LoRA Adapter được lưu an toàn tại: {out_dir / 'lora_sft_adapter.pt'}")
    print(f"Checkpoint SFT mới nhất tại: {out_dir / 'latest_sft_checkpoint.pt'}")
    print(f"Báo cáo số liệu thực nghiệm (Peak VRAM, Time, Loss) tại: {metrics_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
