"""Supervised Fine-Tuning (SFT) with LoRA, prompt-loss masking, and Google Drive Checkpointing."""

import argparse
import glob
import math
import os
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


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Tự động mount Google Drive nếu đang chạy trong môi trường Google Colab."""
    try:
        from google.colab import drive

        drive_path = Path(mount_point)
        if not drive_path.exists() or not (drive_path / "MyDrive").exists():
            print("[Google Drive] Đang tự động kết nối Google Drive...")
            drive.mount(mount_point)
            print("[Google Drive] Đã kết nối Google Drive thành công!")
        return True
    except ImportError:
        # Chạy trên máy local (không có module google.colab)
        return False
    except Exception as e:
        print(f"[Google Drive] Lưu ý khi mount Drive: {e}")
        return False


class SFTDataset(Dataset):
    """
    Instruction dataset with prompt-loss masking:
    - Format prompt theo template chuẩn của Alpaca.
    - CỰC KỲ QUAN TRỌNG: Gán label = -100 cho toàn bộ token thuộc Instruction và template.
    - Chỉ giữ nguyên ID token thật cho phần Response.
    - F.cross_entropy với ignore_index=-100 sẽ tự động bỏ qua toàn bộ phần prompt,
      đảm bảo mô hình chỉ học cách sinh câu trả lời (response), không học lại câu hỏi.
    """

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    def __init__(
        self,
        samples: Union[List[Dict[str, str]], object],
        tokenizer: Tokenizer,
        max_seq_len: int = 512,
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

        # Trích xuất instruction, input (nếu có) và output
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

        # 1. Định dạng prompt theo đúng template chuẩn
        prompt_text = self.PROMPT_TEMPLATE.format(instruction=instruction)
        response_text = f"{output}<|eos|>"

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response_text).ids

        # Tránh prompt quá dài chiếm toàn bộ max_seq_len (đảm bảo chừa chỗ cho response)
        max_prompt_len = max(16, self.max_seq_len - 64)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        # Đảm bảo response không rỗng
        if len(response_ids) == 0:
            response_ids = [self.eos_token_id]

        # 2. CRITICAL: Gán nhãn (labels) là -100 cho toàn bộ token thuộc phần Instruction
        # và các ký tự template, chỉ giữ nguyên ID token cho phần Response.
        # F.cross_entropy với ignore_index=-100 chỉ tính gradient và loss trên phần Response!
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 3. Padding hoặc truncate chuỗi đến target_len = max_seq_len + 1
        # để trích xuất x = input_ids[:-1] và y = labels[1:] có đúng độ dài max_seq_len
        target_len = self.max_seq_len + 1
        if len(input_ids) > target_len:
            input_ids = input_ids[:target_len]
            labels = labels[:target_len]
        else:
            pad_len = target_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        # Cặp dịch 1 token cho causal language modeling:
        # Ở vị trí cuối prompt, x thấy token cuối prompt, y là token đầu tiên của response (không phải -100)
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def find_base_checkpoint(
    explicit_path: Optional[str] = None,
    checkpoint_dirs: Optional[List[str]] = None,
) -> Optional[Path]:
    """Tìm checkpoint pre-train phù hợp nhất (ưu tiên explicit -> step 50000 -> latest)."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        if (REPO_ROOT / p).exists():
            return REPO_ROOT / p

    if checkpoint_dirs is None:
        checkpoint_dirs = [
            "/content/drive/MyDrive/capstone_checkpoints",
            "checkpoints",
            str(REPO_ROOT / "checkpoints"),
        ]

    for d in checkpoint_dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            continue

        # 1. Tìm checkpoint_step_50000.pt theo yêu cầu đề bài
        step50k = dir_path / "checkpoint_step_50000.pt"
        if step50k.exists():
            return step50k

        # 2. Tìm latest_checkpoint.pt
        latest = dir_path / "latest_checkpoint.pt"
        if latest.exists():
            return latest

        # 3. Tìm checkpoint có step cao nhất
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
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) with LoRA & Prompt Masking")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Đường dẫn tới pretrain checkpoint (Tuần 2)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Đường dẫn tới tokenizer.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/content/drive/MyDrive/capstone_checkpoints/sft",
        help="Thư mục lưu SFT checkpoints trên Google Drive",
    )
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn tới SFT checkpoint để train nối tiếp")
    parser.add_argument("--epochs", type=int, default=2, help="Số epoch huấn luyện SFT (1-2 epochs)")
    parser.add_argument("--max_samples", type=int, default=20000, help="Số lượng mẫu tải từ yahma/alpaca-cleaned")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate cho các adapter LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank r cho LoRA")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Scaling factor alpha cho LoRA")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Độ dài ngữ cảnh tối đa")
    parser.add_argument("--save_every", type=int, default=500, help="Số bước lưu checkpoint trung gian")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device huấn luyện (cuda, mps, cpu)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Chạy thử nghiệm nhanh với dữ liệu giả lập")
    return parser.parse_args()


def main() -> None:
    """Vòng lặp chính huấn luyện SFT với LoRA."""
    args = parse_args()

    # 1. Tự động kết nối Google Drive nếu đang chạy trong Google Colab
    mount_google_drive()

    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        # Dự phòng trên máy local nếu đường dẫn /content/drive không khả dụng
        out_dir = Path("checkpoints/sft")
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Checkpoints] Thư mục lưu kết quả: {out_dir}")

    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    print(f"Device: {device} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Effective Batch Size: {args.batch_size * args.grad_accum} | Epochs: {args.epochs}")

    # 2. Nạp Tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists() and (REPO_ROOT / tokenizer_path).exists():
        tokenizer_path = REPO_ROOT / tokenizer_path
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tokenizer tại: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"[Tokenizer] Nạp thành công từ {tokenizer_path} (vocab_size={tokenizer.get_vocab_size()})")

    # 3. Tải dataset yahma/alpaca-cleaned (20,000 mẫu đầu tiên)
    if args.smoke_test or load_dataset is None:
        print("[Dataset] Chạy chế độ smoke test với mẫu dữ liệu giả lập...")
        raw_samples = [
            {"instruction": "What is 2 + 2?", "input": "", "output": "2 + 2 equals 4."},
            {"instruction": "Who wrote Hamlet?", "input": "", "output": "William Shakespeare wrote Hamlet."},
            {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
            {"instruction": "Explain gravity in one sentence.", "input": "", "output": "Gravity is the force that attracts masses."},
        ]
    else:
        print(f"[Dataset] Đang tải {args.max_samples} mẫu từ 'yahma/alpaca-cleaned' qua HuggingFace datasets...")
        try:
            split_str = f"train[:{args.max_samples}]"
            ds = load_dataset("yahma/alpaca-cleaned", split=split_str)
            raw_samples = ds
            print(f"[Dataset] Tải thành công {len(raw_samples)} mẫu huấn luyện.")
        except Exception as e:
            print(f"[Dataset] Lỗi tải online ({e}), chuyển sang chế độ dữ liệu giả lập...")
            raw_samples = [
                {"instruction": "What is 2 + 2?", "input": "", "output": "2 + 2 equals 4."},
                {"instruction": "Who wrote Hamlet?", "input": "", "output": "William Shakespeare wrote Hamlet."},
                {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
                {"instruction": "Explain gravity in one sentence.", "input": "", "output": "Gravity is the force that attracts masses."},
            ]

    dataset = SFTDataset(
        samples=raw_samples,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 4. Khởi tạo mô hình Transformer theo đúng kiến trúc chuẩn
    model = Transformer(
        vocab_size=12000,
        d_model=384,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=args.max_seq_len,
        dropout=0.0,
    ).to(device)

    # 5. Nạp Base Model từ checkpoint Tuần 2 (checkpoint_step_50000.pt hoặc mới nhất)
    base_ckpt = find_base_checkpoint(args.base_ckpt)
    if base_ckpt:
        print(f"[Base Model] Đang nạp trọng số từ {base_ckpt}...")
        ckpt_data = torch.load(base_ckpt, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            model.load_state_dict(ckpt_data)
        print("[Base Model] Nạp checkpoint base thành công!")
    else:
        print("[Base Model] Cảnh báo: Không tìm thấy checkpoint pre-train Tuần 2, khởi tạo trọng số ngẫu nhiên.")

    # 6. Gọi hàm apply_lora từ Task 3.1:
    # - Đóng băng toàn bộ trọng số gốc của base model
    # - Thay thế q_proj và v_proj trong CausalSelfAttention bằng LoRALinear
    # - In ra tổng số tham số, tham số LoRA, và phần trăm tham số được train
    apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    # 7. Khởi tạo Optimizer (chỉ tối ưu các tham số requires_grad = True của LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Thiết lập mixed precision
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device_type == "cuda")
    scaler = torch.amp.GradScaler(device=device_type, enabled=(use_amp and amp_dtype == torch.float16))

    # 8. Tính năng Resume Training SFT
    start_epoch = 1
    global_step = 0
    if args.resume:
        resume_file = Path(args.resume)
        if not resume_file.exists() and (REPO_ROOT / resume_file).exists():
            resume_file = REPO_ROOT / resume_file

        if resume_file.exists():
            print(f"[Resume] Đang nạp SFT checkpoint từ {resume_file}...")
            ckpt = torch.load(resume_file, map_location=device, weights_only=False)
            if "lora_state_dict" in ckpt:
                model.load_state_dict(ckpt["lora_state_dict"], strict=False)
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            global_step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 1))
            print(f"[Resume] Nạp thành công! Tiếp tục từ Epoch {start_epoch}, Step {global_step + 1}...")
        else:
            raise FileNotFoundError(f"Không tìm thấy SFT checkpoint để resume: {args.resume}")

    # 9. Vòng lặp huấn luyện SFT (1-2 epochs)
    print("\n" + "=" * 65)
    print(f"BẮT ĐẦU HUẤN LUYỆN SFT ({args.epochs} EPOCHS, {len(dataloader)} BATCHES/EPOCH)")
    print("=" * 65)

    model.train()
    total_steps_in_epoch = len(dataloader) // args.grad_accum

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        running_loss = 0.0
        accum_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                # CRITICAL: logits.float() trong fp16 để tránh overflow sinh NaN loss.
                # ignore_index = -100 đảm bảo bỏ qua toàn bộ prompt và padding, chỉ tính loss trên response!
                logits_f32 = logits.float()
                loss = F.cross_entropy(
                    logits_f32.view(-1, model.vocab_size),
                    y.view(-1),
                    ignore_index=-100,
                )
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / args.grad_accum

            # Cập nhật gradient sau mỗi grad_accum bước
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += accum_loss

                # Log tiến độ
                if global_step % 20 == 0 or args.smoke_test:
                    print(
                        f"Epoch {epoch:2d}/{args.epochs} | Step {global_step:5d} | "
                        f"Loss: {accum_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )

                # Lưu checkpoint định kỳ
                if global_step % args.save_every == 0 and not args.smoke_test:
                    ckpt_save_path = out_dir / f"sft_step_{global_step}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "lora_state_dict": get_lora_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                            "loss": accum_loss,
                        },
                        ckpt_save_path,
                    )
                    # Đồng thời cập nhật latest_sft_checkpoint.pt và lora adapter
                    torch.save(get_lora_state_dict(model), out_dir / "lora_sft_adapter.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "lora_state_dict": get_lora_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        out_dir / "latest_sft_checkpoint.pt",
                    )
                    print(f"[Checkpoint] Đã lưu SFT checkpoint tại step {global_step} vào {ckpt_save_path}")

                accum_loss = 0.0

            if args.smoke_test and global_step >= 3:
                break

        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / max(1, total_steps_in_epoch)
        print(f"\n--- Kết thúc Epoch {epoch}/{args.epochs} | Loss trung bình: {avg_loss:.4f} | Thời gian: {epoch_time:.2f}s ---")

        # Lưu checkpoint sau mỗi epoch
        epoch_ckpt_path = out_dir / f"sft_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": global_step,
                "lora_state_dict": get_lora_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            epoch_ckpt_path,
        )
        torch.save(get_lora_state_dict(model), out_dir / "lora_sft_adapter.pt")
        print(f"[Checkpoint] Đã lưu SFT Adapter ra {out_dir / 'lora_sft_adapter.pt'}")

        if args.smoke_test:
            break

    print("\n" + "=" * 65)
    print("HOÀN THÀNH HUẤN LUYỆN SFT!")
    print(f"LoRA Adapter cuối cùng đã được lưu an toàn tại: {out_dir / 'lora_sft_adapter.pt'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
