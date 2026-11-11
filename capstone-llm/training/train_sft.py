"""Supervised Fine-Tuning (SFT) with LoRA and prompt-loss masking."""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
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


class SFTDataset(Dataset):
    """Instruction dataset with prompt-loss masking (label = -100 on prompt tokens)."""

    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer_path: str,
        max_seq_len: int = 512,
        pad_token_id: int = 1,
    ) -> None:
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.data: List[tuple[torch.Tensor, torch.Tensor]] = []

        for sample in samples:
            instruction = sample.get("instruction", "").strip()
            response = sample.get("response", "").strip()
            if not instruction or not response:
                continue

            prompt_text = f"User: {instruction}\nAssistant: "
            response_text = f"{response}<|eos|>"

            prompt_ids = self.tokenizer.encode(prompt_text).ids
            response_ids = self.tokenizer.encode(response_text).ids

            # CRITICAL: Gán label = -100 cho toàn bộ prompt_ids để hàm F.cross_entropy
            # (ignore_index = -100) tự động bỏ qua việc tính gradient trên câu hỏi của user,
            # đảm bảo mô hình chỉ được tối ưu hóa để sinh câu trả lời (response) chuẩn xác.
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]
            else:
                pad_len = max_seq_len - len(input_ids)
                input_ids = input_ids + [self.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

            # Chuẩn bị cặp x và y dịch phải 1 token cho causal language modeling
            x = torch.tensor(input_ids[:-1], dtype=torch.long)
            y = torch.tensor(labels[1:], dtype=torch.long)
            self.data.append((x, y))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def train_sft(
    model: Transformer,
    dataset: SFTDataset,
    batch_size: int = 2,
    max_steps: int = 10,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    out_dir: str = "checkpoints/sft",
    save_adapter_only: bool = True,
) -> List[float]:
    """Execute SFT training loop optimizing only LoRA adapter parameters."""
    model.to(device)
    model.train()

    # Thu thập chỉ các tham số LoRA cần huấn luyện
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(
        f"Tham số huấn luyện LoRA: {trainable_count:,} / {total_params:,} "
        f"({100 * trainable_count / total_params:.2f}%)"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    data_iter = iter(dataloader)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=(use_amp and amp_dtype == torch.float16))

    losses = []
    t0 = time.time()

    for step in range(max_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            # CRITICAL: ignore_index = -100 để bỏ qua các token prompt và padding
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1), ignore_index=-100)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % max(1, max_steps // 5) == 0 or step == max_steps - 1:
            print(f"SFT Step {step + 1:3d}/{max_steps} | Loss: {loss_val:.4f} | Time: {time.time() - t0:.2f}s")

    # Lưu checkpoint
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_adapter_only:
        lora_state = get_lora_state_dict(model)
        save_file = out_path / "lora_sft_adapter.pt"
        torch.save(lora_state, save_file)
        print(f"Đã lưu LoRA adapter weights ({len(lora_state)} tensors) ra {save_file}")
    else:
        save_file = out_path / "model_sft_full.pt"
        torch.save(model.state_dict(), save_file)
        print(f"Đã lưu toàn bộ model SFT ra {save_file}")

    return losses


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SFT training."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LoRA")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Path to pre-trained base model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--out_dir", type=str, default="checkpoints/sft", help="Directory to save SFT checkpoint")
    parser.add_argument("--max_steps", type=int, default=200, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling factor")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to train on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Smoke test: Kiểm tra quy trình SFT với dữ liệu instruction giả lập
    device = torch.device("cpu")
    tokenizer_file = REPO_ROOT / "tokenizer" / "tokenizer.json"

    # Dữ liệu instruction mẫu
    sample_dialogues = [
        {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
        {"instruction": "Who wrote Hamlet?", "response": "William Shakespeare wrote Hamlet."},
        {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"instruction": "Explain gravity in one sentence.", "response": "Gravity is the force attracting masses."},
    ]

    sft_ds = SFTDataset(
        samples=sample_dialogues,
        tokenizer_path=str(tokenizer_file),
        max_seq_len=64,
    )

    # Khởi tạo Transformer nhỏ để chạy test
    base_model = Transformer(
        vocab_size=12000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
    )

    # 1. Đóng băng toàn bộ trọng số gốc của mô hình
    for p in base_model.parameters():
        p.requires_grad = False

    # 2. Gắn LoRA adapters vào attention layers (chỉ lora_A, lora_B nhận gradient)
    apply_lora(base_model, target_modules=["q_proj", "v_proj"], r=4, alpha=8.0)

    print("Running SFT smoke test (3 steps)...")
    sft_losses = train_sft(
        model=base_model,
        dataset=sft_ds,
        batch_size=2,
        max_steps=3,
        lr=5e-4,
        device=device,
        out_dir="checkpoints/sft_test",
        save_adapter_only=True,
    )

    assert len(sft_losses) == 3
    assert all(not math.isnan(l) and not math.isinf(l) for l in sft_losses)
    print("SFT smoke test passed successfully!")
