"""Direct Preference Optimization (DPO) training pipeline using PyTorch and LoRA."""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.lora import apply_lora, get_lora_state_dict
from model.transformer import Transformer
from training.dpo_loss import dpo_loss


class DPODataset(Dataset):
    """Preference dataset holding (prompt, chosen, rejected) triplets with response masks."""

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
        self.data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for item in samples:
            prompt = item.get("prompt", "").strip()
            chosen = item.get("chosen", "").strip()
            rejected = item.get("rejected", "").strip()
            if not prompt or not chosen or not rejected:
                continue

            prompt_text = f"User: {prompt}\nAssistant: "
            prompt_ids = self.tokenizer.encode(prompt_text).ids

            # Chuẩn bị nhánh chosen
            chosen_ids = self.tokenizer.encode(f"{chosen}<|eos|>").ids
            c_input, c_target, c_mask = self._build_sequence(prompt_ids, chosen_ids)

            # Chuẩn bị nhánh rejected
            rejected_ids = self.tokenizer.encode(f"{rejected}<|eos|>").ids
            r_input, r_target, r_mask = self._build_sequence(prompt_ids, rejected_ids)

            self.data.append((c_input, c_target, c_mask, r_input, r_target, r_mask))

    def _build_sequence(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct shifted input, target, and binary response mask."""
        full_ids = prompt_ids + response_ids
        # CRITICAL: Mask bằng 1 trên response và 0 trên prompt để chỉ cộng dồn log-prob trên response
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]
            mask = mask[: self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(full_ids)
            full_ids = full_ids + [self.pad_token_id] * pad_len
            mask = mask + [0] * pad_len

        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:], dtype=torch.long)
        # Shift mask tương ứng với target y
        m = torch.tensor(mask[1:], dtype=torch.float32)
        return x, y, m

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def compute_sequence_logps(
    model: Transformer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log probabilities for the masked response tokens."""
    # input_ids: (B, T), target_ids: (B, T), mask: (B, T)
    logits = model(input_ids)  # (B, T, vocab_size)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Lấy log-prob tại các token mục tiêu
    per_token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    # CRITICAL: Nhân với mask để chỉ tính tổng log-likelihood trên các token phản hồi, bỏ qua prompt và pad
    response_logps = (per_token_logps * mask).sum(dim=-1)
    return response_logps


def train_dpo(
    policy_model: Transformer,
    ref_model: Transformer,
    dataset: DPODataset,
    batch_size: int = 2,
    max_steps: int = 10,
    lr: float = 1e-4,
    beta: float = 0.1,
    device: torch.device = torch.device("cpu"),
    out_dir: str = "checkpoints/dpo",
) -> List[float]:
    """Execute DPO training loop aligning the policy model against frozen reference model."""
    policy_model.to(device)
    ref_model.to(device)

    # NOTE: Reference model phải chạy no_grad và requires_grad=False hoàn toàn
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_model.train()
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in policy_model.parameters())
    print(f"Tham số tối ưu DPO LoRA: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=(use_amp and amp_dtype == torch.float16))

    losses = []
    t0 = time.time()

    for step in range(max_steps):
        try:
            c_x, c_y, c_m, r_x, r_y, r_m = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            c_x, c_y, c_m, r_x, r_y, r_m = next(data_iter)

        c_x, c_y, c_m = c_x.to(device), c_y.to(device), c_m.to(device)
        r_x, r_y, r_m = r_x.to(device), r_y.to(device), r_m.to(device)

        # 1. Tính log-prob từ Reference Model (không tính gradient)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                ref_chosen_logps = compute_sequence_logps(ref_model, c_x, c_y, c_m)
                ref_rejected_logps = compute_sequence_logps(ref_model, r_x, r_y, r_m)

        optimizer.zero_grad(set_to_none=True)

        # 2. Tính log-prob từ Policy Model (có gradient cho LoRA)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            policy_chosen_logps = compute_sequence_logps(policy_model, c_x, c_y, c_m)
            policy_rejected_logps = compute_sequence_logps(policy_model, r_x, r_y, r_m)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                beta=beta,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        if (step + 1) % max(1, max_steps // 5) == 0 or step == max_steps - 1:
            print(
                f"DPO Step {step + 1:3d}/{max_steps} | Loss: {loss_val:.4f} | "
                f"Reward Margin: {reward_margin:+.4f} | Time: {time.time() - t0:.2f}s"
            )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dpo_adapter = get_lora_state_dict(policy_model)
    save_file = out_path / "lora_dpo_adapter.pt"
    torch.save(dpo_adapter, save_file)
    print(f"Đã lưu DPO LoRA adapter ra {save_file}")

    return losses


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for DPO training."""
    parser = argparse.ArgumentParser(description="Direct Preference Optimization with LoRA")
    parser.add_argument("--sft_ckpt", type=str, default=None, help="Path to SFT model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--out_dir", type=str, default="checkpoints/dpo", help="Output directory for DPO weights")
    parser.add_argument("--max_steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta coefficient")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to train on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Smoke test: Kiểm tra pipeline DPO trên các cặp dữ liệu giả lập
    device = torch.device("cpu")
    tokenizer_file = REPO_ROOT / "tokenizer" / "tokenizer.json"

    sample_preferences = [
        {
            "prompt": "What is the capital of Vietnam?",
            "chosen": "The capital of Vietnam is Hanoi.",
            "rejected": "I think it is Ho Chi Minh City.",
        },
        {
            "prompt": "How to live a healthy life?",
            "chosen": "Eat balanced food, exercise daily, and sleep 8 hours.",
            "rejected": "Just skip meals and drink energy drinks.",
        },
    ]

    dpo_ds = DPODataset(
        samples=sample_preferences,
        tokenizer_path=str(tokenizer_file),
        max_seq_len=64,
    )

    # 1. Khởi tạo Reference Model (đóng băng hoàn toàn)
    ref = Transformer(vocab_size=12000, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64)

    # 2. Khởi tạo Policy Model (gắn LoRA)
    policy = Transformer(vocab_size=12000, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64)
    policy.load_state_dict(ref.state_dict())

    # Đóng băng trọng số gốc của policy và gắn LoRA
    for p in policy.parameters():
        p.requires_grad = False
    apply_lora(policy, target_modules=["q_proj", "v_proj"], r=4, alpha=8.0)

    print("Running DPO smoke test (3 steps)...")
    dpo_losses = train_dpo(
        policy_model=policy,
        ref_model=ref,
        dataset=dpo_ds,
        batch_size=2,
        max_steps=3,
        lr=5e-4,
        beta=0.1,
        device=device,
        out_dir="checkpoints/dpo_test",
    )

    assert len(dpo_losses) == 3
    assert all(not math.isnan(l) and not math.isinf(l) for l in dpo_losses)
    print("DPO smoke test passed successfully!")
