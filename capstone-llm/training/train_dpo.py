"""Direct Preference Optimization (DPO) training pipeline with LoRA, Google Drive Checkpointing, and Resume Training."""

import argparse
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

from model.lora import apply_lora, get_lora_state_dict
from model.transformer import Transformer
from training.dpo_loss import dpo_loss

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
        return False
    except Exception as e:
        print(f"[Google Drive] Lưu ý khi mount Drive: {e}")
        return False


class DPODataset(Dataset):
    """
    Dataset cho DPO (Direct Preference Optimization).
    Mỗi mẫu trả về 4 tensor:
      1. chosen_input_ids: Input tokens cho nhánh chosen
      2. chosen_labels: Target labels cho nhánh chosen (prompt và pad gán nhãn -100)
      3. rejected_input_ids: Input tokens cho nhánh rejected
      4. rejected_labels: Target labels cho nhánh rejected (prompt và pad gán nhãn -100)
    """

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

    @staticmethod
    def _split_prompt_and_response(text: str) -> Tuple[str, str]:
        """Tách chuỗi đối thoại của Anthropic/hh-rlhf thành phần prompt và response."""
        # Chuẩn đối thoại của Anthropic: lượt trả lời cuối bắt đầu sau '\n\nAssistant:'
        if "\n\nAssistant:" in text:
            parts = text.rsplit("\n\nAssistant:", 1)
            prompt = parts[0] + "\n\nAssistant:"
            response = parts[1]
            return prompt.strip(), response.strip()
        elif "Assistant:" in text:
            parts = text.rsplit("Assistant:", 1)
            prompt = parts[0] + "Assistant:"
            response = parts[1]
            return prompt.strip(), response.strip()
        else:
            return "", text.strip()

    def _encode_pair(self, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_text = f"{prompt} " if prompt else ""
        response_text = f"{response}<|eos|>"

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response_text).ids

        # Giới hạn prompt để không chiếm hết độ dài context, đảm bảo còn chỗ cho response
        max_prompt_len = max(16, self.max_seq_len - 64)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        if len(response_ids) == 0:
            response_ids = [self.eos_token_id]

        input_ids = prompt_ids + response_ids
        # CRITICAL: Gán nhãn -100 cho toàn bộ token thuộc prompt
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]

        if "prompt" in item:
            # Dạng dict giả lập {prompt, chosen, rejected}
            prompt = str(item.get("prompt", "")).strip()
            chosen_resp = str(item.get("chosen", "")).strip()
            rejected_resp = str(item.get("rejected", "")).strip()
        else:
            # Dạng Anthropic/hh-rlhf chứa toàn bộ cuộc hội thoại trong 'chosen' và 'rejected'
            chosen_full = str(item.get("chosen", "")).strip()
            rejected_full = str(item.get("rejected", "")).strip()

            prompt_c, resp_c = self._split_prompt_and_response(chosen_full)
            prompt_r, resp_r = self._split_prompt_and_response(rejected_full)

            prompt = prompt_c if prompt_c else prompt_r
            chosen_resp = resp_c
            rejected_resp = resp_r

        chosen_input_ids, chosen_labels = self._encode_pair(prompt, chosen_resp)
        rejected_input_ids, rejected_labels = self._encode_pair(prompt, rejected_resp)

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def compute_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Tính tổng log-xác suất (sum of log probabilities) của các token trong phần trả lời (response).
    - model: Transformer model (Policy hoặc Reference).
    - input_ids: Tensor (B, T) chứa token IDs đầu vào.
    - labels: Tensor (B, T) nhãn mục tiêu (token prompt và pad có giá trị -100).
    """
    logits = model(input_ids)  # shape: (B, T, vocab_size)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Mặt nạ nhãn hợp lệ: chỉ giữ lại vị trí khác -100 (tức các token phản hồi)
    loss_mask = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # CRITICAL: Nhân với loss_mask để chỉ cộng dồn log-prob trên response,
    # hoàn toàn bỏ qua các token thuộc prompt và padding.
    response_logps = (per_token_logps * loss_mask).sum(dim=-1)
    return response_logps


def find_sft_checkpoint(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Tìm checkpoint SFT phù hợp nhất (ưu tiên explicit -> sft folder trên Drive/local)."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        if (REPO_ROOT / p).exists():
            return REPO_ROOT / p

    candidates = [
        "/content/drive/MyDrive/capstone_checkpoints/sft/lora_sft_adapter.pt",
        "/content/drive/MyDrive/capstone_checkpoints/sft/latest_sft_checkpoint.pt",
        "checkpoints/sft/lora_sft_adapter.pt",
        "checkpoints/sft/latest_sft_checkpoint.pt",
        str(REPO_ROOT / "checkpoints/sft/lora_sft_adapter.pt"),
        str(REPO_ROOT / "checkpoints/sft/latest_sft_checkpoint.pt"),
    ]
    for c in candidates:
        cp = Path(c)
        if cp.exists():
            return cp
    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for DPO training."""
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) with LoRA")
    parser.add_argument("--sft_ckpt", type=str, default=None, help="Đường dẫn tới SFT checkpoint (Task 3.2)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Đường dẫn tới tokenizer.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/content/drive/MyDrive/capstone_checkpoints/dpo",
        help="Thư mục lưu DPO checkpoint trên Google Drive",
    )
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn tới DPO checkpoint để resume")
    parser.add_argument("--max_steps", type=int, default=1000, help="Tổng số bước huấn luyện DPO")
    parser.add_argument("--max_samples", type=int, default=5000, help="Số lượng mẫu tải từ Anthropic/hh-rlhf")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate cho policy LoRA")
    parser.add_argument("--beta", type=float, default=0.1, help="Hệ số DPO beta nhiệt độ")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank LoRA")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Alpha LoRA")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Độ dài ngữ cảnh tối đa")
    parser.add_argument("--save_every", type=int, default=200, help="Số bước lưu checkpoint định kỳ")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device huấn luyện (cuda, mps, cpu)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Chạy thử nghiệm nhanh với dữ liệu giả lập")
    return parser.parse_args()


def main() -> None:
    """Vòng lặp chính huấn luyện DPO."""
    args = parse_args()

    # 1. Tự động kết nối Google Drive nếu đang chạy trong Colab
    mount_google_drive()

    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        out_dir = Path("checkpoints/dpo")
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Checkpoints] Thư mục lưu kết quả: {out_dir}")

    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    print(f"Device: {device} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Beta: {args.beta} | LR: {args.lr} | Max Steps: {args.max_steps}")

    # 2. Nạp Tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists() and (REPO_ROOT / tokenizer_path).exists():
        tokenizer_path = REPO_ROOT / tokenizer_path
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tokenizer tại: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"[Tokenizer] Nạp thành công từ {tokenizer_path} (vocab_size={tokenizer.get_vocab_size()})")

    # 3. Tải dataset Anthropic/hh-rlhf (5,000 mẫu)
    if args.smoke_test or load_dataset is None:
        print("[Dataset] Chạy chế độ smoke test với mẫu preference giả lập...")
        raw_samples = [
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
    else:
        print(f"[Dataset] Đang tải {args.max_samples} mẫu từ 'Anthropic/hh-rlhf'...")
        try:
            split_str = f"train[:{args.max_samples}]"
            ds = load_dataset("Anthropic/hh-rlhf", split=split_str)
            raw_samples = ds
            print(f"[Dataset] Tải thành công {len(raw_samples)} mẫu preference.")
        except Exception as e:
            print(f"[Dataset] Lỗi tải online ({e}), chuyển sang dữ liệu giả lập...")
            raw_samples = [
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

    dataset = DPODataset(
        samples=raw_samples,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_iter = iter(dataloader)

    # 4. Khởi tạo mô hình policy_model và ref_model
    def build_transformer():
        return Transformer(
            vocab_size=12000,
            d_model=384,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            max_seq_len=args.max_seq_len,
            dropout=0.0,
        )

    print("\nKhởi tạo Policy Model và Reference Model...")
    policy_model = build_transformer().to(device)
    ref_model = build_transformer().to(device)

    # Gắn LoRA vào policy model và ref model
    apply_lora(policy_model, r=args.lora_r, alpha=args.lora_alpha)
    apply_lora(ref_model, r=args.lora_r, alpha=args.lora_alpha)

    # Nạp SFT checkpoint (Task 3.2) nếu có
    sft_ckpt = find_sft_checkpoint(args.sft_ckpt)
    if sft_ckpt:
        print(f"[SFT Checkpoint] Đang nạp trọng số SFT từ {sft_ckpt}...")
        sft_data = torch.load(sft_ckpt, map_location=device, weights_only=False)
        if "lora_state_dict" in sft_data:
            policy_model.load_state_dict(sft_data["lora_state_dict"], strict=False)
            ref_model.load_state_dict(sft_data["lora_state_dict"], strict=False)
        elif "model_state_dict" in sft_data:
            policy_model.load_state_dict(sft_data["model_state_dict"], strict=False)
            ref_model.load_state_dict(sft_data["model_state_dict"], strict=False)
        elif isinstance(sft_data, dict):
            policy_model.load_state_dict(sft_data, strict=False)
            ref_model.load_state_dict(sft_data, strict=False)
        print("[SFT Checkpoint] Nạp thành công!")
    else:
        print("[SFT Checkpoint] Cảnh báo: Không tìm thấy checkpoint SFT, sử dụng trọng số khởi tạo ban đầu.")

    # CRITICAL: Reference model đóng băng 100% tham số và chuyển sang chế độ eval()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Policy model ở chế độ train()
    policy_model.train()
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Thiết lập Mixed Precision
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device_type == "cuda")
    scaler = torch.amp.GradScaler(device=device_type, enabled=(use_amp and amp_dtype == torch.float16))

    # 5. Tính năng Resume Training DPO
    start_step = 0
    if args.resume:
        resume_file = Path(args.resume)
        if not resume_file.exists() and (REPO_ROOT / resume_file).exists():
            resume_file = REPO_ROOT / resume_file

        if resume_file.exists():
            print(f"[Resume] Đang nạp DPO checkpoint từ {resume_file}...")
            ckpt = torch.load(resume_file, map_location=device, weights_only=False)
            if "lora_state_dict" in ckpt:
                policy_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_step = int(ckpt.get("step", 0))
            print(f"[Resume] Nạp thành công! Tiếp tục từ Step {start_step + 1}...")
        else:
            raise FileNotFoundError(f"Không tìm thấy DPO checkpoint để resume: {args.resume}")

    # 6. Vòng lặp huấn luyện DPO
    print("\n" + "=" * 65)
    print(f"BẮT ĐẦU HUẤN LUYỆN DPO (BETA={args.beta}, MAX_STEPS={args.max_steps})")
    print("=" * 65)

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_c_rew = 0.0
    accum_r_rew = 0.0

    t0 = time.time()
    steps_to_run = 3 if args.smoke_test else args.max_steps

    for step in range(start_step + 1, steps_to_run + 1):
        for _ in range(args.grad_accum):
            try:
                c_x, c_y, r_x, r_y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                c_x, c_y, r_x, r_y = next(data_iter)

            c_x, c_y = c_x.to(device), c_y.to(device)
            r_x, r_y = r_x.to(device), r_y.to(device)

            # 1. Tính logps từ Reference Model (trong context torch.no_grad())
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    ref_chosen_logps = compute_logps(ref_model, c_x, c_y)
                    ref_rejected_logps = compute_logps(ref_model, r_x, r_y)

            # 2. Tính logps từ Policy Model (có gradient cho LoRA)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                policy_chosen_logps = compute_logps(policy_model, c_x, c_y)
                policy_rejected_logps = compute_logps(policy_model, r_x, r_y)

                # 3. Tính DPO loss và implicit rewards
                loss, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    ref_chosen_logps=ref_chosen_logps,
                    ref_rejected_logps=ref_rejected_logps,
                    beta=args.beta,
                )
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()

            accum_loss += loss.item() / args.grad_accum
            accum_c_rew += chosen_rewards.mean().item() / args.grad_accum
            accum_r_rew += rejected_rewards.mean().item() / args.grad_accum

        # Cập nhật gradient sau mỗi grad_accum bước
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        reward_margin = accum_c_rew - accum_r_rew

        # Log các chỉ số (loss, chosen_rewards, rejected_rewards, reward_margin)
        if step % 10 == 0 or args.smoke_test:
            print(
                f"DPO Step {step:5d}/{steps_to_run} | "
                f"Loss: {accum_loss:.4f} | "
                f"Chosen Reward: {accum_c_rew:+.4f} | "
                f"Rejected Reward: {accum_r_rew:+.4f} | "
                f"Reward Margin: {reward_margin:+.4f} | "
                f"Time: {time.time() - t0:.2f}s"
            )

        # Lưu checkpoint định kỳ
        if step % args.save_every == 0 and not args.smoke_test:
            ckpt_path = out_dir / f"dpo_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "lora_state_dict": get_lora_state_dict(policy_model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "loss": accum_loss,
                    "reward_margin": reward_margin,
                },
                ckpt_path,
            )
            torch.save(get_lora_state_dict(policy_model), out_dir / "lora_dpo_adapter.pt")
            torch.save(
                {
                    "step": step,
                    "lora_state_dict": get_lora_state_dict(policy_model),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                out_dir / "latest_dpo_checkpoint.pt",
            )
            print(f"[Checkpoint] Đã lưu DPO checkpoint tại step {step} ra {ckpt_path}")

        accum_loss = 0.0
        accum_c_rew = 0.0
        accum_r_rew = 0.0

    # Lưu checkpoint DPO cuối cùng
    final_adapter_path = out_dir / "lora_dpo_adapter.pt"
    torch.save(get_lora_state_dict(policy_model), final_adapter_path)
    print("\n" + "=" * 65)
    print("HOÀN THÀNH HUẤN LUYỆN DPO!")
    print(f"DPO LoRA Adapter đã được lưu tại: {final_adapter_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
