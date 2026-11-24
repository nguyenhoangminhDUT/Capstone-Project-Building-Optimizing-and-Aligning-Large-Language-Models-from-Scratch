"""All-in-One Automated Kaggle Pipeline Orchestrator for Capstone LLM.

Automates the entire execution lifecycle on Kaggle (Dual T4 GPUs):
1. Recursively scans /kaggle/input (ignoring __MACOSX and hidden ._* metadata).
2. Auto-copies checkpoints, datasets, and tokenizer files into /kaggle/working/.
3. Detects if Kaggle extracted a .pt checkpoint into a folder (containing data.pkl & version)
   and automatically repacks it into a valid PyTorch zip archive.
4. Sequentially executes:
   - train_tokenizer.py (if tokenizer.json not found)
   - train_pretrain.py  (if best_model.pt not found)
   - train_sft.py       (if lora_sft_adapter.pt not found)
   - train_dpo.py       (if lora_dpo_adapter.pt not found)
   - eval/evaluate.py   (evaluates all 3 models on PPL, Math Pass@1, and Safety)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def is_ignored(path: Path) -> bool:
    """Kiểm tra file hoặc thư mục rác hệ thống (macOS / Kaggle zip metadata)."""
    parts = path.parts
    for p in parts:
        if p == "__MACOSX" or p.startswith("._") or p == ".DS_Store":
            return True
    return False


def repack_pytorch_checkpoint(extracted_dir: Path, target_zip_path: Path) -> None:
    """
    Đóng gói lại thư mục checkpoint bị Kaggle giải nén thành file .pt chuẩn của PyTorch.
    PyTorch checkpoints là file ZIP chứa data.pkl, version, byteorder,... dưới một tiền tố thư mục.
    """
    print(f"[Kaggle Repack] Phát hiện checkpoint dạng thư mục tại: {extracted_dir}")
    print(f"[Kaggle Repack] Đang đóng gói lại vào file chuẩn: {target_zip_path}...")
    target_zip_path.parent.mkdir(parents=True, exist_ok=True)

    has_root_data = (extracted_dir / "data.pkl").exists()
    stem = target_zip_path.stem

    with zipfile.ZipFile(target_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in extracted_dir.rglob("*"):
            if file_path.is_file() and not is_ignored(file_path):
                rel_path = file_path.relative_to(extracted_dir)
                if has_root_data:
                    # Nếu các file nằm ở thư mục gốc của extract, bọc lại dưới tiền tố stem
                    arcname = f"{stem}/{rel_path}"
                else:
                    arcname = str(rel_path)
                zf.write(file_path, arcname)

    size_mb = target_zip_path.stat().st_size / (1024 * 1024)
    print(f"[Kaggle Repack] Đóng gói thành công: {target_zip_path} ({size_mb:.1f} MB)")


def scan_and_sync_kaggle_input(input_root: Path, work_root: Path) -> None:
    """Quét và đồng bộ tài nguyên từ input_root (/kaggle/input) sang work_root (/kaggle/working)."""
    if not input_root.exists():
        print(f"[Storage] Thư mục đầu vào {input_root} không tồn tại. Bỏ qua bước đồng bộ đầu vào.")
        return

    print("=" * 75)
    print(f"QUÉT VÀ ĐỒNG BỘ DỮ LIỆU TỪ: {input_root}")
    print("=" * 75)

    pretrain_dir = work_root / "capstone_checkpoints" / "pretrain"
    sft_dir = work_root / "capstone_checkpoints" / "sft"
    dpo_dir = work_root / "capstone_checkpoints" / "dpo"
    data_dir = work_root / "data"

    pretrain_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quét tìm checkpoint bị giải nén (thư mục chứa data.pkl và version)
    for dirpath, dirnames, filenames in os.walk(input_root):
        current_dir = Path(dirpath)
        if is_ignored(current_dir):
            continue

        if "data.pkl" in filenames and ("version" in filenames or "byteorder" in filenames):
            # Đây là một checkpoint bị giải nén
            name_lower = current_dir.name.lower()
            if "sft" in name_lower:
                target_file = sft_dir / "latest_sft_checkpoint.pt"
            elif "dpo" in name_lower:
                target_file = dpo_dir / "latest_dpo_checkpoint.pt"
            elif "best" in name_lower:
                target_file = pretrain_dir / "best_model.pt"
            else:
                target_file = pretrain_dir / "latest_checkpoint.pt"

            if not target_file.exists():
                repack_pytorch_checkpoint(current_dir, target_file)

    # 2. Quét các file cụ thể (.pt, tokenizer.json, .bin, .txt)
    for file_path in input_root.rglob("*"):
        if not file_path.is_file() or is_ignored(file_path):
            continue

        fname = file_path.name
        fname_lower = fname.lower()

        # Tokenizer
        if fname == "tokenizer.json":
            dest1 = work_root / "tokenizer.json"
            dest2 = REPO_ROOT / "tokenizer" / "tokenizer.json"
            for dest in [dest1, dest2]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(file_path, dest)
                    print(f"[Sync] Copy {fname} -> {dest}")

        # Corpus dữ liệu
        elif fname in ["tinystories_full.bin", "tinystories_pretrain.bin", "tinystories_full.txt"]:
            dest1 = data_dir / fname
            dest2 = REPO_ROOT / "data" / fname
            for dest in [dest1, dest2]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    print(f"[Sync] Đang copy file dữ liệu lớn {fname} ({file_path.stat().st_size / (1024*1024):.1f} MB)...")
                    shutil.copy2(file_path, dest)

        # PyTorch Checkpoint files
        elif fname.endswith(".pt"):
            if "dpo" in fname_lower:
                dest = dpo_dir / fname
            elif "sft" in fname_lower:
                dest = sft_dir / fname
            elif "best_model" in fname_lower:
                dest = pretrain_dir / "best_model.pt"
            else:
                dest = pretrain_dir / fname

            if not dest.exists():
                print(f"[Sync] Copy Checkpoint {fname} ({file_path.stat().st_size / (1024*1024):.1f} MB) -> {dest}")
                shutil.copy2(file_path, dest)


def run_command(cmd: List[str], desc: str) -> None:
    """Thực thi một tiến trình qua subprocess và in log thời gian thực."""
    print("\n" + "=" * 75)
    print(f"TIẾN TRÌNH: {desc}")
    print(f"Lệnh: {' '.join(cmd)}")
    print("=" * 75)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[LỖI] Tiến trình '{desc}' thất bại với mã lỗi {result.returncode} sau {elapsed:.1f}s!")
        sys.exit(result.returncode)

    print(f"\n[THÀNH CÔNG] Hoàn tất '{desc}' trong {elapsed:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="All-in-One Pipeline Runner cho Kaggle (Dual T4 GPUs)")
    parser.add_argument("--input_dir", type=str, default="/kaggle/input", help="Đường dẫn Kaggle Input")
    parser.add_argument("--work_dir", type=str, default="/kaggle/working", help="Đường dẫn Kaggle Working")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--pretrain_steps", type=int, default=5000, help="Số bước pre-training")
    parser.add_argument("--sft_epochs", type=int, default=3, help="Số epoch SFT (mặc định 3)")
    parser.add_argument("--dpo_steps", type=int, default=1000, help="Số bước DPO")
    parser.add_argument("--dry_run", action="store_true", help="Chỉ kiểm tra cấu hình mà không chạy huấn luyện")
    parser.add_argument("--smoke_test", action="store_true", help="Chạy chế độ thử nghiệm siêu nhanh (vài step)")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    work_root = Path(args.work_dir)

    # 1. Đồng bộ file từ /kaggle/input
    scan_and_sync_kaggle_input(input_root, work_root)

    # Các đường dẫn mốc kiểm tra
    tok_path_1 = REPO_ROOT / "tokenizer" / "tokenizer.json"
    tok_path_2 = work_root / "tokenizer.json"

    base_model_path = work_root / "capstone_checkpoints" / "pretrain" / "best_model.pt"
    sft_adapter_path = work_root / "capstone_checkpoints" / "sft" / "lora_sft_adapter.pt"
    dpo_adapter_path = work_root / "capstone_checkpoints" / "dpo" / "lora_dpo_adapter.pt"

    if args.dry_run:
        print("\n[Dry Run] Kiểm tra đường dẫn tài nguyên:")
        print(f"  - Tokenizer tồn tại : {tok_path_1.exists() or tok_path_2.exists()}")
        print(f"  - Base Model tồn tại: {base_model_path.exists()}")
        print(f"  - SFT Adapter       : {sft_adapter_path.exists()}")
        print(f"  - DPO Adapter       : {dpo_adapter_path.exists()}")
        print("[Dry Run] Hoàn tất kiểm tra!")
        return

    # 2. Bước 1: Huấn luyện Tokenizer (nếu chưa có tokenizer.json)
    if not tok_path_1.exists() and not tok_path_2.exists():
        run_command(
            [sys.executable, "tokenizer/train_tokenizer.py"],
            desc="Bước 1/5: Huấn luyện BPE Tokenizer 32k (Alpaca + GSM8K + TinyStories)",
        )
    else:
        print("\n[Bước 1/5] Tokenizer (tokenizer.json) đã sẵn sàng -> Bỏ qua.")

    # 3. Bước 2: Huấn luyện Pre-training 100M (nếu chưa có best_model.pt)
    if not base_model_path.exists():
        cmd = [
            sys.executable,
            "training/train_pretrain.py",
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
            "--max_steps", str(3 if args.smoke_test else args.pretrain_steps),
            "--out_dir", str(work_root / "capstone_checkpoints" / "pretrain"),
        ]
        run_command(cmd, desc="Bước 2/5: Huấn luyện Pre-training 100M LLM (Kaggle T4 x2 DataParallel)")
    else:
        print(f"\n[Bước 2/5] Base Model đã tồn tại ({base_model_path}) -> Bỏ qua.")

    # 4. Bước 3: Huấn luyện SFT LoRA r=32 (nếu chưa có lora_sft_adapter.pt)
    if not sft_adapter_path.exists():
        cmd = [
            sys.executable,
            "training/train_sft.py",
            "--base_ckpt", str(base_model_path),
            "--epochs", str(1 if args.smoke_test else args.sft_epochs),
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
            "--out_dir", str(work_root / "capstone_checkpoints" / "sft"),
        ]
        if args.smoke_test:
            cmd.append("--smoke_test")
        run_command(cmd, desc="Bước 3/5: Huấn luyện SFT LoRA r=32 (Alpaca 20k + Math 10k)")
    else:
        print(f"\n[Bước 3/5] SFT LoRA Adapter đã tồn tại ({sft_adapter_path}) -> Bỏ qua.")

    # 5. Bước 4: Huấn luyện DPO Alignment (nếu chưa có lora_dpo_adapter.pt)
    if not dpo_adapter_path.exists():
        cmd = [
            sys.executable,
            "training/train_dpo.py",
            "--base_ckpt", str(base_model_path),
            "--sft_ckpt", str(sft_adapter_path),
            "--max_steps", str(3 if args.smoke_test else args.dpo_steps),
            "--beta", "0.1",
            "--batch_size", "1",
            "--grad_accum", "64",
            "--out_dir", str(work_root / "capstone_checkpoints" / "dpo"),
        ]
        if args.smoke_test:
            cmd.append("--smoke_test")
        run_command(cmd, desc="Bước 4/5: Huấn luyện DPO Alignment (Anthropic/hh-rlhf, beta=0.1)")
    else:
        print(f"\n[Bước 4/5] DPO LoRA Adapter đã tồn tại ({dpo_adapter_path}) -> Bỏ qua.")

    # 6. Bước 5: Đánh giá định lượng toàn diện
    eval_cmd = [
        sys.executable,
        "eval/evaluate.py",
        "--base_ckpt", str(base_model_path),
        "--sft_ckpt", str(sft_adapter_path),
        "--dpo_ckpt", str(dpo_adapter_path),
        "--out_dir", str(work_root / "eval"),
    ]
    full_sft_path = work_root / "capstone_checkpoints" / "sft" / "full_sft_model.pt"
    if full_sft_path.exists():
        eval_cmd.extend(["--full_ft_ckpt", str(full_sft_path)])
    run_command(eval_cmd, desc="Bước 5/5: Đánh giá định lượng các phiên bản (Validation PPL, Math Pass@1, Safety Refusal)")

    print("\n" + "=" * 75)
    print("HOÀN TẤT TOÀN BỘ PIPELINE CAPSTONE TRÊN KAGGLE!")
    print(f"Các artifacts đã được lưu đầy đủ tại: {work_root}")
    print("=" * 75)


if __name__ == "__main__":
    main()
