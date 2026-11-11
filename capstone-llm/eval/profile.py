"""Performance profiling script measuring TTFT, TPOT, throughput, and VRAM across context lengths."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.kv_cache import KVCache
from model.transformer import Transformer


def benchmark_generation(
    model: Transformer,
    prompt_ids: List[int],
    max_new_tokens: int = 32,
    use_kv_cache: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, float]:
    """Measure first token latency (TTFT), next token latency (TPOT), throughput, and peak memory."""
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    n_heads = model.layers[0].attn.n_heads
    head_dim = model.layers[0].attn.head_dim
    dtype = model.embedding.weight.dtype
    total_len = len(prompt_ids) + max_new_tokens

    # CRITICAL: Đo first_token_latency_ms (Time to First Token - TTFT) riêng biệt.
    # Đây là thời gian từ lúc bắt đầu prefill prompt cho đến khi sinh xong token đầu tiên.
    # Metric bắt buộc trong đề bài phản ánh độ trễ pha prefill khi độ dài context L tăng dần.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    if use_kv_cache:
        kv_caches = [
            KVCache(
                batch_size=1,
                n_heads=n_heads,
                head_dim=head_dim,
                max_len=total_len,
                device=device,
                dtype=dtype,
            )
            for _ in range(len(model.layers))
        ]

        # Pha 1: Prefill prompt và sinh token đầu tiên
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor, kv_caches=kv_caches)
            next_id = int(torch.argmax(logits[0, -1, :], dim=-1).item())

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter()
        first_token_latency_ms = (t_first - t_start) * 1000.0

        # Pha 2: Loop sinh các token tiếp theo bằng cache
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits = model(next_tensor, kv_caches=kv_caches)
                next_id = int(torch.argmax(logits[0, -1, :], dim=-1).item())

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

    else:
        generated = list(prompt_ids)

        # Pha 1: Forward prompt ban đầu và sinh token đầu tiên
        input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor)
            next_id = int(torch.argmax(logits[0, -1, :], dim=-1).item())
            generated.append(next_id)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter()
        first_token_latency_ms = (t_first - t_start) * 1000.0

        # Pha 2: Forward lại toàn bộ chuỗi cho các token sau
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                logits = model(input_tensor)
                next_id = int(torch.argmax(logits[0, -1, :], dim=-1).item())
                generated.append(next_id)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

    total_time_sec = t_end - t_start
    decode_time_sec = t_end - t_first
    next_token_count = max(1, max_new_tokens - 1)
    next_token_latency_ms = (decode_time_sec * 1000.0) / next_token_count
    throughput_tokens_per_sec = max_new_tokens / total_time_sec if total_time_sec > 0 else 0.0

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    return first_token_latency_ms, next_token_latency_ms, throughput_tokens_per_sec, peak_vram_mb


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for profiling."""
    parser = argparse.ArgumentParser(description="Profile LLM inference speed and memory")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--lengths", type=str, default="128,512,1024,2048", help="Comma-separated context lengths")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of benchmark trials per configuration")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run profiling on (cuda, mps, cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function running benchmarks and printing tabular report."""
    args = parse_args()
    device = torch.device(args.device)
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    max_len_needed = max(lengths) + 64

    print(f"Device: {device} | Trials per setting: {args.num_trials} | Context lengths: {lengths}")

    # 1. Khởi tạo mô hình
    model = Transformer(
        vocab_size=12000,
        d_model=384,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=max_len_needed,
        dropout=0.0,
    ).to(device)

    # 2. Nạp checkpoint nếu có
    if args.ckpt and Path(args.ckpt).exists():
        print(f"Loading checkpoint from {args.ckpt}...")
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    results: List[Dict[str, Any]] = []

    # 3. Chạy benchmark cho từng chiều dài L và từng chế độ cache
    for L in lengths:
        # Tạo prompt ngẫu nhiên độ dài L
        torch.manual_seed(42)
        prompt_ids = torch.randint(0, model.vocab_size, (L,)).tolist()

        for use_kv_cache in [True, False]:
            ttft_list, tpot_list, throughput_list, vram_list = [], [], [], []

            # Warmup 1 lần để tránh overhead khởi tạo CUDA/kernel
            benchmark_generation(model, prompt_ids, max_new_tokens=4, use_kv_cache=use_kv_cache, device=device)

            for _ in range(args.num_trials):
                ttft, tpot, throughput, vram = benchmark_generation(
                    model=model,
                    prompt_ids=prompt_ids,
                    max_new_tokens=16,
                    use_kv_cache=use_kv_cache,
                    device=device,
                )
                ttft_list.append(ttft)
                tpot_list.append(tpot)
                throughput_list.append(throughput)
                vram_list.append(vram)

            row = {
                "length": L,
                "use_kv_cache": use_kv_cache,
                "first_token_ms": statistics.median(ttft_list),
                "next_token_ms": statistics.median(tpot_list),
                "throughput": statistics.median(throughput_list),
                "peak_vram_mb": statistics.median(vram_list),
            }
            results.append(row)

    # 4. In bảng kết quả căn lề đẹp mắt ra stdout
    print("\n" + "=" * 90)
    print(
        f"{'Context (L)':<12} | {'KV Cache':<10} | {'TTFT (ms)':<14} | "
        f"{'TPOT (ms)':<14} | {'Tok/sec':<14} | {'Peak VRAM (MB)':<14}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['length']:<12} | {str(r['use_kv_cache']):<10} | "
            f"{r['first_token_ms']:<14.2f} | {r['next_token_ms']:<14.2f} | "
            f"{r['throughput']:<14.2f} | {r['peak_vram_mb']:<14.2f}"
        )
    print("=" * 90 + "\n")

    # 5. Lưu kết quả ra file JSON
    out_json = REPO_ROOT / "eval" / "profile_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Đã lưu kết quả đo đạc ra {out_json}")


if __name__ == "__main__":
    main()
