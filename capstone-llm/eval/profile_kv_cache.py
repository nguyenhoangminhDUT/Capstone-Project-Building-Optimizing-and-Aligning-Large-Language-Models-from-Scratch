"""Comprehensive KV Cache Profiler (On vs. Off) across context lengths L in {128, 512, 1024, 2048}.

Measures:
- Throughput (Tokens/sec)
- Time to First Token (TTFT, ms)
- Time Per Output Token (TPOT, ms/token)
- Peak VRAM (MB) via torch.cuda.max_memory_allocated()

Exports:
- <out_dir>/kv_cache_profile.json
- <out_dir>/kv_cache_profile.md
- <out_dir>/kv_throughput.png (300 DPI)
- <out_dir>/kv_latency.png (300 DPI)
- <out_dir>/kv_vram.png (300 DPI)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.kv_cache import KVCache
from model.transformer import Transformer


def sync_device(device: torch.device) -> None:
    """Synchronize accelerator for accurate wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def benchmark_single_run(
    model: Transformer,
    prompt_len: int,
    gen_tokens: int,
    use_cache: bool,
    device: torch.device,
    vocab_size: int,
) -> Dict[str, float]:
    """Benchmark a single forward generation pass (TTFT, TPOT, Throughput, Peak VRAM)."""
    model.eval()
    prompt_ids = torch.randint(4, max(5, vocab_size), (1, prompt_len), dtype=torch.long, device=device)

    n_layers = len(model.layers)
    n_heads = model.layers[0].attn.n_heads
    head_dim = model.layers[0].attn.head_dim
    dtype = model.embedding.weight.dtype

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync_device(device)

    t0 = time.perf_counter()

    # Pha 1: Prefill / First Token Latency (TTFT)
    if use_cache:
        total_len = prompt_len + gen_tokens
        kv_caches = [
            KVCache(
                batch_size=1,
                n_heads=n_heads,
                head_dim=head_dim,
                max_len=total_len,
                device=device,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]
        logits = model(prompt_ids, kv_caches=kv_caches)
    else:
        kv_caches = None
        logits = model(prompt_ids)

    next_id = int(torch.argmax(logits[0, -1, :]).item())
    sync_device(device)
    t_first = time.perf_counter()
    ttft_ms = (t_first - t0) * 1000.0

    # Pha 2: Decode remaining (gen_tokens - 1) tokens
    generated_ids = prompt_ids[0].tolist() + [next_id]
    t_decode_start = time.perf_counter()

    for _ in range(max(0, gen_tokens - 1)):
        if use_cache:
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(next_tensor, kv_caches=kv_caches)
        else:
            full_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
            logits = model(full_tensor)

        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated_ids.append(next_id)

    sync_device(device)
    t_end = time.perf_counter()

    decode_sec = t_end - t_decode_start
    total_sec = t_end - t0

    tpot_ms = (decode_sec / max(1, gen_tokens - 1)) * 1000.0
    throughput = gen_tokens / max(1e-9, total_sec)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "throughput_tok_s": round(throughput, 2),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "total_time_ms": round(total_sec * 1000.0, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def plot_charts(results: List[Dict[str, object]], out_dir: Path) -> None:
    """Vẽ 3 biểu đồ chuẩn bài báo (300 DPI) với chú thích số liệu rõ ràng."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    lengths = [r["context_length"] for r in results]
    x_pos = list(range(len(lengths)))
    x_labels = [str(L) for L in lengths]

    thr_on = [r["kv_cache_on"]["throughput_tok_s"] for r in results]
    thr_off = [r["kv_cache_off"]["throughput_tok_s"] for r in results]

    ttft_on = [r["kv_cache_on"]["ttft_ms"] for r in results]
    ttft_off = [r["kv_cache_off"]["ttft_ms"] for r in results]

    tpot_on = [r["kv_cache_on"]["tpot_ms"] for r in results]
    tpot_off = [r["kv_cache_off"]["tpot_ms"] for r in results]

    vram_on = [r["kv_cache_on"]["peak_vram_mb"] for r in results]
    vram_off = [r["kv_cache_off"]["peak_vram_mb"] for r in results]

    color_on = "#1f77b4"
    color_off = "#d62728"

    # 1. Throughput Chart
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = 0.35
    bars_on = ax.bar([i - width / 2 for i in x_pos], thr_on, width, label="KV Cache ON", color=color_on, edgecolor="black", linewidth=0.6)
    bars_off = ax.bar([i + width / 2 for i in x_pos], thr_off, width, label="KV Cache OFF (Baseline)", color=color_off, edgecolor="black", linewidth=0.6)

    for bar in bars_on:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_on)
    for bar in bars_off:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_off)

    ax.set_title("Generation Throughput: KV Cache ON vs. OFF")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Throughput (Tokens / sec) ↑")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_throughput.png", dpi=300)
    plt.close(fig)

    # 2. Latency Chart (2 subplots: TTFT & TPOT)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    ax1.plot(x_pos, ttft_on, marker="o", linewidth=2.2, markersize=7, color=color_on, label="KV Cache ON")
    ax1.plot(x_pos, ttft_off, marker="s", linewidth=2.2, markersize=7, linestyle="--", color=color_off, label="KV Cache OFF")
    for i, (v_on, v_off) in enumerate(zip(ttft_on, ttft_off)):
        ax1.annotate(f"{v_on:.1f}", (i, v_on), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, color=color_on)
        ax1.annotate(f"{v_off:.1f}", (i, v_off), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=9, color=color_off)
    ax1.set_title("First Token Latency (TTFT)")
    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("TTFT (ms) ↓")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend(frameon=True)

    ax2.plot(x_pos, tpot_on, marker="o", linewidth=2.2, markersize=7, color=color_on, label="KV Cache ON")
    ax2.plot(x_pos, tpot_off, marker="s", linewidth=2.2, markersize=7, linestyle="--", color=color_off, label="KV Cache OFF")
    for i, (v_on, v_off) in enumerate(zip(tpot_on, tpot_off)):
        ax2.annotate(f"{v_on:.1f}", (i, v_on), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=9, color=color_on, fontweight="bold")
        ax2.annotate(f"{v_off:.1f}", (i, v_off), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, color=color_off, fontweight="bold")
    ax2.set_title("Average Next Token Latency (TPOT)")
    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("TPOT (ms / token) ↓")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.legend(frameon=True)

    plt.suptitle("Inference Latency Breakdown: KV Cache ON vs. OFF", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Peak VRAM Chart
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars_on = ax.bar([i - width / 2 for i in x_pos], vram_on, width, label="KV Cache ON", color=color_on, edgecolor="black", linewidth=0.6)
    bars_off = ax.bar([i + width / 2 for i in x_pos], vram_off, width, label="KV Cache OFF (Baseline)", color=color_off, edgecolor="black", linewidth=0.6)

    for bar in bars_on:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color_on)
    for bar in bars_off:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color_off)

    ax.set_title("Peak GPU VRAM Usage: KV Cache ON vs. OFF")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "kv_vram.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile KV Cache ON vs OFF across L in {128, 512, 1024, 2048}")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Đường dẫn base checkpoint (tuỳ chọn)")
    parser.add_argument("--gen_tokens", type=int, default=64, help="Số token sinh thêm ở mỗi mốc (mặc định 64)")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048], help="Các mốc độ dài ngữ cảnh L")
    parser.add_argument("--out_dir", type=str, default="eval", help="Thư mục lưu JSON, Markdown và biểu đồ PNG")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Thiết bị benchmark",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Chạy kiểm thử nhanh với mô hình nhỏ trên CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    max_prompt_len = max(args.lengths)
    gen_tokens = 8 if args.smoke_test else args.gen_tokens
    required_max_len = max(2200, max_prompt_len + gen_tokens + 32)

    print("=" * 80)
    print("BENCHMARK HIỆU NĂNG KV CACHE (ON vs. OFF)")
    print(f"Device: {device} | Context Lengths: {args.lengths} | Gen Tokens: {gen_tokens}")
    print(f"RoPE & Attention Max Length: {required_max_len}")
    print("=" * 80)

    if args.smoke_test:
        vocab_size = 1000
        model = Transformer(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_seq_len=required_max_len,
            dropout=0.0,
        ).to(device)
    else:
        vocab_size = 32000
        model = Transformer(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=2048,
            max_seq_len=required_max_len,
            dropout=0.0,
        ).to(device)
        if args.base_ckpt and Path(args.base_ckpt).exists():
            print(f"[Model] Đang nạp checkpoint từ {args.base_ckpt}...")
            ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(clean_sd, strict=False)
            print("[Model] Nạp thành công!")

    model.eval()

    # Warm-up 1 lượt ở độ dài nhỏ nhất
    print("[Warm-up] Đang chạy warm-up 1 lượt...")
    _ = benchmark_single_run(model, prompt_len=min(64, args.lengths[0]), gen_tokens=4, use_cache=True, device=device, vocab_size=vocab_size)
    _ = benchmark_single_run(model, prompt_len=min(64, args.lengths[0]), gen_tokens=4, use_cache=False, device=device, vocab_size=vocab_size)

    results: List[Dict[str, object]] = []

    for L in args.lengths:
        print(f"\n--- Benchmarking Context Length L = {L} (+ {gen_tokens} new tokens -> total {L + gen_tokens}) ---")
        # Warm-up cho từng mốc L
        _ = benchmark_single_run(model, prompt_len=L, gen_tokens=2, use_cache=True, device=device, vocab_size=vocab_size)
        res_on = benchmark_single_run(model, prompt_len=L, gen_tokens=gen_tokens, use_cache=True, device=device, vocab_size=vocab_size)

        _ = benchmark_single_run(model, prompt_len=L, gen_tokens=2, use_cache=False, device=device, vocab_size=vocab_size)
        res_off = benchmark_single_run(model, prompt_len=L, gen_tokens=gen_tokens, use_cache=False, device=device, vocab_size=vocab_size)

        speedup = round(res_on["throughput_tok_s"] / max(1e-6, res_off["throughput_tok_s"]), 2)
        print(
            f"  [KV ON ] Throughput: {res_on['throughput_tok_s']:7.2f} tok/s | "
            f"TTFT: {res_on['ttft_ms']:7.2f} ms | TPOT: {res_on['tpot_ms']:6.2f} ms | VRAM: {res_on['peak_vram_mb']:7.1f} MB"
        )
        print(
            f"  [KV OFF] Throughput: {res_off['throughput_tok_s']:7.2f} tok/s | "
            f"TTFT: {res_off['ttft_ms']:7.2f} ms | TPOT: {res_off['tpot_ms']:6.2f} ms | VRAM: {res_off['peak_vram_mb']:7.1f} MB"
        )
        print(f"  -> Speedup: {speedup:.2f}x")

        results.append({
            "context_length": L,
            "gen_tokens": gen_tokens,
            "kv_cache_on": res_on,
            "kv_cache_off": res_off,
            "speedup": speedup,
        })

    # Lưu JSON
    json_path = out_dir / "kv_cache_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Tạo bảng Markdown
    header = "| Context Length (L) | Mode | Throughput (tok/s) ↑ | TTFT (ms) ↓ | TPOT (ms/tok) ↓ | Peak VRAM (MB) | Speedup |"
    divider = "|---|---|---|---|---|---|---|"
    md_rows = []
    for r in results:
        L = r["context_length"]
        on = r["kv_cache_on"]
        off = r["kv_cache_off"]
        md_rows.append(
            f"| {L} | **KV Cache ON** | {on['throughput_tok_s']:.2f} | {on['ttft_ms']:.2f} | {on['tpot_ms']:.2f} | {on['peak_vram_mb']:.2f} | **{r['speedup']:.2f}x** |"
        )
        md_rows.append(
            f"| {L} | KV Cache OFF | {off['throughput_tok_s']:.2f} | {off['ttft_ms']:.2f} | {off['tpot_ms']:.2f} | {off['peak_vram_mb']:.2f} | 1.00x |"
        )

    md_content = (
        "# KV Cache Profiling Report (100M Transformer)\n\n"
        f"- Device: `{device}`\n"
        f"- Generated tokens per run: `{gen_tokens}`\n\n"
        f"{header}\n{divider}\n" + "\n".join(md_rows) + "\n"
    )
    md_path = out_dir / "kv_cache_profile.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Vẽ 3 biểu đồ chuẩn 300 DPI
    plot_charts(results, out_dir)

    print("\n" + "=" * 80)
    print("HOÀN THÀNH BENCHMARK KV CACHE!")
    print(f"  - JSON      : {json_path}")
    print(f"  - Markdown  : {md_path}")
    print(f"  - Charts    : {out_dir / 'kv_throughput.png'}, {out_dir / 'kv_latency.png'}, {out_dir / 'kv_vram.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
