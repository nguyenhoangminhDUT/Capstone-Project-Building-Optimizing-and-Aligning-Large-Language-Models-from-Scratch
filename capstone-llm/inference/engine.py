"""Autoregressive text generation engine supporting greedy, temperature, top-k, top-p, and KV Cache."""

import sys
from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn.functional as F

# Đảm bảo import được các module từ thư mục capstone-llm
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import Transformer
from inference.kv_cache import KVCache


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample next token from logits using temperature, top-k, and nucleus (top-p) filtering."""
    # 1. Greedy decoding khi temperature <= 0
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Điều chỉnh nhiệt độ (temperature scaling)
    logits = logits / temperature

    # 3. Top-k filtering: Giữ lại k logit có giá trị lớn nhất
    if top_k is not None and top_k > 0:
        min_top_k = min(top_k, logits.size(-1))
        val_at_k, _ = torch.topk(logits, min_top_k, dim=-1)
        # CRITICAL: Mask bằng float("-inf") để softmax gán xác suất chính xác bằng 0
        cutoff = val_at_k[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    # 4. Top-p (Nucleus) filtering: Giữ lại nhóm token tích lũy xác suất >= top_p
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # CRITICAL: Dịch mask sang phải 1 vị trí để giữ lại ít nhất 1 token đầu tiên
        # (token có xác suất cao nhất), tránh việc toàn bộ logits bị mask thành -inf.
        sorted_mask_to_remove = cumulative_probs > top_p
        sorted_mask_to_remove[..., 1:] = sorted_mask_to_remove[..., :-1].clone()
        sorted_mask_to_remove[..., 0] = False

        # Đưa mask từ thứ tự đã sort trở lại thứ tự gốc của logits
        mask_to_remove = sorted_mask_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_mask_to_remove)
        logits = logits.masked_fill(mask_to_remove, float("-inf"))

    # 5. Phân phối xác suất và lấy mẫu (multinomial sampling)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: Union[torch.Tensor, List[int]],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = 3,
    use_kv_cache: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a prompt with optional KV Cache acceleration."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Chuẩn hóa prompt thành Tensor 2D (batch_size, seq_len)
    if isinstance(prompt_tokens, list):
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    elif prompt_tokens.ndim == 1:
        tokens = prompt_tokens.unsqueeze(0).to(device)
    else:
        tokens = prompt_tokens.to(device)

    batch_size, prompt_len = tokens.shape
    total_len = prompt_len + max_new_tokens

    if total_len > model.max_seq_len:
        raise ValueError(
            f"Tổng độ dài chuỗi ({total_len}) vượt quá max_seq_len của mô hình ({model.max_seq_len})"
        )

    # Khởi tạo KV Cache nếu bật
    kv_cache = None
    if use_kv_cache:
        head_dim = model.d_model // len(model.layers[0].attn.q_proj.weight) * model.d_model  # head_dim
        # Lấy tham số kiến trúc từ Block đầu tiên
        head_dim = model.layers[0].attn.head_dim
        n_heads = model.layers[0].attn.n_heads
        kv_cache = KVCache(
            n_layers=len(model.layers),
            batch_size=batch_size,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=model.max_seq_len,
            device=device,
            dtype=next(model.parameters()).dtype,
        )

    # 1. Pha Prefill: Xử lý toàn bộ chuỗi prompt ban đầu
    if use_kv_cache:
        logits = model(tokens, start_pos=0, kv_cache=kv_cache)
    else:
        logits = model(tokens)

    next_token = sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    generated_tokens = [tokens, next_token]
    current_pos = prompt_len

    # 2. Pha Decode: Lặp sinh từng token mới
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        if use_kv_cache:
            # CRITICAL: Khi dùng KV Cache, chỉ cần đưa đúng 1 token vừa sinh vào model (input shape 1)
            # với start_pos tương ứng với chỉ số bước hiện tại.
            logits = model(next_token, start_pos=current_pos, kv_cache=kv_cache)
        else:
            # Không dùng cache: Phải forward lại toàn bộ lịch sử tokens từ đầu (O(N^2))
            all_tokens = torch.cat(generated_tokens, dim=-1)
            logits = model(all_tokens)

        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_tokens.append(next_token)
        current_pos += 1

    return torch.cat(generated_tokens, dim=-1)


if __name__ == "__main__":
    # Smoke test: So sánh output giữa có KV Cache và không có KV Cache (Greedy)
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = Transformer(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=128,
    ).to(device)

    prompt = torch.randint(0, 1000, (1, 8), device=device)
    max_new = 10

    # 1. Sinh bằng KV Cache
    tokens_with_cache = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=max_new,
        temperature=0.0,
        use_kv_cache=True,
    )

    # 2. Sinh không dùng KV Cache (chậm hơn, duyệt lại toàn bộ chuỗi)
    tokens_no_cache = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=max_new,
        temperature=0.0,
        use_kv_cache=False,
    )

    print(f"Prompt: {prompt.tolist()[0]}")
    print(f"With KV Cache: {tokens_with_cache.tolist()[0]}")
    print(f"No   KV Cache: {tokens_no_cache.tolist()[0]}")

    # Hai kết quả khi greedy decoding BẮT BUỘC phải trùng khớp hoàn toàn từng token
    assert torch.equal(
        tokens_with_cache, tokens_no_cache
    ), "Kết quả sinh của KV Cache không khớp với forward thông thường!"
    print("InferenceEngine KV Cache consistency test passed!")

