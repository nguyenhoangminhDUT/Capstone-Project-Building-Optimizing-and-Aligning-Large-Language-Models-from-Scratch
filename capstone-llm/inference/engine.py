"""Autoregressive text generation engine supporting KV Cache and baseline full-sequence forward."""

import sys
from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import Transformer
from inference.kv_cache import KVCache


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> int:
    """Sample a single token index from logits using greedy, temperature, or top-k filtering."""
    # Greedy decoding khi temperature = 0
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    # Scale logits theo temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        val_at_k, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < val_at_k[..., [-1]], float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_ids: Union[List[int], torch.Tensor],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    use_kv_cache: bool = True,
) -> List[int]:
    """Generate tokens autoregressively from prompt IDs with optional KV Cache acceleration."""
    model.eval()
    device = next(model.parameters()).device

    if isinstance(prompt_ids, torch.Tensor):
        tokens_list = prompt_ids.squeeze().tolist()
        if isinstance(tokens_list, int):
            tokens_list = [tokens_list]
    else:
        tokens_list = list(prompt_ids)

    if not tokens_list:
        raise ValueError("prompt_ids không được rỗng!")

    generated = list(tokens_list)
    batch_size = 1
    n_heads = model.layers[0].attn.n_heads
    head_dim = model.layers[0].attn.head_dim
    dtype = model.embedding.weight.dtype

    if use_kv_cache:
        # CRITICAL: Khởi tạo danh sách KVCache tiền cấp phát cho từng tầng của Transformer.
        # Dung lượng max_len = độ dài prompt + max_new_tokens để tránh re-allocation.
        total_len = len(tokens_list) + max_new_tokens
        kv_caches = [
            KVCache(
                batch_size=batch_size,
                n_heads=n_heads,
                head_dim=head_dim,
                max_len=total_len,
                device=device,
                dtype=dtype,
            )
            for _ in range(len(model.layers))
        ]

        # Pha 1: Prefill toàn bộ chuỗi prompt 1 lần duy nhất
        input_tensor = torch.tensor([tokens_list], dtype=torch.long, device=device)
        logits = model(input_tensor, kv_caches=kv_caches)
        next_id = sample_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
        generated.append(next_id)

        # Pha 2: Loop sinh từng token mới. Mỗi bước chỉ forward đúng 1 token vừa sinh (T=1).
        for _ in range(max_new_tokens - 1):
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(next_tensor, kv_caches=kv_caches)
            next_id = sample_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
            generated.append(next_id)

    else:
        # Baseline không dùng KV Cache: Mỗi step forward lại toàn bộ chuỗi lịch sử từ đầu (O(N^2))
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_id = sample_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
            generated.append(next_id)

    return generated


if __name__ == "__main__":
    # Smoke test: So sánh output greedy giữa KV Cache và Baseline
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

    prompt = [12, 45, 78, 90]
    max_new = 8

    # 1. Sinh có KV Cache (Greedy)
    out_cache = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=max_new,
        temperature=0.0,
        use_kv_cache=True,
    )

    # 2. Sinh không dùng KV Cache (Greedy)
    out_nocache = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=max_new,
        temperature=0.0,
        use_kv_cache=False,
    )

    print(f"Prompt:      {prompt}")
    print(f"With Cache:  {out_cache}")
    print(f"No Cache:    {out_nocache}")

    assert out_cache == out_nocache, "Kết quả sinh của KV Cache không khớp với baseline!"
    assert len(out_cache) == len(prompt) + max_new, "Độ dài sinh không đúng!"
    print("InferenceEngine smoke test passed successfully!")
