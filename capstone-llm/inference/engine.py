"""Autoregressive text generation engine supporting KV Cache, Repetition Penalty, Top-p, and Guardrails."""

import math
import sys
from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.guardrails import check_input, check_output
from inference.kv_cache import KVCache
from model.lora import apply_lora
from model.transformer import Transformer


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: List[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    """Áp dụng repetition penalty để phạt các token đã từng xuất hiện trong ngữ cảnh, chống lặp từ suy biến."""
    if penalty == 1.0 or not generated_tokens:
        return logits

    logits = logits.clone()
    unique_tokens = set(generated_tokens)
    for token_id in unique_tokens:
        if token_id < logits.size(-1):
            if logits[..., token_id] > 0:
                logits[..., token_id] /= penalty
            else:
                logits[..., token_id] *= penalty
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    generated_tokens: Optional[List[int]] = None,
    repetition_penalty: float = 1.2,
) -> int:
    """Sample token kết hợp greedy, temperature, top-k, top-p (nucleus) và repetition penalty."""
    # 1. Repetition penalty
    if generated_tokens and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_tokens, penalty=repetition_penalty)

    # 2. Greedy decoding khi temperature <= 0
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    # 3. Scale logits theo temperature
    logits = logits / temperature

    # 4. Top-k filtering
    if top_k is not None and top_k > 0:
        val_at_k, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < val_at_k[..., [-1]], float("-inf"))

    # 5. Top-p (Nucleus) filtering: chỉ giữ lại tập token có tích lũy xác suất <= top_p
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Loại bỏ các token có tích lũy xác suất vượt quá top_p (luôn giữ lại token đầu tiên)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_ids: Union[List[int], torch.Tensor],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.2,
    use_kv_cache: bool = True,
    eos_token_id: int = 3,
) -> List[int]:
    """
    Sinh văn bản tự hồi quy (Autoregressive Generation) nâng cấp:
    - Hỗ trợ KVCache tăng tốc suy luận
    - Repetition penalty (1.2) chống lặp từ
    - Top-p (0.9) và Top-k (50) lọc phân phối xác suất
    - Tự động ngắt (break) ngay khi sinh token <|eos|> (ID=3)
    """
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
        # Khởi tạo KVCache tiền cấp phát
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
        next_id = sample_token(
            logits[0, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_tokens=generated,
            repetition_penalty=repetition_penalty,
        )

        # Ngắt nếu gặp token <|eos|>
        if next_id == eos_token_id:
            return generated

        generated.append(next_id)

        # Pha 2: Loop sinh từng token mới với KVCache (T = 1)
        for _ in range(max_new_tokens - 1):
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(next_tensor, kv_caches=kv_caches)
            next_id = sample_token(
                logits[0, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_tokens=generated,
                repetition_penalty=repetition_penalty,
            )

            # Tự động break khi sinh token <|eos|> (ID=3)
            if next_id == eos_token_id:
                break

            generated.append(next_id)

    else:
        # Baseline không dùng KV Cache
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_id = sample_token(
                logits[0, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_tokens=generated,
                repetition_penalty=repetition_penalty,
            )

            if next_id == eos_token_id:
                break

            generated.append(next_id)

    return generated


def is_full_ft_checkpoint(data: object) -> bool:
    """Kiểm tra xem checkpoint có phải là mô hình Full Fine-Tuning hay không."""
    if not isinstance(data, dict):
        return False
    if data.get("mode") == "full_ft":
        return True
    sd = data.get("model_state_dict", data)
    if isinstance(sd, dict):
        has_lora = any("lora_" in k for k in sd.keys())
        has_full_weights = any("layers.0.attn.q_proj.weight" in k or "layers.0.ffn.w1.weight" in k for k in sd.keys())
        if has_full_weights and not has_lora:
            return True
    return False


def load_model_for_inference(
    base_model_path: Optional[str] = None,
    lora_checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    vocab_size: int = 32000,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    d_ff: int = 2048,
    max_seq_len: int = 1024,
    lora_r: int = 32,
    lora_alpha: float = 64.0,
) -> Transformer:
    """
    Nạp mô hình Transformer 100M hoàn chỉnh phục vụ suy luận / demo:
    1. Khởi tạo Transformer gốc theo kiến trúc 100M (GPT-2 Small scale).
    2. Nếu checkpoint là Full FT (mode == "full_ft" hoặc chứa toàn bộ trọng số gốc mà không có "lora_"):
       - Nạp trực tiếp vào Transformer mà KHÔNG gọi apply_lora.
    3. Ngược lại (LoRA / Base):
       - Nạp base weights từ base_model_path (nếu có).
       - Gọi apply_lora với rank r=32, alpha=64.
       - Nạp trọng số LoRA từ lora_checkpoint_path (chỉ lấy key có chứa 'lora_' hoặc 'embedding.weight').
    4. Chuyển sang model.eval() và trả về model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device)

    # 1. Khởi tạo Transformer gốc 100M
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )

    def resolve_path(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        path = Path(p)
        if path.exists():
            return path
        if (REPO_ROOT / path).exists():
            return REPO_ROOT / path
        return None

    base_p = resolve_path(base_model_path)
    lora_p = resolve_path(lora_checkpoint_path)

    base_raw = None
    if base_p and base_p.exists():
        try:
            base_raw = torch.load(base_p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Engine] Cảnh báo lỗi nạp base checkpoint ({e})")

    lora_raw = None
    if lora_p and lora_p.exists():
        try:
            lora_raw = torch.load(lora_p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Engine] Cảnh báo lỗi nạp lora checkpoint ({e})")

    # Kiểm tra Full Fine-Tuning (có thể truyền qua lora_checkpoint_path hoặc base_model_path)
    is_full_ft = False
    full_ft_data = None

    if lora_raw is not None and is_full_ft_checkpoint(lora_raw):
        is_full_ft = True
        full_ft_data = lora_raw
    elif base_raw is not None and is_full_ft_checkpoint(base_raw) and lora_checkpoint_path is None:
        is_full_ft = True
        full_ft_data = base_raw

    if is_full_ft:
        # Chế độ Full Fine-Tuning: nạp trực tiếp toàn bộ trọng số mà KHÔNG gọi apply_lora
        sd = full_ft_data.get("model_state_dict", full_ft_data)
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        print("[Engine] Nạp Full Fine-Tuning model weights thành công (bỏ qua LoRA)!")
    else:
        # Nạp base model weights nếu có
        if base_raw is not None:
            sd = base_raw.get("model_state_dict", base_raw)
            clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(clean_sd)
            print(f"[Engine] Nạp base model weights thành công từ {base_p}!")
        elif base_model_path:
            print(f"[Engine] Cảnh báo: Không tìm thấy base checkpoint tại {base_model_path}")

        # Gắn LoRA
        apply_lora(model, r=lora_r, alpha=lora_alpha)

        # Nạp trọng số LoRA nếu có
        if lora_raw is not None:
            sd = lora_raw.get("lora_state_dict", lora_raw.get("model_state_dict", lora_raw))
            lora_weights = {
                k.replace("module.", ""): v for k, v in sd.items()
                if "lora_" in k or "embedding.weight" in k
            }
            model.load_state_dict(lora_weights, strict=False)
            print(f"[Engine] Nạp thành công {len(lora_weights)} tensors LoRA adapter & Embedding!")
        elif lora_checkpoint_path:
            print(f"[Engine] Cảnh báo: Không tìm thấy LoRA checkpoint tại {lora_checkpoint_path}")

    model.to(device)
    model.eval()
    return model


def chat(
    user_prompt: str,
    model: Transformer,
    tokenizer: object,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.2,
    use_kv_cache: bool = True,
    enable_guardrails: bool = True,
) -> str:
    """
    Hàm trò chuyện tương tác với LLM 100M:
    1. Kiểm duyệt đầu vào qua check_input (nếu enable_guardrails=True).
    2. Format prompt: '### Instruction:\\n{user_prompt}\\n\\n### Response:\\n'.
    3. Sinh token với KV Cache, repetition_penalty=1.2, top_p=0.9, tự động dừng khi gặp <|eos|>.
    4. Kiểm duyệt đầu ra qua check_output (PII / Toxic) (nếu enable_guardrails=True).
    5. Trả về kết quả hoàn chỉnh, mượt mà, không lặp từ.
    """
    # 1. Kiểm tra an toàn đầu vào
    if enable_guardrails and not check_input(user_prompt):
        return "Xin lỗi, yêu cầu của bạn đã bị từ chối do vi phạm chính sách an toàn (phát hiện prompt injection hoặc nội dung độc hại)."

    # 2. Format prompt theo chuẩn template
    formatted_prompt = f"### Instruction:\n{user_prompt.strip()}\n\n### Response:\n"

    prompt_ids = tokenizer.encode(formatted_prompt).ids if hasattr(tokenizer, "encode") else [0]
    if not prompt_ids:
        prompt_ids = [0]

    # 3. Sinh phản hồi với KV Cache, repetition_penalty và top_p
    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_kv_cache=use_kv_cache,
    )

    # 4. Trích xuất text phản hồi mới sinh
    gen_token_ids = output_ids[len(prompt_ids) :]
    raw_response = tokenizer.decode(gen_token_ids).strip() if hasattr(tokenizer, "decode") else ""

    # Cắt ngắn nếu token kết thúc <|eos|> hoặc lặp instruction
    if "<|eos|>" in raw_response:
        raw_response = raw_response.split("<|eos|>")[0].strip()
    if "### Instruction:" in raw_response:
        raw_response = raw_response.split("### Instruction:")[0].strip()

    # 5. Kiểm tra an toàn đầu ra
    if enable_guardrails and not check_output(raw_response):
        return "Nội dung phản hồi đã bị ẩn do vi phạm chính sách an toàn (phát hiện thông tin định danh cá nhân PII hoặc nội dung không an toàn)."

    return raw_response


class LLMInferenceEngine:
    """
    Công cụ suy luận cấp cao (High-Level Inference Engine):
    - Đóng gói Transformer, LoRA Adapter, Tokenizer, và KVCache
    - Cung cấp phương thức chat(), generate() và compute_perplexity()
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        lora_checkpoint_path: Optional[str] = None,
        tokenizer: Optional[Union[str, object]] = None,
        device: Optional[Union[str, torch.device]] = None,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        lora_r: int = 32,
        lora_alpha: float = 64.0,
    ) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

        if isinstance(tokenizer, (str, Path)):
            tok_path = Path(tokenizer)
            if not tok_path.exists() and (REPO_ROOT / tok_path).exists():
                tok_path = REPO_ROOT / tok_path
            self.tokenizer = Tokenizer.from_file(str(tok_path))
        elif tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            cand = [
                REPO_ROOT / "tokenizer" / "tokenizer.json",
                Path("tokenizer/tokenizer.json"),
            ]
            for c in cand:
                if c.exists():
                    self.tokenizer = Tokenizer.from_file(str(c))
                    break

        actual_vocab = self.tokenizer.get_vocab_size() if self.tokenizer is not None else vocab_size

        self.model = load_model_for_inference(
            base_model_path=base_model_path,
            lora_checkpoint_path=lora_checkpoint_path,
            device=self.device,
            vocab_size=actual_vocab,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

    def chat(
        self,
        user_prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        use_kv_cache: bool = True,
        enable_guardrails: bool = True,
    ) -> str:
        return chat(
            user_prompt=user_prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_kv_cache=use_kv_cache,
            enable_guardrails=enable_guardrails,
        )

    def generate(
        self,
        prompt_ids: Union[List[int], torch.Tensor],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        use_kv_cache: bool = True,
        eos_token_id: int = 3,
    ) -> List[int]:
        return generate(
            model=self.model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_kv_cache=use_kv_cache,
            eos_token_id=eos_token_id,
        )

    @torch.no_grad()
    def compute_perplexity(self, texts: List[str], max_seq_len: int = 1024) -> float:
        """Tính toán validation perplexity trên danh sách chuỗi văn bản."""
        if not texts or self.tokenizer is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            encoded = self.tokenizer.encode(text).ids
            if len(encoded) < 2:
                continue
            if len(encoded) > max_seq_len:
                encoded = encoded[:max_seq_len]

            inp = torch.tensor([encoded[:-1]], dtype=torch.long, device=self.device)
            target = torch.tensor([encoded[1:]], dtype=torch.long, device=self.device)

            logits = self.model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="sum")

            total_loss += loss.item()
            total_tokens += target.numel()

        if total_tokens == 0:
            return float("nan")

        avg_loss = total_loss / total_tokens
        return math.exp(min(avg_loss, 20.0))


if __name__ == "__main__":
    print("=" * 65)
    print("CHẠY SMOKE TEST ENGINE 100M (TOP-P, REPETITION PENALTY, EOS)")
    print("=" * 65)

    torch.manual_seed(42)
    device = torch.device("cpu")

    toy_model = Transformer(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=128,
    ).to(device)

    prompt = [12, 45, 78, 90]
    max_new = 8

    # 1. Test greedy equivalence (KV Cache vs No Cache)
    out_cache = generate(toy_model, prompt, max_new_tokens=max_new, temperature=0.0, use_kv_cache=True)
    out_nocache = generate(toy_model, prompt, max_new_tokens=max_new, temperature=0.0, use_kv_cache=False)
    assert out_cache == out_nocache, "KV Cache greedy output mismatch!"
    print("Test 1: KV Cache vs Baseline generate identical (PASS)")

    # 2. Test repetition penalty & top_p
    out_penalized = generate(
        toy_model,
        prompt,
        max_new_tokens=12,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        use_kv_cache=True,
    )
    assert len(out_penalized) <= len(prompt) + 12
    print(f"Test 2: Repetition penalty & Top-p output length: {len(out_penalized)} (PASS)")

    # 3. Test load_model_for_inference
    loaded_model = load_model_for_inference(
        base_model_path=None,
        lora_checkpoint_path=None,
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        lora_r=4,
        lora_alpha=8.0,
        device="cpu",
    )
    assert hasattr(loaded_model.layers[0].attn.q_proj, "lora_A")
    print("Test 3: load_model_for_inference passed successfully!")

    print("\n" + "=" * 65)
    print("TẤT CẢ SMOKE TEST CHO ENGINE NÂNG CẤP ĐÃ HOÀN THÀNH!")
    print("=" * 65)
