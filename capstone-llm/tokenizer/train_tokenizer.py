"""Train a Byte Pair Encoding (BPE) tokenizer scaled to 32,000 vocab with digit pre-tokenization."""

import os
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Digits, Sequence, Whitespace
from tokenizers.trainers import BpeTrainer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_dataset_stream(
    corpus_path: str = "data/tinystories_full.txt",
    tinystories_limit: int = 100_000,
    alpaca_limit: int = 20_000,
    gsm8k_limit: int = 5_000,
) -> Iterator[str]:
    """
    Tập hợp và stream luồng dữ liệu văn bản từ 3 nguồn:
    1. 100,000 dòng TinyStories (ưu tiên đọc file cục bộ, nếu không thì stream từ HuggingFace).
    2. 20,000 dòng từ yahma/alpaca-cleaned (instruction/response).
    3. 5,000 dòng từ gsm8k (toán học suy luận logic).
    """
    total_yielded = 0

    # 1. TinyStories (100,000 dòng)
    ts_file = Path(corpus_path)
    if not ts_file.exists() and (REPO_ROOT / corpus_path).exists():
        ts_file = REPO_ROOT / corpus_path

    if ts_file.exists():
        print(f"[Dataset] Đang đọc tối đa {tinystories_limit:,} dòng từ file TinyStories cục bộ: {ts_file}...")
        with open(ts_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= tinystories_limit:
                    break
                line = line.strip()
                if line:
                    yield line
                    total_yielded += 1
    elif load_dataset is not None:
        print(f"[Dataset] Đang stream {tinystories_limit:,} mẫu từ HuggingFace 'roneneldan/TinyStories'...")
        try:
            ds_ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            ts_count = 0
            for item in ds_ts:
                text = item.get("text", "").strip()
                if text:
                    yield text
                    ts_count += 1
                    total_yielded += 1
                    if ts_count >= tinystories_limit:
                        break
        except Exception as e:
            print(f"[Dataset] Cảnh báo khi tải TinyStories online: {e}")
    else:
        print("[Dataset] Bỏ qua TinyStories online do chưa có thư viện datasets hoặc file dữ liệu.")

    # 2. Alpaca Cleaned (20,000 dòng)
    if load_dataset is not None and alpaca_limit > 0:
        print(f"[Dataset] Đang tải {alpaca_limit:,} dòng từ 'yahma/alpaca-cleaned'...")
        try:
            ds_alpaca = load_dataset("yahma/alpaca-cleaned", split=f"train[:{alpaca_limit}]")
            for item in ds_alpaca:
                inst = str(item.get("instruction", "")).strip()
                inp = str(item.get("input", "")).strip()
                out = str(item.get("output", "")).strip()
                combined = f"{inst}\n{inp}\n{out}".strip() if inp else f"{inst}\n{out}".strip()
                if combined:
                    yield combined
                    total_yielded += 1
        except Exception as e:
            print(f"[Dataset] Cảnh báo khi nạp yahma/alpaca-cleaned: {e}")

    # 3. GSM8K (5,000 dòng toán học)
    if load_dataset is not None and gsm8k_limit > 0:
        print(f"[Dataset] Đang tải {gsm8k_limit:,} dòng từ 'gsm8k'...")
        try:
            try:
                ds_gsm = load_dataset("openai/gsm8k", "main", split=f"train[:{gsm8k_limit}]")
            except Exception:
                ds_gsm = load_dataset("gsm8k", "main", split=f"train[:{gsm8k_limit}]")

            for item in ds_gsm:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a:
                    yield f"Question: {q}\nAnswer: {a}"
                    total_yielded += 1
        except Exception as e:
            print(f"[Dataset] Cảnh báo khi nạp gsm8k: {e}")

    # Dự phòng nếu không có mạng và không có file lớn (chạy smoke test cục bộ)
    if total_yielded == 0:
        print("[Dataset] Sử dụng dữ liệu dự phòng cục bộ cho tokenizer...")
        fallback_texts = [
            "Once upon a time, there was a little girl named Lily.",
            "She loved to play in the park with her dog.",
            "What is 47 + 28? The answer is 75.",
            "Calculate 100 - 35. The result is 65.",
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            "### Instruction:\nSolve the math problem.\n\n### Response:\n42 is the answer.<|eos|>",
        ]
        for t in fallback_texts:
            yield t


def train_bpe(
    corpus_path: str = "data/tinystories_full.txt",
    vocab_size: int = 32000,
    save_path: str = "tokenizer/tokenizer.json",
    min_frequency: int = 2,
    tinystories_limit: int = 100_000,
    alpaca_limit: int = 20_000,
    gsm8k_limit: int = 5_000,
) -> Tokenizer:
    """
    Huấn luyện BPE Tokenizer nâng cấp:
    - vocab_size = 32,000
    - Pre-tokenizer Sequence: Whitespace() + Digits(individual_digits=True) để tách số
    - Special tokens: ['<|unk|>', '<|pad|>', '<|bos|>', '<|eos|>']
    """
    print("=" * 65)
    print("BẮT ĐẦU HUẤN LUYỆN TOKENIZER 32,000 VOCAB (ULTIMATE SCALE-UP)")
    print("=" * 65)
    print(f"  - Vocab size target : {vocab_size:,}")
    print(f"  - Min frequency     : {min_frequency}")
    print(f"  - Save path         : {save_path}")

    # Khởi tạo mô hình BPE
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # CRITICAL: Tách số từng chữ số đơn lẻ kết hợp Whitespace để tối ưu hóa suy luận toán học
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])

    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    # Đảm bảo thư mục lưu trữ tồn tại
    out_dir = Path(save_path).parent
    if str(out_dir) and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Nạp luồng dữ liệu tổng hợp
    data_stream = get_dataset_stream(
        corpus_path=corpus_path,
        tinystories_limit=tinystories_limit,
        alpaca_limit=alpaca_limit,
        gsm8k_limit=gsm8k_limit,
    )

    print("\nĐang tiến hành huấn luyện từ luồng dữ liệu kết hợp...")
    tokenizer.train_from_iterator(data_stream, trainer=trainer)

    # Lưu tokenizer ra file json
    tokenizer.save(save_path)
    print(f"\nĐã lưu tokenizer thành công vào: {save_path}")
    return tokenizer


if __name__ == "__main__":
    corpus_path = "data/tinystories_full.txt"
    save_path = "tokenizer/tokenizer.json"

    # Cho phép truyền file hoặc đường dẫn qua sys.argv
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]

    tok = train_bpe(
        corpus_path=corpus_path,
        vocab_size=32000,
        save_path=save_path,
        min_frequency=2,
    )

    actual_vocab_size = tok.get_vocab_size()
    print(f"\nVocab size thực tế: {actual_vocab_size:,}")

    test_sentence = "Once upon a time, there was a little girl."
    encoded = tok.encode(test_sentence)
    print(f"Câu kiểm thử: '{test_sentence}'")
    print(f"Token IDs   : {encoded.ids}")
    print(f"Decoded text: '{tok.decode(encoded.ids)}'")

    # Kiểm tra khả năng tách số của Digits pre-tokenizer
    math_test = "47 + 28 = 75"
    encoded_math = tok.encode(math_test)
    print(f"\nKiểm thử toán số học: '{math_test}'")
    print(f"Math Token IDs       : {encoded_math.ids}")
    print(f"Math Decoded         : '{tok.decode(encoded_math.ids)}'")
