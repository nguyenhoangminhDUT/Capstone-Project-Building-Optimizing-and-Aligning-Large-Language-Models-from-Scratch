"""Train a Byte Pair Encoding (BPE) tokenizer using the HuggingFace tokenizers library."""

import os
import sys
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_bpe(
    corpus_path: str,
    vocab_size: int = 12000,
    save_path: str = "tokenizer/tokenizer.json",
) -> Tokenizer:
    """Train a Byte Pair Encoding (BPE) tokenizer and save it to disk."""
    # CRITICAL: BPE model must specify unk_token matching the trainer's special tokens
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    # CRITICAL: Ensure output directory exists before saving tokenizer
    output_dir = Path(save_path).parent
    if str(output_dir) and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


if __name__ == "__main__":
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "data/tinystories_sample.txt"
    save_path = "tokenizer/tokenizer.json"

    # Tạo file mẫu nếu chưa có dữ liệu để chạy smoke test
    if not os.path.exists(corpus_path):
        os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(
                "Once upon a time, there was a little girl named Lily. "
                "She loved to play in the park with her dog. "
                "Once upon a time, there was a little boy named Tim."
            )

    tok = train_bpe(corpus_path=corpus_path, vocab_size=12000, save_path=save_path)
    print(f"Vocab size thực tế: {tok.get_vocab_size()}")

    test_sentence = "Once upon a time, there was a little girl."
    encoded = tok.encode(test_sentence)
    print(f"IDs: {encoded.ids}")
    print(f"Decoded: {tok.decode(encoded.ids)}")

