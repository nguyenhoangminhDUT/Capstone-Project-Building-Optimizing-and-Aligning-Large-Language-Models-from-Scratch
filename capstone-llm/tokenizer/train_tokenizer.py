"""Train a Byte Pair Encoding (BPE) tokenizer using the HuggingFace tokenizers library."""

import os
import sys
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_bpe(
    corpus_path: str = "data/tinystories_full.txt",
    vocab_size: int = 12000,
    save_path: str = "tokenizer/tokenizer.json",
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a Byte Pair Encoding (BPE) tokenizer and save it to disk."""
    # Kiểm tra file corpus
    if not os.path.exists(corpus_path):
        repo_relative = Path(__file__).resolve().parent.parent / corpus_path
        if repo_relative.exists():
            corpus_path = str(repo_relative)
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file corpus tại: '{corpus_path}'. "
                f"Vui lòng đảm bảo file dữ liệu TinyStories đã được tải vào thư mục 'data/'."
            )

    print(f"Bắt đầu huấn luyện BPE Tokenizer từ file: {corpus_path}")
    print(f"Cấu hình: vocab_size={vocab_size}, min_frequency={min_frequency}")

    # Khởi tạo mô hình BPE với unk_token
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    # Đảm bảo thư mục lưu trữ tồn tại trước khi lưu
    output_dir = Path(save_path).parent
    if str(output_dir) and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Huấn luyện tokenizer trên toàn bộ file dữ liệu
    tokenizer.train(files=[corpus_path], trainer=trainer)

    # Lưu tokenizer ra file json
    tokenizer.save(save_path)
    print(f"Đã lưu tokenizer thành công vào: {save_path}")
    return tokenizer


if __name__ == "__main__":
    # Đường dẫn cố định theo yêu cầu
    corpus_path = "data/tinystories_full.txt"
    save_path = "tokenizer/tokenizer.json"

    # Hỗ trợ truyền đường dẫn qua dòng lệnh nếu cần
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]

    tok = train_bpe(
        corpus_path=corpus_path,
        vocab_size=12000,
        save_path=save_path,
        min_frequency=2,
    )

    print(f"Vocab size thực tế: {tok.get_vocab_size()}")

    test_sentence = "Once upon a time, there was a little girl."
    encoded = tok.encode(test_sentence)
    print(f"Test sentence: '{test_sentence}'")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Decoded text: '{tok.decode(encoded.ids)}'")
