"""Dataset and tokenization utilities for LLM pre-training using memory-mapped binary files."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TokenDataset(Dataset):
    """Dataset that slices a memory-mapped 1D token array into contiguous chunks of (x, y) with a 1-token shift."""

    def __init__(
        self,
        tokens: Union[list[int], np.ndarray, np.memmap, str, Path],
        block_size: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        # Hỗ trợ nạp trực tiếp đường dẫn file .bin qua memmap (0% RAM overhead)
        if isinstance(tokens, (str, Path)):
            bin_path = Path(tokens)
            if not bin_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file nhị phân: {bin_path}")
            # CRITICAL: nanoGPT pattern — nạp uint16 memmap ở chế độ chỉ đọc 'r' để không tốn RAM
            self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        elif isinstance(tokens, np.memmap):
            self.tokens = tokens
        elif isinstance(tokens, np.ndarray):
            self.tokens = tokens
        elif isinstance(tokens, list):
            self.tokens = np.array(tokens, dtype=np.uint16 if max(tokens or [0]) < 65535 else np.int64)
        else:
            raise TypeError(f"tokens không hợp lệ: {type(tokens)}")

        # CRITICAL: Cần (block_size + 1) tokens để tạo 1 cặp (x, y) có độ dài block_size
        total_tokens = len(self.tokens)
        self.num_samples = max(0, (total_tokens - 1) // block_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} nằm ngoài khoảng [0, {self.num_samples})")

        start = idx * self.block_size
        # Đọc 1 lát cắt (block_size + 1) trực tiếp từ đĩa thông qua memmap
        chunk = self.tokens[start : start + self.block_size + 1]

        # CRITICAL: Ép kiểu sang int64 (torch.long) vì nn.Embedding trong PyTorch yêu cầu index kiểu int64
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def load_and_tokenize(
    corpus_path: str,
    tokenizer_path: str,
    bin_path: Optional[str] = None,
    batch_size: int = 1000,
) -> np.memmap:
    """Tokenize raw text corpus into a memory-mapped binary file (uint16) with nanoGPT-style caching."""
    corpus_file = Path(corpus_path)
    if bin_path is None:
        bin_file = corpus_file.with_suffix(".bin")
    else:
        bin_file = Path(bin_path)

    # 1. Nếu file .bin đã tồn tại, load trực tiếp không cần tokenize lại
    if bin_file.exists() and bin_file.stat().st_size > 0:
        print(f"[Dataset] Đã tìm thấy cache nhị phân {bin_file}. Nạp trực tiếp qua memmap...")
        return np.memmap(bin_file, dtype=np.uint16, mode="r")

    # 2. Nếu chưa có, tokenize toàn bộ file text theo từng batch và ghi trực tiếp ra đĩa (tránh tràn RAM)
    print(f"[Dataset] Chưa có cache. Đang tokenize {corpus_file} -> {bin_file}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    bin_file.parent.mkdir(parents=True, exist_ok=True)
    temp_bin = bin_file.with_suffix(".tmp")

    total_tokens = 0
    with open(corpus_file, "r", encoding="utf-8") as f_in, open(temp_bin, "wb") as f_out:
        batch_lines = []
        for line in f_in:
            line_str = line.strip()
            if line_str:
                batch_lines.append(line_str)

            # Tokenize theo batch dùng Rust backend đa luồng của tokenizers
            if len(batch_lines) >= batch_size:
                encodings = tokenizer.encode_batch(batch_lines)
                batch_tokens = []
                for enc in encodings:
                    batch_tokens.extend(enc.ids)

                if batch_tokens:
                    # CRITICAL: Ghi ra file dưới dạng uint16 (2 bytes/token, tiết kiệm 75% bộ nhớ so với int64)
                    np.array(batch_tokens, dtype=np.uint16).tofile(f_out)
                    total_tokens += len(batch_tokens)
                batch_lines.clear()

        # Xử lý các dòng còn sót lại ở cuối file
        if batch_lines:
            encodings = tokenizer.encode_batch(batch_lines)
            batch_tokens = []
            for enc in encodings:
                batch_tokens.extend(enc.ids)
            if batch_tokens:
                np.array(batch_tokens, dtype=np.uint16).tofile(f_out)
                total_tokens += len(batch_tokens)

    # Đổi tên file tạm thành file chính thức sau khi ghi thành công hoàn tất
    temp_bin.rename(bin_file)
    size_mb = bin_file.stat().st_size / (1024**2)
    print(f"[Dataset] Tokenize thành công! Tổng số: {total_tokens:,} tokens ({size_mb:.2f} MB).")

    return np.memmap(bin_file, dtype=np.uint16, mode="r")


if __name__ == "__main__":
    # Smoke test: Kiểm tra TokenDataset và caching memmap
    dummy_tokens = np.arange(105, dtype=np.uint16)
    block_size = 10
    dataset = TokenDataset(tokens=dummy_tokens, block_size=block_size)

    print(f"Num samples: {len(dataset)} (kỳ vọng: 10)")
    assert len(dataset) == 10, "Số lượng mẫu không đúng!"

    x0, y0 = dataset[0]
    assert x0.shape == (block_size,)
    assert y0.shape == (block_size,)
    assert x0.dtype == torch.int64, "x phải có dtype torch.int64!"
    assert y0.dtype == torch.int64, "y phải có dtype torch.int64!"
    assert torch.equal(x0, torch.arange(0, 10))
    assert torch.equal(y0, torch.arange(1, 11))
    assert torch.equal(x0[1:], y0[:-1]), "y không phải là chuỗi dịch phải 1 token của x!"
    print("TokenDataset memmap smoke test passed successfully!")
