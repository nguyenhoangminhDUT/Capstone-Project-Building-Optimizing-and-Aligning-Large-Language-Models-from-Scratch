"""Dataset and tokenization utilities for LLM pre-training."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TokenDataset(Dataset):
    """Dataset that slices a 1D token array into contiguous chunks of (x, y) with a 1-token shift."""

    def __init__(
        self,
        tokens: Union[list[int], np.ndarray],
        block_size: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        if isinstance(tokens, list):
            self.tokens = np.array(tokens, dtype=np.int64)
        elif isinstance(tokens, np.ndarray):
            self.tokens = tokens.astype(np.int64) if tokens.dtype != np.int64 else tokens
        else:
            raise TypeError(f"tokens phải là list[int] hoặc np.ndarray, nhận được: {type(tokens)}")

        # CRITICAL: Cần (block_size + 1) tokens để tạo 1 cặp (x, y) có độ dài block_size
        self.num_samples = max(0, (len(self.tokens) - 1) // block_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} nằm ngoài khoảng [0, {self.num_samples})")

        start = idx * self.block_size
        # x lấy block_size token đầu, y lấy block_size token kế tiếp (dịch phải 1 token)
        x = torch.from_numpy(self.tokens[start : start + self.block_size]).long()
        y = torch.from_numpy(self.tokens[start + 1 : start + self.block_size + 1]).long()
        return x, y


def load_and_tokenize(
    corpus_path: str,
    tokenizer_path: str,
    max_tokens: Optional[int] = None,
) -> np.ndarray:
    """Load text corpus, tokenize non-empty lines, and concatenate into a single token ID array."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    all_token_ids: list[int] = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue
            encoding = tokenizer.encode(clean_line)
            all_token_ids.extend(encoding.ids)
            if max_tokens is not None and len(all_token_ids) >= max_tokens:
                all_token_ids = all_token_ids[:max_tokens]
                break

    return np.array(all_token_ids, dtype=np.int64)


if __name__ == "__main__":
    # Smoke test: Kiểm tra TokenDataset
    dummy_tokens = np.arange(105)
    block_size = 10
    dataset = TokenDataset(tokens=dummy_tokens, block_size=block_size)

    print(f"Num samples: {len(dataset)} (kỳ vọng: 10)")
    assert len(dataset) == 10, "Số lượng mẫu không đúng!"

    x0, y0 = dataset[0]
    assert x0.shape == (block_size,)
    assert y0.shape == (block_size,)
    assert torch.equal(x0, torch.arange(0, 10))
    assert torch.equal(y0, torch.arange(1, 11))
    assert torch.equal(x0[1:], y0[:-1]), "y không phải là chuỗi dịch phải 1 token của x!"
    print("TokenDataset smoke test passed successfully!")
