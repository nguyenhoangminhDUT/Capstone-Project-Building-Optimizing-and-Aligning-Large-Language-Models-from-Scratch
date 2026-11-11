"""Pre-training dataset with fixed-length chunking for autoregressive language modeling."""

from pathlib import Path
from typing import Sequence, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PretrainDataset(Dataset):
    """Dataset that slices a 1D token array into contiguous chunks of (seq_len + 1) for next-token prediction."""

    def __init__(
        self,
        tokens: Union[torch.Tensor, np.ndarray, Sequence[int], str, Path],
        seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # Hỗ trợ nạp từ file binary (.npy) hoặc chuỗi token có sẵn
        if isinstance(tokens, (str, Path)):
            token_path = Path(tokens)
            if token_path.suffix == ".npy":
                # Dùng mmap để tiết kiệm RAM khi đọc dataset lớn
                self.tokens = np.load(token_path, mmap_mode="r")
            else:
                raise ValueError(f"Định dạng file không hỗ trợ: {token_path}")
        elif isinstance(tokens, torch.Tensor):
            self.tokens = tokens.detach().cpu().numpy()
        elif isinstance(tokens, np.ndarray):
            self.tokens = tokens
        else:
            self.tokens = np.array(tokens, dtype=np.int64)

        # CRITICAL: Mỗi mẫu cần (seq_len + 1) tokens để x = chunk[:-1] và y = chunk[1:].
        # Số lượng mẫu được tính dựa trên tổng số token có thể chia đều cho seq_len.
        total_tokens = len(self.tokens)
        self.num_samples = max(0, (total_tokens - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} nằm ngoài dải hợp lệ [0, {self.num_samples})")

        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]

        # CRITICAL: x lấy seq_len token đầu, y lấy seq_len token tiếp theo (dịch phải 1 vị trí).
        # Cách này đảm bảo nhãn y là next-token chính xác tại mọi bước t mà không cần padding.
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


if __name__ == "__main__":
    # Giả lập 2500 token IDs ngẫu nhiên trong khoảng vocab_size [0, 12000)
    seq_len = 512
    total_dummy_tokens = 2500
    dummy_tokens = np.random.randint(0, 12000, size=total_dummy_tokens, dtype=np.int64)

    dataset = PretrainDataset(tokens=dummy_tokens, seq_len=seq_len)
    expected_samples = (total_dummy_tokens - 1) // seq_len
    print(f"Tổng số mẫu trong dataset: {len(dataset)} (Kỳ vọng: {expected_samples})")
    assert len(dataset) == expected_samples, "Sai số lượng mẫu!"

    # Kiểm tra kích thước và tính chất dịch phải 1 token
    x0, y0 = dataset[0]
    assert x0.shape == (seq_len,), f"Sai shape x: {x0.shape}"
    assert y0.shape == (seq_len,), f"Sai shape y: {y0.shape}"
    # Token thứ 1 của x phải bằng token thứ 0 của y
    assert torch.equal(x0[1:], y0[:-1]), "Cặp (x, y) không thỏa mãn dịch 1 token!"

    # Kiểm tra chạy cùng DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, seq_len)
    assert batch_y.shape == (2, seq_len)
    print("PretrainDataset smoke test passed!")

