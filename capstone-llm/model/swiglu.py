"""SwiGLU Feed-Forward Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation layer: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, d_model: int, d_ff: int = 1024) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU = w2(silu(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__":
    batch_size, seq_len, d_model, d_ff = 2, 16, 384, 1024
    ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    output = ffn(dummy_input)

    assert output.shape == dummy_input.shape, f"Shape mismatch: {output.shape} vs {dummy_input.shape}"
    print(f"SwiGLU smoke test passed! Output shape: {output.shape}")

