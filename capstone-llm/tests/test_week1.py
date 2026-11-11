"""Unit tests for Week 1 LLM core components using pytest."""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Thêm đường dẫn thư mục gốc và capstone-llm vào sys.path để import linh hoạt
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CAPSTONE_DIR = REPO_ROOT / "capstone-llm"
if CAPSTONE_DIR.exists() and str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))

try:
    from model.rmsnorm import RMSNorm
    from model.rope import RoPE, apply_rope
    from model.attention import CausalSelfAttention
    from model.swiglu import SwiGLU
    from model.transformer import Transformer
    from model.lora import LoRALinear
    from training.dpo_loss import dpo_loss
except ImportError:
    from capstone_llm.model.rmsnorm import RMSNorm
    from capstone_llm.model.rope import RoPE, apply_rope
    from capstone_llm.model.attention import CausalSelfAttention
    from capstone_llm.model.swiglu import SwiGLU
    from capstone_llm.model.transformer import Transformer
    from capstone_llm.model.lora import LoRALinear
    from capstone_llm.training.dpo_loss import dpo_loss


def test_rmsnorm_identity() -> None:
    """Test that RMSNorm with unit weights normalizes RMS to approximately 1."""
    d_model = 64
    norm = RMSNorm(d_model=d_model, eps=1e-5)
    x = torch.randn(2, 16, d_model)
    out = norm(x)

    # Kiểm tra RMS của từng vector sau chuẩn hóa xấp xỉ 1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    # So sánh trực tiếp với công thức chuẩn hóa
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    assert torch.allclose(out, expected, atol=1e-5)


def test_rope_identity_at_pos0() -> None:
    """Test that RoPE applied at position 0 leaves q unchanged."""
    head_dim = 48
    rope = RoPE(head_dim=head_dim, max_len=512)
    q = torch.randn(2, 4, 1, head_dim)
    k = torch.randn(2, 4, 1, head_dim)

    cos, sin = rope(q)  # seq_len = 1 (tại vị trí 0)
    q_rot, _ = apply_rope(q, k, cos, sin)

    diff = (q_rot - q).abs().max().item()
    assert diff < 1e-6, f"q changed at position 0: max diff = {diff}"


def test_rope_relative_invariance() -> None:
    """Test that <RoPE(q, i), RoPE(k, j)> depends only on the relative offset (i - j)."""
    head_dim = 48
    rope = RoPE(head_dim=head_dim, max_len=512)
    dummy = torch.empty(16, head_dim)
    cos_table, sin_table = rope(dummy)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Cặp 1: i=5, j=3 (offset = 2)
    q5, _ = apply_rope(q, q, cos_table[5:6], sin_table[5:6])
    _, k3 = apply_rope(k, k, cos_table[3:4], sin_table[3:4])
    dot_53 = (q5 * k3).sum().item()

    # Cặp 2: i=6, j=4 (offset = 2)
    q6, _ = apply_rope(q, q, cos_table[6:7], sin_table[6:7])
    _, k4 = apply_rope(k, k, cos_table[4:5], sin_table[4:5])
    dot_64 = (q6 * k4).sum().item()

    diff = abs(dot_53 - dot_64)
    assert diff < 1e-5, f"Relative shift invariance failed: diff = {diff}"


def test_causal_mask_no_future_leak() -> None:
    """Test that future tokens do not affect past token logits in causal attention."""
    torch.manual_seed(42)
    model = Transformer(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
    )
    model.eval()

    seq_len = 16
    t = 8
    input1 = torch.randint(0, 1000, (1, seq_len))
    input2 = input1.clone()
    # Thay đổi toàn bộ token từ vị trí t trở đi
    input2[:, t:] = (input2[:, t:] + 137) % 1000

    with torch.no_grad():
        logits1 = model(input1)
        logits2 = model(input2)

    # Logits tại vị trí t-1 không được bị ảnh hưởng bởi token tại [t:]
    diff_at_t_minus_1 = (logits1[:, t - 1, :] - logits2[:, t - 1, :]).abs().max().item()
    assert diff_at_t_minus_1 < 1e-5, f"Future leak detected at t-1: diff = {diff_at_t_minus_1}"

    # Toàn bộ logits từ 0 đến t-1 cũng phải giống hệt nhau
    diff_prefix = (logits1[:, :t, :] - logits2[:, :t, :]).abs().max().item()
    assert diff_prefix < 1e-5, f"Future leak detected in prefix: diff = {diff_prefix}"


def test_attention_output_shape() -> None:
    """Test that CausalSelfAttention preserves input tensor dimensions."""
    d_model, n_heads, seq_len = 64, 4, 16
    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_len=64)
    rope = RoPE(head_dim=d_model // n_heads, max_len=64)
    x = torch.randn(2, seq_len, d_model)
    out = attn(x, rope=rope)
    assert out.shape == (2, seq_len, d_model)


def test_swiglu_shape() -> None:
    """Test that SwiGLU preserves input tensor dimensions."""
    d_model, d_ff, seq_len = 64, 128, 16
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
    x = torch.randn(2, seq_len, d_model)
    out = swiglu(x)
    assert out.shape == (2, seq_len, d_model)


def test_lora_identity_init() -> None:
    """Test that newly initialized LoRALinear outputs exactly equal base layer outputs."""
    base = nn.Linear(64, 64)
    lora = LoRALinear(base=base, r=4, alpha=8.0)
    x = torch.randn(2, 8, 64)
    out_lora = lora(x)
    out_base = base(x)
    diff = (out_lora - out_base).abs().max().item()
    assert diff < 1e-7, f"LoRA not initialized as identity: diff = {diff}"


def test_dpo_loss_zero_margin() -> None:
    """Test that DPO loss equals log(2) when policy matches reference model."""
    batch_size = 4
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)
    policy_chosen = ref_chosen.clone()
    policy_rejected = ref_rejected.clone()

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
        beta=0.1,
    )

    expected_loss = math.log(2.0)
    diff = abs(loss.item() - expected_loss)
    assert diff < 1e-6, f"DPO zero-margin loss mismatch: diff = {diff}"

