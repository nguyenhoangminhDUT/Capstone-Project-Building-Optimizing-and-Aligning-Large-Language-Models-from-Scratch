"""Direct Preference Optimization (DPO) Loss."""

import math
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Direct Preference Optimization (DPO) loss and implicit rewards."""
    # NOTE: Reference model phải chạy trong context torch.no_grad() và có toàn bộ
    # tham số đặt requires_grad=False để tránh rò rỉ gradient và tiết kiệm bộ nhớ GPU.
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards, rejected_rewards


if __name__ == "__main__":
    # Smoke test: khi policy == ref cho cả 2, loss phải = log(2) (diff < 1e-6)
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
    print(f"DPO Loss: {loss.item():.6f}, Expected: {expected_loss:.6f}, Diff: {diff:.8e}")
    assert diff < 1e-6, f"DPO smoke test failed, diff: {diff}"
    print("DPO loss smoke test passed!")

