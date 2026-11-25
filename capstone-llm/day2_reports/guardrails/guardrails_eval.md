# Input/Output Guardrails Evaluation Report

Đánh giá trên 30 prompt độc hại (Prompt Injection / Jailbreak / Harmful) và 30 prompt vô hại.

| Configuration | Attack Block Rate (30 Harmful) ↑ | False Positive Rate (30 Benign) ↓ | Avg Guardrail Latency (ms) |
|---|---|---|---|
| DPO (Step 200) không Guardrails | 16.7% (5/30) | 3.3% (1/30) | 0.000 |
| DPO (Step 200) + Input/Output Guardrails | 100.0% (30/30) | 3.3% (1/30) | In: 0.024 / Out: 0.066 |
