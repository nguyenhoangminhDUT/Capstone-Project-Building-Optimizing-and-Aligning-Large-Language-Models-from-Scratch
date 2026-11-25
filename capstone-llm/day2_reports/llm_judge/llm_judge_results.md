# LLM-as-a-Judge Pairwise Evaluation Report

- Phương pháp: Pairwise Comparison kèm đảo vị trí A/B (Position-Bias Mitigation) trên 30 prompt đa lĩnh vực.
- Judge Provider: `auto`

| Pair (Model A vs Model B) | Win A (%) | Tie (%) | Win B (%) | Avg Score A (1-10) | Avg Score B (1-10) |
|---|---|---|---|---|---|
| **SFT (LoRA+Embed) vs Base** | 23.3% | 76.7% | 0.0% | 2.28 | 1.43 |
| **SFT (Full FT) vs SFT (LoRA+Embed)** | 10.0% | 83.3% | 6.7% | 2.82 | 2.72 |
| **DPO (Step 200) vs SFT (LoRA+Embed)** | 16.7% | 70.0% | 13.3% | 2.60 | 1.33 |
