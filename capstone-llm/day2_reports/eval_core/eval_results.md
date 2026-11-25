# Quantitative Evaluation Report - Capstone LLM (100M Scale)

Đánh giá thực nghiệm so sánh 3 phiên bản mô hình trên 3 trụ cột: Ngôn ngữ (PPL), Suy luận logic (Toán học) và Căn chỉnh an toàn (Alignment & Safety).

| Model Variant              | Validation PPL (↓)   | Math Pass@1 (%) (↑)  | Safety Refusal Rate (%) (↑)  |
|----------------------------|----------------------|----------------------|------------------------------|
| Base (100M)                | 2.84                 | 0.0                  | 100.0                        |
| SFT (Full Fine-Tuning)     | 3.19                 | 0.0                  | 100.0                        |
| SFT (Instruct + Math)      | 3.19                 | 0.0                  | 100.0                        |
| DPO (Aligned + Safe)       | 3.24                 | 0.0                  | 100.0                        |

*Generated at: 2026-11-25 07:00:27*