# KV Cache Profiling Report (100M Transformer)

- Device: `cuda`
- Generated tokens per run: `64`

| Context Length (L) | Mode | Throughput (tok/s) ↑ | TTFT (ms) ↓ | TPOT (ms/tok) ↓ | Peak VRAM (MB) | Speedup |
|---|---|---|---|---|---|---|
| 128 | **KV Cache ON** | 74.58 | 19.63 | 13.31 | 1771.81 | **1.27x** |
| 128 | KV Cache OFF | 58.57 | 15.70 | 17.09 | 1790.50 | 1.00x |
| 512 | **KV Cache ON** | 68.71 | 36.01 | 14.21 | 1854.44 | **3.18x** |
| 512 | KV Cache OFF | 21.63 | 42.12 | 46.29 | 1884.20 | 1.00x |
| 1024 | **KV Cache ON** | 68.67 | 94.18 | 13.30 | 1950.07 | **7.41x** |
| 1024 | KV Cache OFF | 9.27 | 96.33 | 108.06 | 2010.40 | 1.00x |
| 2048 | **KV Cache ON** | 59.33 | 249.95 | 13.15 | 2305.45 | **16.53x** |
| 2048 | KV Cache OFF | 3.59 | 258.63 | 279.21 | 2452.28 | 1.00x |
