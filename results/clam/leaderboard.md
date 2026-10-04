# Long-Horizon Character

Judge: **Qwen3.5-397B-A17B**.

Values and model order follow Table 1 of the paper. All metrics use a 0–1 scale and are shown to four decimal places. See [result notes](../../docs/RESULTS.md).

| Model | Score | Per-Turn | Holistic | ACC@85 | ACC@90 | ACC@95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-4.6 Thinking | 0.9860 | 0.9823 | 0.9888 | 0.9800 | 0.9600 | 0.8600 |
| Gemini-3.5 Flash | 0.9720 | 0.9952 | 0.9494 | 0.9000 | 0.8400 | 0.7600 |
| GPT-5.5 | 0.9820 | 0.9941 | 0.9704 | 0.9600 | 0.9400 | 0.9000 |
| DeepSeek-V4-Flash | 0.9460 | 0.9957 | 0.8966 | 0.7800 | 0.6800 | 0.5200 |
| Qwen3.5-397B-A17B | 0.7600 | 0.8728 | 0.6474 | 0.3600 | 0.2400 | 0.1600 |
| Qwen3.6-35B-A3B | 0.6930 | 0.9006 | 0.4857 | 0.0800 | 0.0400 | 0.0000 |
| Qwen3.5-9B | 0.2120 | 0.3374 | 0.0871 | 0.0000 | 0.0000 | 0.0000 |
