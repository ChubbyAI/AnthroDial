# Everyday Chat

Judge: **Qwen3.5-397B-A17B**.

Values and model order follow Table 1 of the paper. All metrics use a 0–1 scale and are shown to four decimal places. See [result notes](../../docs/RESULTS.md).

| Model | Score | Per-Turn | Holistic | ACC@85 | ACC@90 | ACC@95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-4.6 Thinking | 0.9780 | 0.9722 | 0.9844 | 0.9400 | 0.9000 | 0.7800 |
| Gemini-3.5 Flash | 0.9780 | 0.9823 | 0.9747 | 0.9800 | 0.9600 | 0.7600 |
| GPT-5.5 | 0.8820 | 0.8058 | 0.9586 | 0.6400 | 0.5400 | 0.3000 |
| DeepSeek-V4-Flash | 0.9680 | 0.9779 | 0.9589 | 0.9400 | 0.8800 | 0.5200 |
| Qwen3.5-397B-A17B | 0.9410 | 0.9636 | 0.9191 | 0.8600 | 0.8000 | 0.4400 |
| Qwen3.6-35B-A3B | 0.9000 | 0.9362 | 0.8634 | 0.6000 | 0.4200 | 0.1400 |
| Qwen3.5-9B | 0.6410 | 0.7261 | 0.5558 | 0.1200 | 0.0400 | 0.0000 |
