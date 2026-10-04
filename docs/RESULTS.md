# Results from Table 1

The three Markdown leaderboards reproduce the same seven models from Table 1 of the paper: Claude-4.6 Thinking, Gemini-3.5 Flash, GPT-5.5, DeepSeek-V4-Flash, Qwen3.5-397B-A17B, Qwen3.6-35B-A3B and Qwen3.5-9B. Model names and row order follow the paper.

| Dataset | Models | Scenarios per model | Evaluated cases per model |
| --- | ---: | ---: | ---: |
| Everyday Chat | 7 | 50 | 100 |
| Long-Horizon Character | 7 | 50 | 100 |
| Game Interaction | 7 | 59 | 118 |

All results use the Qwen3.5-397B-A17B judge. Each scenario evaluates both participant roles.

## Metrics

All six metrics use a 0–1 scale and preserve the four decimal places printed in Table 1:

- **Score**: overall quality score combining per-turn and whole-dialogue evidence. Scores average the evaluated sides within each scenario, then average scenarios with equal weight.
- **Per-Turn**: average per-turn score.
- **Holistic**: average whole-dialogue score.
- **ACC@85/90/95**: fraction of scenarios where both evaluated sides are L0-valid and reach a score of at least 0.85, 0.90 or 0.95, respectively.

These values are transcribed from Table 1 on page 6 of `iclr2027_conference.pdf`. Scores and pass rates retain the values and precision printed in the paper. No new model calls or judging were performed.

The published results consist only of the three Markdown tables. Per-model JSON files, case-level records and judge explanations are not included. New benchmark executions write to `runs/` and do not overwrite these tables.
