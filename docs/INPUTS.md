# Preparing evaluation inputs

This release includes scoring rubrics and prompt assets, but not the original persona cards, scenario cards, bindings or raw conversations. Put your own inputs under the paths configured in `configs/<domain>/*.yaml`. The private input locations and runtime outputs are ignored by Git.

```text
data/<domain>_benchmark/
  rubric.yaml                  included scoring rule
  bindings.yaml                your list of persona/scenario pairings
  personas/<persona_id>/card.yaml
  personas/<persona_id>/memory.md    optional
  scenarios/                   your scenario cards
  skills/                      optional prompt assets
  dialogues.jsonl              needed only for real_replay mode
```

A binding is a mapping in a YAML list, with `persona_a`, `persona_b` and `scenario_id`; `tested_role` can select a side for single-role evaluation. IDs must match supplied cards. By default both sides are evaluated. Persona cards are loaded by `libs/chat/persona.py`; scenario formats and nested catalog resolution are implemented by `libs/chat/scenario.py`. A catalog, when required by your layout, can be supplied through `paths.catalog_path`.

Illustrative binding using synthetic identifiers (not an original benchmark sample):

```yaml
- persona_a: DEMO_A
  persona_b: DEMO_B
  scenario_id: A0001_B0001_C0001
```

A minimal persona card contains `persona_id` and `nickname`; richer fields such as occupation, relationship, personality and response habits are rendered into the prompt. Scenario cards contain the matching `scenario_id`, name, category/sub-category, relationship, start time, place, trigger event and emotion tone as appropriate. They must use the schema accepted by `Scenario.load`; simply creating a binding without its cards is insufficient.

`real_dialogues.yaml` configurations for Game Interaction and Long-Horizon Character use `benchmark.mode: real_replay`. They require your own JSONL input at `benchmark.real_dialogues_path`; the accepted trace format is implemented in `apps/benchmark/src/real_replay.py`. No human baseline results or underlying human conversations are included in this release.

For a new execution, set model service endpoints and API-key environment variables, then invoke `run_benchmark.py`. To recompute scores from your cached traces, use `rescore_benchmark.py --no-rejudge` when no additional LLM calls are intended. `rejudge_benchmark.py` calls the judge again. All execution output goes to the configured `runs/` directory.
