---
pretty_name: ELSST Track2
language:
  - en
task_categories:
  - text-generation
tags:
  - benchmark
  - generation
  - social-science
size_categories:
  - 1K<n<10K
---

# ELSST Track2: Open Knowledge Discovery

ELSST Track2 evaluates whether a model can read the same long synthetic passage used in Track1 and generate the latent ELSST concepts as a semantic set. The target is a small concept set, not a free-form explanation. Models must recover implicit concepts that are grounded in the passage but not always directly named.

This card is the authoritative task description for the generation track. The published dataset lives at [JohnWang10086/elsst-track2](https://huggingface.co/datasets/JohnWang10086/elsst-track2). The companion retrieval track is published at [JohnWang10086/elsst-track1](https://huggingface.co/datasets/JohnWang10086/elsst-track1). The reference code lives in [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines).

## Task

Given a passage-level prompt, output the hidden concepts as a short semantic set.

- Train and validation expose `chosen` gold concepts and `rejected` distractors for post-training and evaluation.
- `test_input.jsonl` exposes only `id` and `prompt`.
- Public sample identifiers are release-safe IDs of the form `split_legacy_id`, aligned one-to-one with Track1.

## Splits

| Split | File | Rows | Labels exposed |
| --- | --- | ---: | --- |
| train | `train.jsonl` | 2,985 | Yes |
| val | `val.jsonl` | 756 | Yes |
| test | `test_input.jsonl` | 1,911 | No |
| pool | `reject_concept_pool.json` | 5,342 concepts | Distractor pool |

## Schema

`train.jsonl` and `val.jsonl`

```json
{
  "id": "val_v00029",
  "prompt": "Find the hidden concepts ...",
  "chosen": [
    {"concept_id": "...", "term": "...", "definition": "..."}
  ],
  "rejected": [
    {"concept_id": "...", "term": "...", "definition": "..."}
  ]
}
```

`test_input.jsonl`

```json
{
  "id": "test_t00009",
  "prompt": "Find the hidden concepts ..."
}
```

## Evaluation

The reference evaluator reports:

- `parse_rate`
- `json_parse_rate`
- `average_predicted_terms`
- exact `precision`, `recall`, `f1`
- semantic-set `precision`, `recall`, `f1`

Semantic-set matching uses BERTScore-based alignment with `tau = 0.85` in the baseline code.

## Data Design And Quality

The corpus is fully synthetic and was generated for benchmark construction. No real personal data is included.

Observed passage-length statistics in the underlying released corpus:

| Split | Min | Median | Max |
| --- | ---: | ---: | ---: |
| train | 629 | 894 | 1,177 |
| val | 672 | 889.5 | 1,130 |
| test | 675 | 896 | 1,154 |

Prompt and supervision properties in the released Track2 source:

- prompt templates in train source: `1`
- chosen concept counts: `{1: 605, 2: 598, 3: 579, 4: 613, 5: 590}`
- rejected concept counts in train source: `{8: 7, 9: 20, 10: 1302, 11: 340, 12: 1170, 13: 51, 14: 77, 15: 17, 16: 1}`
- rejected concept counts in validation export: `{11: 756}`
- validation reject-pool consistency violations in this release: `0`

## Release Notes

Release date: March 29, 2026.

- Public IDs were normalized to `train_*`, `val_*`, and `test_*`, shared with Track1.
- The DPO negative-sampling seed in the reference code was decoupled from `row["id"]` and moved to a stable content signature so public ID normalization no longer changes training pair construction.

## Baseline Status

The reference generation stack lives in [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines).

- nominal base model: `Qwen/Qwen3.5-4B`
- recommended training path: `train_sft` -> `train_dpo`
- legacy path retained for compatibility: `train_orpo`

## Citation

If you use Track2, cite this benchmark release together with the companion Track1 card so the full task definition is preserved.
