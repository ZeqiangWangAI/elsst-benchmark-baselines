---
pretty_name: ELSST Track1
language:
  - en
task_categories:
  - text-retrieval
tags:
  - benchmark
  - information-retrieval
  - social-science
size_categories:
  - 1K<n<10K
---

# ELSST Track1: Implicit Concept Retrieval

ELSST Track1 evaluates whether a model can read a long synthetic social-science passage and retrieve the most relevant concepts from a fixed ELSST concept pool. The target is not lexical matching. The concepts are intentionally implicit, cross-sentence, and often require discourse-level reasoning over topic, framing, and social context.

This card is the authoritative task description for the retrieval track. The companion generation track is described in [`track2/README.md`](../track2/README.md). The reference code lives in [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines).

## Task

Given a passage, rank the ELSST concepts in `concept_pool.jsonl` by relevance.

- Train and validation contain gold concepts and hard negatives.
- `test_input.jsonl` does **not** expose any gold labels.
- Public sample identifiers are release-safe IDs of the form `split_legacy_id`, for example `train_t00023` and `val_v00029`.

## Splits

| Split | File | Rows | Labels exposed |
| --- | --- | ---: | --- |
| train | `train.jsonl` | 2,985 | Yes |
| val | `val.jsonl` | 756 | Yes |
| test | `test_input.jsonl` | 1,911 | No |
| pool | `concept_pool.jsonl` | 3,433 concepts | N/A |

## Schema

`train.jsonl` and `val.jsonl`

```json
{
  "id": "train_t00023",
  "text": "long passage...",
  "document_type": "op_ed",
  "generation_labels": [
    {"concept_id": "...", "term": "...", "definition": "..."}
  ],
  "retrieval_labels": {
    "positive_ids": ["..."],
    "hard_negative_ids": ["..."]
  }
}
```

`test_input.jsonl`

```json
{
  "id": "test_t00009",
  "text": "long passage...",
  "document_type": "research_summary"
}
```

## Evaluation

The baseline retrieval evaluator reports:

- `MRR`
- `Recall@5`
- `Recall@10`
- `NDCG@10`

Evaluation is defined over the validation split only. The public test split is input-only.

## Data Design And Quality

The corpus is fully synthetic. It was generated for benchmark construction and does not contain real personal data.

Document types are intentionally balanced across 10 genres:

- `blog_post`
- `case_study`
- `encyclopedia_entry`
- `forum_discussion`
- `interview_transcript`
- `news_article`
- `op_ed`
- `policy_brief`
- `report_excerpt`
- `research_summary`

Observed passage-length statistics in the current release:

| Split | Min | Median | Max |
| --- | ---: | ---: | ---: |
| train | 629 | 894 | 1,177 |
| val | 672 | 889.5 | 1,130 |
| test | 675 | 896 | 1,154 |

Label-count distributions:

- train: `{1: 605, 2: 598, 3: 579, 4: 613, 5: 590}`
- val: `{1: 153, 2: 153, 3: 152, 4: 155, 5: 143}`
- test gold labels are intentionally withheld

## Release Notes

Release date: March 29, 2026.

- All public IDs were normalized to `train_*`, `val_*`, and `test_*` so Track1 and Track2 share the same stable public ID mapping.

## Baseline Status

The reference retrieval code lives in [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines). The nominal baseline model is `Qwen/Qwen3-Embedding-0.6B`.

## Citation

If you use Track1, cite this benchmark release together with the companion Track2 card so the full task definition is preserved.
