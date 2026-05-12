# ELSST Benchmark Baselines

This repository contains the release tooling, baseline code, and verification workflow for the ELSST benchmark.

GitHub repository: [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines)

ELSST is published as two coordinated tracks:

- [Track1 card](track1/README.md): implicit concept retrieval from a fixed ELSST concept pool. Dataset: [JohnWang10086/elsst-track1](https://huggingface.co/datasets/JohnWang10086/elsst-track1)
- [Track2 card](track2/README.md): open concept discovery as semantic-set generation. Dataset: [JohnWang10086/elsst-track2](https://huggingface.co/datasets/JohnWang10086/elsst-track2)

Treat those two cards as the full benchmark specification. They are the release documents that should be cited together when describing the task.

## Release Conventions

- Track1 public IDs use stable `train_*`, `val_*`, and `test_*` identifiers.
- Track2 shares the same public IDs as Track1.
- Track2 DPO sampling is seeded from row content rather than `row["id"]`.

The public test splits remain input-only. No gold labels are exposed in:

- [`track1/test_input.jsonl`](track1/test_input.jsonl)
- [`track2/test_input.jsonl`](track2/test_input.jsonl)

## Repository Scope

This GitHub repository hosts code, tests, scripts, and track cards. The dataset payloads themselves are intended for two separate Hugging Face dataset repositories:

- [JohnWang10086/elsst-track1](https://huggingface.co/datasets/JohnWang10086/elsst-track1)
- [JohnWang10086/elsst-track2](https://huggingface.co/datasets/JohnWang10086/elsst-track2)

## Baseline Entrypoints

- `python -m elsst_baselines.retrieval.train`
- `python -m elsst_baselines.retrieval.evaluate`
- `python -m elsst_baselines.retrieval.infer`
- `python -m elsst_baselines.generation.train_sft`
- `python -m elsst_baselines.generation.train_dpo`
- `python -m elsst_baselines.generation.train_orpo`
- `python -m elsst_baselines.generation.evaluate`
- `python -m elsst_baselines.remote.run`

Recommended Track2 training path:

1. Run `train_sft` on `track2/train.jsonl`.
2. Run `train_dpo` from the SFT adapter.
3. Evaluate with `python -m elsst_baselines.generation.evaluate --dataset-root track2 ...`.

## Hosted Evaluator

The hosted evaluator is available at [JohnWang10086/elsst-evaluator](https://huggingface.co/spaces/JohnWang10086/elsst-evaluator).

Workflow:

1. Select Track1 or Track2.
2. Upload validation predictions first. This checks the submission schema and reports validation metrics.
3. Sign in with Hugging Face.
4. Upload test predictions for the same track to submit to the leaderboard.

Test submissions require Hugging Face identity through browser login or an authenticated Gradio client request with `Authorization: Bearer $HF_TOKEN`, and are rate-limited to three submissions per user, per track, per 24 hours. Public test files remain input-only; hidden test labels are not included in this repository or in the public Space repository.

Submission formats:

- Track1 retrieval: JSONL rows of `{"id": "...", "ranked_ids": ["concept_id", "..."]}`.
- Track2 generation: JSONL rows of `{"id": "...", "predicted_terms": ["term", "..."]}`. A `raw_text` fallback is accepted for generated `term: definition;` style output.

See [`docs/evaluator.md`](docs/evaluator.md) for complete submission and deployment details.

## Verification

Re-export the standardized datasets:

```bash
python3 scripts/export_standard_datasets.py --source-root dataset --output-root .
```

Generate a release summary:

```bash
python3 scripts/audit_release.py --repo-root . --output release/release_audit.json
```

Run regression tests:

```bash
python3 -m unittest discover -s tests -p 'test_export_standard_datasets.py' -v
python3 -m unittest discover -s tests -p 'test_baselines_*.py' -v
python3 -m unittest discover -s tests -p 'test_evaluator.py' -v
```
