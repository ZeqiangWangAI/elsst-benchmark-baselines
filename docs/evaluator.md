# ELSST Hosted Evaluator

The hosted evaluator is published as a Hugging Face Space:

- Space: <https://huggingface.co/spaces/JohnWang10086/elsst-evaluator>
- Direct app URL: <https://johnwang10086-elsst-evaluator.hf.space>

Use the Space to validate submission format on the public validation split before submitting hidden-test predictions to the leaderboard.

## User Workflow

1. Select `track1` or `track2`.
2. Upload a validation JSONL file.
3. Check the returned metrics and diagnostics.
4. Sign in with Hugging Face.
5. Upload the matching test JSONL file.

Test submissions are rate-limited to three submissions per Hugging Face user, per track, per 24 hours. Public test files remain input-only; hidden test labels are not released.

Programmatic submissions should pass an authenticated Hugging Face token as an `Authorization` header. The evaluator resolves the token with Hugging Face `whoami` and applies the same per-user rate limit.

```python
import os
from gradio_client import Client, handle_file

token = os.environ["HF_TOKEN"]
headers = {"Authorization": f"Bearer {token}"}
client = Client("https://johnwang10086-elsst-evaluator.hf.space", headers=headers)

val_result = client.predict(
    "track1",
    handle_file("track1_val_submission.jsonl"),
    api_name="/_score_val_ui",
    headers=headers,
)
val_state = val_result[1]

client.predict(
    "track1",
    "my-model-or-team",
    handle_file("track1_test_submission.jsonl"),
    val_state,
    api_name="/_submit_test_ui",
    headers=headers,
)
```

## Track1 Retrieval Format

Each JSONL row must contain an ID and ranked ELSST concept IDs:

```json
{"id": "val_v00029", "ranked_ids": ["concept_id_1", "concept_id_2", "concept_id_3"]}
```

Validation rules:

- Include every ID from the selected split exactly once.
- Do not include extra or duplicate row IDs.
- `ranked_ids` must be a list of concept IDs from `track1/concept_pool.jsonl`.
- Do not include duplicate or unknown concept IDs.
- Provide at least 10 ranked IDs for every row.
- Only the first 100 ranked IDs are used.

The primary leaderboard metric is `NDCG@10`. The evaluator also reports `MRR`, `Recall@5`, and `Recall@10`.

## Track2 Generation Format

Preferred format:

```json
{"id": "val_v00029", "predicted_terms": ["TERM 1", "TERM 2"]}
```

Fallback format for raw model output:

```json
{"id": "val_v00029", "raw_text": "TERM 1: short definition; TERM 2: short definition;"}
```

Validation rules:

- Include every ID from the selected split exactly once.
- Do not include extra or duplicate row IDs.
- `predicted_terms` must be a list of strings.
- `raw_text` is parsed with the same `term: definition;` parser used by the reference code.
- Empty predictions are allowed and score zero for that row.
- Terms are whitespace-normalized and deduplicated.
- Only the first 5 deduplicated terms are used.

The primary leaderboard metric is `semantic_f1`. Semantic matching uses BERTScore-based alignment with `tau = 0.85`. The evaluator also reports exact precision/recall/F1, semantic precision/recall/F1, parse rate, valid prediction rate, and average predicted set size.

## Example Submission Builders

Track1 validation skeleton:

```python
import json
from pathlib import Path

concept_ids = [
    json.loads(line)["concept_id"]
    for line in Path("track1/concept_pool.jsonl").open(encoding="utf-8")
]

with Path("track1/val.jsonl").open(encoding="utf-8") as source, Path("track1_val_submission.jsonl").open("w", encoding="utf-8") as output:
    for line in source:
        row = json.loads(line)
        output.write(json.dumps({"id": row["id"], "ranked_ids": concept_ids[:100]}) + "\n")
```

Track2 validation skeleton:

```python
import json
from pathlib import Path

with Path("track2/val.jsonl").open(encoding="utf-8") as source, Path("track2_val_submission.jsonl").open("w", encoding="utf-8") as output:
    for line in source:
        row = json.loads(line)
        output.write(json.dumps({"id": row["id"], "predicted_terms": []}) + "\n")
```

These examples are format checks only. They are not meaningful baselines.

## Deployment Notes

The Space code lives in `app.py` and `src/elsst_baselines/evaluator/`. It reads public validation files from the public dataset repositories or local release files. Hidden test labels are loaded from a private Hugging Face dataset configured through Space variables:

- `ELSST_PRIVATE_TRACK1_REPO`
- `ELSST_PRIVATE_TRACK1_TEST_FILE`
- `ELSST_PRIVATE_TRACK2_REPO`
- `ELSST_PRIVATE_TRACK2_TEST_FILE`

The Space needs a secret `HF_TOKEN` with access to the private hidden-label dataset. The current deployment uses `JohnWang10086/elsst-hidden-test-gold`, which contains only:

- `track1_full/test.jsonl`
- `track2_full/test.jsonl`

Do not commit hidden labels to this repository or to the public Space repository.

Leaderboard storage is configured through `ELSST_LEADERBOARD_DB=/data/elsst_leaderboard.sqlite`. The current runtime reports `storage=None`; leaderboard writes work while the Space is live, but durable persistence should be enabled in the Hugging Face UI or through a working storage API before public leaderboard use.
