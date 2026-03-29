import hashlib
import json
import random
from pathlib import Path

from elsst_baselines.common.jsonl import read_jsonl
from elsst_baselines.generation.modeling import build_generation_prompt, canonicalize_generation_prompt


def serialize_concept_list(concepts):
    segments = []
    for item in concepts:
        term = str(item["term"]).strip()
        definition = " ".join(str(item["definition"]).strip().split())
        if not term or not definition:
            continue
        segments.append(f"{term}: {definition};")
    return " ".join(segments)


def load_track2_rows(path, max_rows=None):
    rows = read_jsonl(path)
    if max_rows is not None:
        return rows[:max_rows]
    return rows


def load_preference_rows(path, max_rows=None):
    return load_track2_rows(path, max_rows=max_rows)


def _stable_rejected_seed(row, pair_index):
    payload = {
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "pair_index": pair_index,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_sft_records(rows, tokenizer):
    records = []
    for row in rows:
        prompt = build_generation_prompt(row["prompt"], tokenizer, disable_thinking=True)
        response = serialize_concept_list(row["chosen"])
        records.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "response": response,
                "text": prompt + response,
                "gold_terms": [item["term"] for item in row["chosen"]],
            }
        )
    return records


def build_dpo_records(rows, tokenizer, pairs_per_row=2):
    records = []
    for row in rows:
        prompt = build_generation_prompt(row["prompt"], tokenizer, disable_thinking=True)
        chosen = serialize_concept_list(row["chosen"])
        rejected_pool = list(row["rejected"])
        sample_size = min(len(row["chosen"]), len(rejected_pool))
        if sample_size == 0:
            continue

        for pair_index in range(pairs_per_row):
            rng = random.Random(_stable_rejected_seed(row, pair_index))
            rejected = serialize_concept_list(rng.sample(rejected_pool, k=sample_size))
            records.append(
                {
                    "id": row["id"],
                    "pair_index": pair_index,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "gold_terms": [item["term"] for item in row["chosen"]],
                }
            )
    return records


def build_orpo_records(rows):
    records = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "prompt": canonicalize_generation_prompt(row["prompt"]),
                "chosen": serialize_concept_list(row["chosen"]),
                "rejected": serialize_concept_list(row["rejected"]),
                "gold_terms": [item["term"] for item in row["chosen"]],
            }
        )
    return records


def generation_dataset_summary(dataset_root, max_train_samples=None, max_eval_samples=None):
    dataset_root = Path(dataset_root)
    train_rows = load_track2_rows(dataset_root / "train.jsonl")
    val_rows = load_track2_rows(dataset_root / "val.jsonl")
    return {
        "dataset_root": str(dataset_root),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "requested_max_train_samples": max_train_samples,
        "requested_max_eval_samples": max_eval_samples,
        "effective_train_samples": min(len(train_rows), max_train_samples) if max_train_samples else len(train_rows),
        "effective_eval_samples": min(len(val_rows), max_eval_samples) if max_eval_samples else len(val_rows),
    }
