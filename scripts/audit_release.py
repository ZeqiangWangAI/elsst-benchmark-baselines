#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def word_count(text):
    return len(text.split())


def unique_ids(rows, key="id"):
    ids = [row[key] for row in rows]
    return len(ids) == len(set(ids))


def summarize_raw_split(split, rows):
    counts = [word_count(row["text"]) for row in rows]
    return {
        "count": len(rows),
        "word_count": {
            "min": min(counts),
            "median": statistics.median(counts),
            "max": max(counts),
        },
        "document_types": dict(
            sorted(Counter(row["provenance"]["blueprint"]["document_type"] for row in rows).items())
        ),
        "label_sizes": dict(sorted(Counter(len(row["labels"]["concepts"]) for row in rows).items())),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit ELSST release correctness and quality.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dataset_root = repo_root / "dataset"
    track1_root = repo_root / "track1"
    track2_root = repo_root / "track2"

    raw_train = read_jsonl(dataset_root / "train" / "samples.jsonl")
    raw_val = read_jsonl(dataset_root / "val" / "samples.jsonl")
    raw_test = read_jsonl(dataset_root / "test" / "samples.jsonl")

    track1_train = read_jsonl(track1_root / "train.jsonl")
    track1_val = read_jsonl(track1_root / "val.jsonl")
    track1_test = read_jsonl(track1_root / "test_input.jsonl")
    track2_train = read_jsonl(track2_root / "train.jsonl")
    track2_val = read_jsonl(track2_root / "val.jsonl")
    track2_test = read_jsonl(track2_root / "test_input.jsonl")
    reject_pool = json.loads((track2_root / "reject_concept_pool.json").read_text(encoding="utf-8"))
    train_preference = read_jsonl(dataset_root / "preference" / "train_preference.jsonl")

    reject_pool_ids = {row["concept_id"] for row in reject_pool}
    track2_val_consistency_violations = 0
    for row in track2_val:
        chosen_ids = {item["concept_id"] for item in row["chosen"]}
        rejected_ids = {item["concept_id"] for item in row["rejected"]}
        if chosen_ids & rejected_ids or not rejected_ids.issubset(reject_pool_ids):
            track2_val_consistency_violations += 1

    prompt_templates = Counter(row["prompt"].split("\n\n", 1)[0] for row in train_preference)

    report = {
        "correctness": {
            "source_val_ids_unique": unique_ids(raw_val, key="sample_id"),
            "track1": {
                "counts": {
                    "train": len(track1_train),
                    "val": len(track1_val),
                    "test_input": len(track1_test),
                },
                "split_ids_unique": {
                    "train": unique_ids(track1_train),
                    "val": unique_ids(track1_val),
                    "test": unique_ids(track1_test),
                },
                "cross_split_ids_unique": len(
                    {row["id"] for row in track1_train + track1_val + track1_test}
                )
                == len(track1_train) + len(track1_val) + len(track1_test),
                "test_schema": sorted(track1_test[0].keys()),
                "val_query_index_size": len({row["id"]: row for row in track1_val}),
            },
            "track2": {
                "counts": {
                    "train": len(track2_train),
                    "val": len(track2_val),
                    "test_input": len(track2_test),
                    "reject_concept_pool": len(reject_pool),
                },
                "split_ids_unique": {
                    "train": unique_ids(track2_train),
                    "val": unique_ids(track2_val),
                    "test": unique_ids(track2_test),
                },
                "cross_split_ids_unique": len(
                    {row["id"] for row in track2_train + track2_val + track2_test}
                )
                == len(track2_train) + len(track2_val) + len(track2_test),
                "id_alignment_with_track1": {
                    "train": [row["id"] for row in track1_train] == [row["id"] for row in track2_train],
                    "val": [row["id"] for row in track1_val] == [row["id"] for row in track2_val],
                    "test": [row["id"] for row in track1_test] == [row["id"] for row in track2_test],
                },
                "test_schema": sorted(track2_test[0].keys()),
                "val_query_index_size": len({row["id"]: row for row in track2_val}),
                "val_reject_pool_consistency_violations": track2_val_consistency_violations,
            },
        },
        "quality": {
            "raw_splits": {
                "train": summarize_raw_split("train", raw_train),
                "val": summarize_raw_split("val", raw_val),
                "test": summarize_raw_split("test", raw_test),
            },
            "track2_train_prompt_templates": len(prompt_templates),
            "track2_train_chosen_sizes": dict(
                sorted(Counter(len(row["chosen"]) for row in train_preference).items())
            ),
            "track2_train_rejected_sizes": dict(
                sorted(Counter(len(row["rejected"]) for row in train_preference).items())
            ),
            "track2_val_rejected_sizes": dict(
                sorted(Counter(len(row["rejected"]) for row in track2_val).items())
            ),
        },
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
