#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import export_standard_datasets as standard


def verify_track1_full_outputs(track1_full_dir):
    train_rows = standard.read_jsonl(track1_full_dir / "train.jsonl")
    val_rows = standard.read_jsonl(track1_full_dir / "val.jsonl")
    test_rows = standard.read_jsonl(track1_full_dir / "test.jsonl")
    concept_pool = standard.read_jsonl(track1_full_dir / "concept_pool.jsonl")

    if len(train_rows) != standard.EXPECTED_COUNTS["train"]:
        raise ValueError("track1_full train export count mismatch")
    if len(val_rows) != standard.EXPECTED_COUNTS["val"]:
        raise ValueError("track1_full val export count mismatch")
    if len(test_rows) != standard.EXPECTED_COUNTS["test"]:
        raise ValueError("track1_full test export count mismatch")
    if len(concept_pool) != standard.EXPECTED_CONCEPT_POOL_SIZE:
        raise ValueError("track1_full concept pool export count mismatch")

    standard.verify_unique_ids(train_rows, "track1_full train")
    standard.verify_unique_ids(val_rows, "track1_full val")
    standard.verify_unique_ids(test_rows, "track1_full test")
    standard.verify_split_prefix(train_rows, "train_", "track1_full train")
    standard.verify_split_prefix(val_rows, "val_", "track1_full val")
    standard.verify_split_prefix(test_rows, "test_", "track1_full test")

    for row in test_rows[:10]:
        if "generation_labels" not in row or "retrieval_labels" not in row:
            raise ValueError(f"track1_full test row missing labels for {row['id']}")


def verify_track2_full_outputs(track2_full_dir):
    train_rows = standard.read_jsonl(track2_full_dir / "train.jsonl")
    val_rows = standard.read_jsonl(track2_full_dir / "val.jsonl")
    test_rows = standard.read_jsonl(track2_full_dir / "test.jsonl")
    reject_pool = json.loads((track2_full_dir / "reject_concept_pool.json").read_text(encoding="utf-8"))

    if len(train_rows) != standard.EXPECTED_COUNTS["train"]:
        raise ValueError("track2_full train export count mismatch")
    if len(val_rows) != standard.EXPECTED_COUNTS["val"]:
        raise ValueError("track2_full val export count mismatch")
    if len(test_rows) != standard.EXPECTED_COUNTS["test"]:
        raise ValueError("track2_full test export count mismatch")
    if len(reject_pool) != standard.EXPECTED_REJECT_POOL_SIZE:
        raise ValueError("track2_full reject pool export count mismatch")

    standard.verify_unique_ids(train_rows, "track2_full train")
    standard.verify_unique_ids(val_rows, "track2_full val")
    standard.verify_unique_ids(test_rows, "track2_full test")
    standard.verify_split_prefix(train_rows, "train_", "track2_full train")
    standard.verify_split_prefix(val_rows, "val_", "track2_full val")
    standard.verify_split_prefix(test_rows, "test_", "track2_full test")

    for row in test_rows[:10]:
        if set(row.keys()) != {"id", "prompt", "chosen", "rejected"}:
            raise ValueError(f"track2_full test schema mismatch for {row['id']}")
        chosen_ids = {item["concept_id"] for item in row["chosen"]}
        rejected_ids = {item["concept_id"] for item in row["rejected"]}
        if not chosen_ids or not rejected_ids or chosen_ids & rejected_ids:
            raise ValueError(f"track2_full test labels invalid for {row['id']}")


def export_full_internal_datasets(source_root, output_root):
    raw_splits = standard.load_raw_samples(source_root)
    public_id_maps = standard.build_public_id_maps(raw_splits)
    concept_pool, _ = standard.build_concept_pool(raw_splits)
    similarity_index = standard.ConceptSimilarityIndex(concept_pool)

    track1_full_dir = output_root / "track1_full"
    standard.write_jsonl(
        track1_full_dir / "train.jsonl",
        standard.build_track1_rows(raw_splits["train"], similarity_index, True, "train", public_id_maps),
    )
    standard.write_jsonl(
        track1_full_dir / "val.jsonl",
        standard.build_track1_rows(raw_splits["val"], similarity_index, True, "val", public_id_maps),
    )
    standard.write_jsonl(
        track1_full_dir / "test.jsonl",
        standard.build_track1_rows(raw_splits["test"], similarity_index, True, "test", public_id_maps),
    )
    standard.write_jsonl(track1_full_dir / "concept_pool.jsonl", concept_pool)

    prompt_prefix = standard.infer_prompt_prefix(source_root)
    reject_pool = standard.load_reject_pool(source_root)
    track2_full_dir = output_root / "track2_full"
    standard.write_jsonl(
        track2_full_dir / "train.jsonl",
        standard.normalize_track2_train_rows(source_root, raw_splits["train"], public_id_maps),
    )
    standard.write_jsonl(
        track2_full_dir / "val.jsonl",
        standard.build_labeled_track2_rows(
            raw_splits["val"], prompt_prefix, reject_pool, public_id_maps, "val"
        ),
    )
    standard.write_jsonl(
        track2_full_dir / "test.jsonl",
        standard.build_labeled_track2_rows(
            raw_splits["test"], prompt_prefix, reject_pool, public_id_maps, "test"
        ),
    )
    standard.write_json(track2_full_dir / "reject_concept_pool.json", reject_pool)

    verify_track1_full_outputs(track1_full_dir)
    verify_track2_full_outputs(track2_full_dir)

    return {
        "track1_full": {
            "train": standard.EXPECTED_COUNTS["train"],
            "val": standard.EXPECTED_COUNTS["val"],
            "test": standard.EXPECTED_COUNTS["test"],
            "concept_pool": standard.EXPECTED_CONCEPT_POOL_SIZE,
        },
        "track2_full": {
            "train": standard.EXPECTED_COUNTS["train"],
            "val": standard.EXPECTED_COUNTS["val"],
            "test": standard.EXPECTED_COUNTS["test"],
            "reject_concept_pool": standard.EXPECTED_REJECT_POOL_SIZE,
        },
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export full internal ELSST track1 and track2 datasets with test labels."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo_root / "dataset",
        help="Path to the raw dataset source directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root,
        help="Path where track1_full/ and track2_full/ will be written.",
    )
    args = parser.parse_args()

    summary = export_full_internal_datasets(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
