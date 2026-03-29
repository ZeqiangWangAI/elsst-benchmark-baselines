#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_COUNTS = {
    "train": 2985,
    "val": 756,
    "test": 1911,
}
EXPECTED_CONCEPT_POOL_SIZE = 3433
EXPECTED_REJECT_POOL_SIZE = 5342
HARD_NEGATIVE_K = 20
VAL_REJECTED_K = 11
RANDOM_SEED = 20260321
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def normalize_gold_concept(raw_concept):
    return {
        "concept_id": raw_concept["id"],
        "term": raw_concept["label_en"],
        "definition": raw_concept["definition"],
    }


def normalize_track2_concept(raw_concept):
    return {
        "concept_id": raw_concept.get("concept_id", raw_concept.get("id")),
        "term": raw_concept["term"],
        "definition": raw_concept["definition"],
    }


def make_public_id(split, legacy_id):
    return f"{split}_{legacy_id}"


def stable_seed(payload):
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_raw_split(source_root, split):
    path = source_root / split / "samples.jsonl"
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split} count mismatch: expected {EXPECTED_COUNTS[split]}, got {len(rows)}"
        )

    seen_ids = set()
    for row in rows:
        if row["split"] != split:
            raise ValueError(f"raw row split mismatch: expected {split}, got {row['split']}")
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"duplicate raw sample_id within {split}: {sample_id}")
        seen_ids.add(sample_id)
    return rows


def load_raw_samples(source_root):
    return {
        split: read_raw_split(source_root, split)
        for split in ("train", "val", "test")
    }


def build_public_id_maps(raw_splits):
    split_maps = {}
    seen_public_ids = set()
    for split, rows in raw_splits.items():
        split_map = {}
        for row in rows:
            public_id = make_public_id(split, row["sample_id"])
            if public_id in seen_public_ids:
                raise ValueError(f"public id collision detected: {public_id}")
            split_map[row["sample_id"]] = public_id
            seen_public_ids.add(public_id)
        split_maps[split] = split_map
    return split_maps


def extract_document_type(sample):
    return sample["provenance"]["blueprint"]["document_type"]


def build_concept_pool(raw_splits):
    concept_map = {}
    split_concept_sets = {}
    for split, rows in raw_splits.items():
        split_ids = set()
        for row in rows:
            for raw_concept in row["labels"]["concepts"]:
                concept = normalize_gold_concept(raw_concept)
                concept_id = concept["concept_id"]
                split_ids.add(concept_id)
                if concept_id in concept_map and concept_map[concept_id] != concept:
                    raise ValueError(f"inconsistent concept metadata for {concept_id}")
                concept_map[concept_id] = concept
        split_concept_sets[split] = split_ids

    if len(concept_map) != EXPECTED_CONCEPT_POOL_SIZE:
        raise ValueError(
            "concept pool size mismatch: "
            f"expected {EXPECTED_CONCEPT_POOL_SIZE}, got {len(concept_map)}"
        )

    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left >= right:
                continue
            overlap = split_concept_sets[left] & split_concept_sets[right]
            if overlap:
                raise ValueError(f"concept leakage between {left} and {right}: {len(overlap)}")

    concept_pool = [concept_map[concept_id] for concept_id in sorted(concept_map)]
    return concept_pool, split_concept_sets


class ConceptSimilarityIndex:
    def __init__(self, concept_pool):
        self.concept_ids = [concept["concept_id"] for concept in concept_pool]
        self.sorted_concept_ids = sorted(self.concept_ids)
        documents = {
            concept["concept_id"]: tokenize(f"{concept['term']} {concept['definition']}")
            for concept in concept_pool
        }

        doc_freq = Counter()
        for tokens in documents.values():
            doc_freq.update(set(tokens))

        total_docs = len(documents)
        self.vectors = {}
        self.inverted = defaultdict(list)

        for concept_id, tokens in documents.items():
            counts = Counter(tokens)
            weights = {}
            norm = 0.0
            for token, count in counts.items():
                tf = 1.0 + math.log(count)
                idf = math.log((1.0 + total_docs) / (1.0 + doc_freq[token])) + 1.0
                weight = tf * idf
                weights[token] = weight
                norm += weight * weight
            norm = math.sqrt(norm) or 1.0
            normalized = {token: weight / norm for token, weight in weights.items()}
            self.vectors[concept_id] = normalized
            for token, weight in normalized.items():
                self.inverted[token].append((concept_id, weight))

        self._neighbors = {}

    def nearest_neighbors(self, concept_id, limit=64):
        if concept_id in self._neighbors:
            return self._neighbors[concept_id][:limit]

        query_vector = self.vectors[concept_id]
        scores = defaultdict(float)
        for token, query_weight in query_vector.items():
            for other_id, other_weight in self.inverted[token]:
                if other_id == concept_id:
                    continue
                scores[other_id] += query_weight * other_weight

        ranked = [
            other_id
            for other_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]
        self._neighbors[concept_id] = ranked
        return ranked[:limit]

    def sample_hard_negatives(self, positive_ids):
        positives = set(positive_ids)
        ranked = []
        seen = set()

        for concept_id in positive_ids:
            for candidate_id in self.nearest_neighbors(concept_id):
                if candidate_id in positives or candidate_id in seen:
                    continue
                ranked.append(candidate_id)
                seen.add(candidate_id)
                if len(ranked) == HARD_NEGATIVE_K:
                    return ranked

        for candidate_id in self.sorted_concept_ids:
            if candidate_id in positives or candidate_id in seen:
                continue
            ranked.append(candidate_id)
            if len(ranked) == HARD_NEGATIVE_K:
                break

        return ranked


def build_track1_rows(raw_rows, similarity_index, include_labels, split, public_id_maps):
    output_rows = []
    for row in raw_rows:
        output = {
            "id": public_id_maps[split][row["sample_id"]],
            "text": row["text"],
            "document_type": extract_document_type(row),
        }
        if include_labels:
            generation_labels = [
                normalize_gold_concept(raw_concept) for raw_concept in row["labels"]["concepts"]
            ]
            positive_ids = [concept["concept_id"] for concept in generation_labels]
            output["generation_labels"] = generation_labels
            output["retrieval_labels"] = {
                "positive_ids": positive_ids,
                "hard_negative_ids": similarity_index.sample_hard_negatives(positive_ids),
            }
        output_rows.append(output)
    return output_rows


def infer_prompt_prefix(source_root):
    train_rows = read_jsonl(source_root / "train" / "samples.jsonl")
    train_text_by_id = {row["sample_id"]: row["text"] for row in train_rows}
    preference_rows = read_jsonl(source_root / "preference" / "train_preference.jsonl")
    prefixes = set()
    for row in preference_rows:
        text = train_text_by_id[row["id"]]
        if not row["prompt"].endswith(text):
            raise ValueError(f"prompt for {row['id']} does not end with raw text")
        prefixes.add(row["prompt"][: -len(text)])
    if len(prefixes) != 1:
        raise ValueError(f"expected a single prompt template, found {len(prefixes)}")
    return prefixes.pop()


def normalize_track2_train_rows(source_root, raw_train_rows, public_id_maps):
    source_rows = read_jsonl(source_root / "preference" / "train_preference.jsonl")
    if len(source_rows) != EXPECTED_COUNTS["train"]:
        raise ValueError("train preference source count mismatch")

    by_legacy_id = {}
    for row in source_rows:
        legacy_id = row["id"]
        if legacy_id in by_legacy_id:
            raise ValueError(f"duplicate train preference id: {legacy_id}")
        by_legacy_id[legacy_id] = row

    normalized = []
    for raw_row in raw_train_rows:
        legacy_id = raw_row["sample_id"]
        if legacy_id not in by_legacy_id:
            raise ValueError(f"missing train preference row for {legacy_id}")
        row = by_legacy_id[legacy_id]
        normalized.append(
            {
                "id": public_id_maps["train"][legacy_id],
                "prompt": row["prompt"],
                "chosen": [normalize_track2_concept(item) for item in row["chosen"]],
                "rejected": [normalize_track2_concept(item) for item in row["rejected"]],
            }
        )
    return normalized


def build_val_track2_rows(raw_val_rows, prompt_prefix, reject_pool, public_id_maps):
    output_rows = []
    for row in raw_val_rows:
        chosen = [normalize_gold_concept(item) for item in row["labels"]["concepts"]]
        chosen_ids = sorted(item["concept_id"] for item in chosen)
        candidate_pool = [item for item in reject_pool if item["concept_id"] not in set(chosen_ids)]
        rng = random.Random(
            stable_seed(
                {
                    "seed": RANDOM_SEED,
                    "text": row["text"],
                    "chosen_ids": chosen_ids,
                }
            )
        )
        rejected = rng.sample(candidate_pool, k=min(VAL_REJECTED_K, len(candidate_pool)))
        output_rows.append(
            {
                "id": public_id_maps["val"][row["sample_id"]],
                "prompt": f"{prompt_prefix}{row['text']}",
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return output_rows


def build_test_track2_rows(raw_test_rows, prompt_prefix, public_id_maps):
    output_rows = []
    for row in raw_test_rows:
        output_rows.append(
            {
                "id": public_id_maps["test"][row["sample_id"]],
                "prompt": f"{prompt_prefix}{row['text']}",
            }
        )
    return output_rows


def load_reject_pool(source_root):
    rows = json.loads(
        (source_root / "preference" / "reject_concept_pool.json").read_text(encoding="utf-8")
    )
    normalized = [normalize_track2_concept(row) for row in rows]
    if len(normalized) != EXPECTED_REJECT_POOL_SIZE:
        raise ValueError(
            "reject concept pool size mismatch: "
            f"expected {EXPECTED_REJECT_POOL_SIZE}, got {len(normalized)}"
        )
    if len({row["concept_id"] for row in normalized}) != len(normalized):
        raise ValueError("reject concept pool contains duplicate concept ids")
    return normalized


def verify_unique_ids(rows, label):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids detected in {label}")


def verify_split_prefix(rows, prefix, label):
    for row in rows:
        if not row["id"].startswith(prefix):
            raise ValueError(f"{label} row has unexpected id prefix: {row['id']}")


def verify_track1_outputs(track1_dir):
    train_rows = read_jsonl(track1_dir / "train.jsonl")
    val_rows = read_jsonl(track1_dir / "val.jsonl")
    test_rows = read_jsonl(track1_dir / "test_input.jsonl")
    concept_pool = read_jsonl(track1_dir / "concept_pool.jsonl")

    if len(train_rows) != EXPECTED_COUNTS["train"]:
        raise ValueError("track1 train export count mismatch")
    if len(val_rows) != EXPECTED_COUNTS["val"]:
        raise ValueError("track1 val export count mismatch")
    if len(test_rows) != EXPECTED_COUNTS["test"]:
        raise ValueError("track1 test export count mismatch")
    if len({row["concept_id"] for row in concept_pool}) != EXPECTED_CONCEPT_POOL_SIZE:
        raise ValueError("track1 concept pool export count mismatch")

    verify_unique_ids(train_rows, "track1 train")
    verify_unique_ids(val_rows, "track1 val")
    verify_unique_ids(test_rows, "track1 test")
    verify_split_prefix(train_rows, "train_", "track1 train")
    verify_split_prefix(val_rows, "val_", "track1 val")
    verify_split_prefix(test_rows, "test_", "track1 test")

    if len({row["id"] for row in train_rows + val_rows + test_rows}) != sum(
        len(rows) for rows in (train_rows, val_rows, test_rows)
    ):
        raise ValueError("track1 ids collide across splits")

    train_concepts = {
        item["concept_id"] for row in train_rows for item in row["generation_labels"]
    }
    val_concepts = {
        item["concept_id"] for row in val_rows for item in row["generation_labels"]
    }
    test_blob = json.dumps(test_rows[:10], ensure_ascii=False)

    if train_concepts & val_concepts:
        raise ValueError("train and val concepts overlap in track1 export")
    if '"labels"' in test_blob or '"provenance"' in test_blob:
        raise ValueError("track1 test export leaked raw metadata")
    if len({row["id"]: row for row in val_rows}) != len(val_rows):
        raise ValueError("track1 val ids do not form a one-to-one query index")

    pool_ids = {row["concept_id"] for row in concept_pool}
    for row in train_rows + val_rows:
        positive_ids = set(row["retrieval_labels"]["positive_ids"])
        hard_negative_ids = set(row["retrieval_labels"]["hard_negative_ids"])
        if not positive_ids:
            raise ValueError(f"missing positive ids for {row['id']}")
        if not hard_negative_ids:
            raise ValueError(f"missing hard negatives for {row['id']}")
        if positive_ids & hard_negative_ids:
            raise ValueError(f"overlapping positives and negatives for {row['id']}")
        if not positive_ids.issubset(pool_ids) or not hard_negative_ids.issubset(pool_ids):
            raise ValueError(f"track1 export references concept outside concept_pool for {row['id']}")


def verify_track2_outputs(track2_dir):
    train_rows = read_jsonl(track2_dir / "train.jsonl")
    val_rows = read_jsonl(track2_dir / "val.jsonl")
    test_rows = read_jsonl(track2_dir / "test_input.jsonl")
    reject_pool = json.loads((track2_dir / "reject_concept_pool.json").read_text(encoding="utf-8"))

    if len(train_rows) != EXPECTED_COUNTS["train"]:
        raise ValueError("track2 train export count mismatch")
    if len(val_rows) != EXPECTED_COUNTS["val"]:
        raise ValueError("track2 val export count mismatch")
    if len(test_rows) != EXPECTED_COUNTS["test"]:
        raise ValueError("track2 test export count mismatch")
    if len(reject_pool) != EXPECTED_REJECT_POOL_SIZE:
        raise ValueError("track2 reject pool export count mismatch")

    verify_unique_ids(train_rows, "track2 train")
    verify_unique_ids(val_rows, "track2 val")
    verify_unique_ids(test_rows, "track2 test")
    verify_split_prefix(train_rows, "train_", "track2 train")
    verify_split_prefix(val_rows, "val_", "track2 val")
    verify_split_prefix(test_rows, "test_", "track2 test")

    if len({row["id"] for row in train_rows + val_rows + test_rows}) != sum(
        len(rows) for rows in (train_rows, val_rows, test_rows)
    ):
        raise ValueError("track2 ids collide across splits")
    if len({row["concept_id"] for row in reject_pool}) != len(reject_pool):
        raise ValueError("track2 reject concept pool contains duplicate concept ids")
    if len({row["id"]: row for row in val_rows}) != len(val_rows):
        raise ValueError("track2 val ids do not form a one-to-one evaluation index")

    for row in train_rows + val_rows:
        chosen_ids = {item["concept_id"] for item in row["chosen"]}
        rejected_ids = {item["concept_id"] for item in row["rejected"]}
        if not chosen_ids:
            raise ValueError(f"missing chosen concepts for {row['id']}")
        if not rejected_ids:
            raise ValueError(f"missing rejected concepts for {row['id']}")
        if chosen_ids & rejected_ids:
            raise ValueError(f"chosen and rejected overlap for {row['id']}")

    for row in test_rows:
        if set(row.keys()) != {"id", "prompt"}:
            raise ValueError(f"track2 test schema leaked labels for {row['id']}")


def verify_track_alignment(track1_dir, track2_dir):
    for filename in ("train.jsonl", "val.jsonl", "test_input.jsonl"):
        track1_rows = read_jsonl(track1_dir / filename)
        track2_rows = read_jsonl(track2_dir / filename)
        if [row["id"] for row in track1_rows] != [row["id"] for row in track2_rows]:
            raise ValueError(f"track1 and track2 ids diverge for {filename}")


def export_standard_datasets(source_root, output_root):
    raw_splits = load_raw_samples(source_root)
    public_id_maps = build_public_id_maps(raw_splits)
    concept_pool, _ = build_concept_pool(raw_splits)
    similarity_index = ConceptSimilarityIndex(concept_pool)

    track1_dir = output_root / "track1"
    write_jsonl(
        track1_dir / "train.jsonl",
        build_track1_rows(raw_splits["train"], similarity_index, True, "train", public_id_maps),
    )
    write_jsonl(
        track1_dir / "val.jsonl",
        build_track1_rows(raw_splits["val"], similarity_index, True, "val", public_id_maps),
    )
    write_jsonl(
        track1_dir / "test_input.jsonl",
        build_track1_rows(raw_splits["test"], similarity_index, False, "test", public_id_maps),
    )
    write_jsonl(track1_dir / "concept_pool.jsonl", concept_pool)

    prompt_prefix = infer_prompt_prefix(source_root)
    reject_pool = load_reject_pool(source_root)
    track2_dir = output_root / "track2"
    write_jsonl(
        track2_dir / "train.jsonl",
        normalize_track2_train_rows(source_root, raw_splits["train"], public_id_maps),
    )
    write_jsonl(
        track2_dir / "val.jsonl",
        build_val_track2_rows(raw_splits["val"], prompt_prefix, reject_pool, public_id_maps),
    )
    write_jsonl(
        track2_dir / "test_input.jsonl",
        build_test_track2_rows(raw_splits["test"], prompt_prefix, public_id_maps),
    )
    write_json(track2_dir / "reject_concept_pool.json", reject_pool)

    verify_track1_outputs(track1_dir)
    verify_track2_outputs(track2_dir)
    verify_track_alignment(track1_dir, track2_dir)

    return {
        "track1": {
            "train": EXPECTED_COUNTS["train"],
            "val": EXPECTED_COUNTS["val"],
            "test_input": EXPECTED_COUNTS["test"],
            "concept_pool": EXPECTED_CONCEPT_POOL_SIZE,
            "hard_negative_k": HARD_NEGATIVE_K,
        },
        "track2": {
            "train": EXPECTED_COUNTS["train"],
            "val": EXPECTED_COUNTS["val"],
            "test_input": EXPECTED_COUNTS["test"],
            "reject_concept_pool": len(reject_pool),
            "val_rejected_k": VAL_REJECTED_K,
        },
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export the standard ELSST track1 and track2 datasets."
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
        help="Path where track1/ and track2/ will be written.",
    )
    args = parser.parse_args()

    summary = export_standard_datasets(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
