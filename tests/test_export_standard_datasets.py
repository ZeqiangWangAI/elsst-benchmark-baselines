import json
import subprocess
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path("/Users/zeqiangwang/Desktop/ELSST")
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_standard_datasets.py"
SOURCE_ROOT = REPO_ROOT / "dataset"


@lru_cache(maxsize=1)
def build_standard_datasets():
    tmpdir = tempfile.TemporaryDirectory()
    output_root = Path(tmpdir.name)
    cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--source-root",
        str(SOURCE_ROOT),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            "export script failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return tmpdir, output_root


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class ExportStandardDatasetsTest(unittest.TestCase):
    def test_raw_val_source_ids_are_unique_after_backfill(self):
        val_rows = load_jsonl(SOURCE_ROOT / "val" / "samples.jsonl")
        val_ids = [row["sample_id"] for row in val_rows]

        self.assertEqual(len(val_ids), len(set(val_ids)))
        self.assertIn("v00162", val_ids)
        self.assertIn("v00366", val_ids)
        self.assertEqual(val_ids.count("v00162"), 1)
        self.assertEqual(val_ids.count("v00366"), 1)

    def test_exports_track1_with_expected_schema_and_counts(self):
        _, output_root = build_standard_datasets()
        track1_dir = output_root / "track1"

        train_rows = load_jsonl(track1_dir / "train.jsonl")
        val_rows = load_jsonl(track1_dir / "val.jsonl")
        test_rows = load_jsonl(track1_dir / "test_input.jsonl")
        concept_rows = load_jsonl(track1_dir / "concept_pool.jsonl")

        self.assertEqual(len(train_rows), 2985)
        self.assertEqual(len(val_rows), 756)
        self.assertEqual(len(test_rows), 1911)
        self.assertEqual(len(concept_rows), 3433)

        train_row = train_rows[0]
        self.assertEqual(
            set(train_row.keys()),
            {"id", "text", "document_type", "generation_labels", "retrieval_labels"},
        )
        self.assertTrue(train_row["id"])
        self.assertTrue(train_row["text"])
        self.assertTrue(train_row["document_type"])
        self.assertTrue(train_row["generation_labels"])
        self.assertTrue(train_row["retrieval_labels"]["positive_ids"])
        self.assertTrue(train_row["retrieval_labels"]["hard_negative_ids"])

        label = train_row["generation_labels"][0]
        self.assertEqual(set(label.keys()), {"concept_id", "term", "definition"})

        concept_row = concept_rows[0]
        self.assertEqual(set(concept_row.keys()), {"concept_id", "term", "definition"})

        test_row = test_rows[0]
        self.assertEqual(set(test_row.keys()), {"id", "text", "document_type"})

    def test_track1_outputs_exclude_gold_from_test_and_keep_split_concepts_disjoint(self):
        _, output_root = build_standard_datasets()
        track1_dir = output_root / "track1"

        train_rows = load_jsonl(track1_dir / "train.jsonl")
        val_rows = load_jsonl(track1_dir / "val.jsonl")
        test_rows = load_jsonl(track1_dir / "test_input.jsonl")
        concept_rows = load_jsonl(track1_dir / "concept_pool.jsonl")

        train_concepts = {
            label["concept_id"]
            for row in train_rows
            for label in row["generation_labels"]
        }
        val_concepts = {
            label["concept_id"]
            for row in val_rows
            for label in row["generation_labels"]
        }
        concept_pool = {row["concept_id"] for row in concept_rows}

        self.assertFalse(train_concepts & val_concepts)
        self.assertEqual(len(concept_pool), 3433)
        self.assertEqual(len(train_concepts | val_concepts), 2273)

        for row in train_rows[:50] + val_rows[:50]:
            positive_ids = set(row["retrieval_labels"]["positive_ids"])
            hard_negative_ids = set(row["retrieval_labels"]["hard_negative_ids"])
            self.assertTrue(positive_ids)
            self.assertTrue(hard_negative_ids)
            self.assertFalse(positive_ids & hard_negative_ids)
            self.assertTrue(positive_ids.issubset(concept_pool))
            self.assertTrue(hard_negative_ids.issubset(concept_pool))

        for row in test_rows[:50]:
            self.assertNotIn("generation_labels", row)
            self.assertNotIn("retrieval_labels", row)
            serialized = json.dumps(row, ensure_ascii=False)
            self.assertNotIn('"labels"', serialized)
            self.assertNotIn('"provenance"', serialized)

    def test_exports_track2_with_train_val_and_hidden_test_splits(self):
        _, output_root = build_standard_datasets()
        track2_dir = output_root / "track2"

        train_rows = load_jsonl(track2_dir / "train.jsonl")
        val_rows = load_jsonl(track2_dir / "val.jsonl")
        test_rows = load_jsonl(track2_dir / "test_input.jsonl")
        with (track2_dir / "reject_concept_pool.json").open("r", encoding="utf-8") as handle:
            reject_pool = json.load(handle)

        self.assertEqual(len(train_rows), 2985)
        self.assertEqual(len(val_rows), 756)
        self.assertEqual(len(test_rows), 1911)
        self.assertEqual(len(reject_pool), 5342)

        row = val_rows[0]
        self.assertEqual(set(row.keys()), {"id", "prompt", "chosen", "rejected"})
        self.assertIn("Output a JSON array", row["prompt"])
        self.assertTrue(row["chosen"])
        self.assertTrue(row["rejected"])

        chosen = row["chosen"][0]
        rejected = row["rejected"][0]
        self.assertEqual(set(chosen.keys()), {"concept_id", "term", "definition"})
        self.assertEqual(set(rejected.keys()), {"concept_id", "term", "definition"})

        for row in train_rows[:50] + val_rows[:50]:
            chosen_ids = {item["concept_id"] for item in row["chosen"]}
            rejected_ids = {item["concept_id"] for item in row["rejected"]}
            self.assertTrue(chosen_ids)
            self.assertTrue(rejected_ids)
            self.assertFalse(chosen_ids & rejected_ids)

        test_row = test_rows[0]
        self.assertEqual(set(test_row.keys()), {"id", "prompt"})
        self.assertNotIn("chosen", test_row)
        self.assertNotIn("rejected", test_row)
        for row in test_rows[:50]:
            serialized = json.dumps(row, ensure_ascii=False)
            self.assertNotIn('"chosen"', serialized)
            self.assertNotIn('"rejected"', serialized)

    def test_exported_public_ids_are_unique_prefixed_and_aligned_across_tracks(self):
        _, output_root = build_standard_datasets()
        track1_dir = output_root / "track1"
        track2_dir = output_root / "track2"

        track1_train = load_jsonl(track1_dir / "train.jsonl")
        track1_val = load_jsonl(track1_dir / "val.jsonl")
        track1_test = load_jsonl(track1_dir / "test_input.jsonl")
        track2_train = load_jsonl(track2_dir / "train.jsonl")
        track2_val = load_jsonl(track2_dir / "val.jsonl")
        track2_test = load_jsonl(track2_dir / "test_input.jsonl")

        def assert_prefixed_unique(rows, prefix):
            ids = [row["id"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            for row_id in ids[:20]:
                self.assertTrue(row_id.startswith(prefix), msg=row_id)

        assert_prefixed_unique(track1_train, "train_")
        assert_prefixed_unique(track1_val, "val_")
        assert_prefixed_unique(track1_test, "test_")
        assert_prefixed_unique(track2_train, "train_")
        assert_prefixed_unique(track2_val, "val_")
        assert_prefixed_unique(track2_test, "test_")

        track1_all_ids = [row["id"] for row in track1_train + track1_val + track1_test]
        track2_all_ids = [row["id"] for row in track2_train + track2_val + track2_test]
        self.assertEqual(len(track1_all_ids), len(set(track1_all_ids)))
        self.assertEqual(len(track2_all_ids), len(set(track2_all_ids)))

        self.assertEqual([row["id"] for row in track1_train], [row["id"] for row in track2_train])
        self.assertEqual([row["id"] for row in track1_val], [row["id"] for row in track2_val])
        self.assertEqual([row["id"] for row in track1_test], [row["id"] for row in track2_test])


if __name__ == "__main__":
    unittest.main()
