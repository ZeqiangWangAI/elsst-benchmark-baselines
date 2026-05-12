import json
import subprocess
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_full_internal_datasets.py"
SOURCE_ROOT = REPO_ROOT / "dataset"


@lru_cache(maxsize=1)
def build_full_internal_datasets():
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
            "full export script failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return tmpdir, output_root


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class FullInternalExportsTest(unittest.TestCase):
    def test_exports_track1_full_with_labeled_test(self):
        _, output_root = build_full_internal_datasets()
        track1_full_dir = output_root / "track1_full"

        train_rows = load_jsonl(track1_full_dir / "train.jsonl")
        val_rows = load_jsonl(track1_full_dir / "val.jsonl")
        test_rows = load_jsonl(track1_full_dir / "test.jsonl")
        concept_rows = load_jsonl(track1_full_dir / "concept_pool.jsonl")

        self.assertEqual(len(train_rows), 2985)
        self.assertEqual(len(val_rows), 756)
        self.assertEqual(len(test_rows), 1911)
        self.assertEqual(len(concept_rows), 3433)

        test_row = test_rows[0]
        self.assertEqual(
            set(test_row.keys()),
            {"id", "text", "document_type", "generation_labels", "retrieval_labels"},
        )
        self.assertTrue(test_row["generation_labels"])
        self.assertTrue(test_row["retrieval_labels"]["positive_ids"])
        self.assertTrue(test_row["retrieval_labels"]["hard_negative_ids"])
        self.assertTrue(test_row["id"].startswith("test_"))

    def test_exports_track2_full_with_labeled_test(self):
        _, output_root = build_full_internal_datasets()
        track2_full_dir = output_root / "track2_full"

        train_rows = load_jsonl(track2_full_dir / "train.jsonl")
        val_rows = load_jsonl(track2_full_dir / "val.jsonl")
        test_rows = load_jsonl(track2_full_dir / "test.jsonl")
        with (track2_full_dir / "reject_concept_pool.json").open("r", encoding="utf-8") as handle:
            reject_pool = json.load(handle)

        self.assertEqual(len(train_rows), 2985)
        self.assertEqual(len(val_rows), 756)
        self.assertEqual(len(test_rows), 1911)
        self.assertEqual(len(reject_pool), 5342)

        test_row = test_rows[0]
        self.assertEqual(set(test_row.keys()), {"id", "prompt", "chosen", "rejected"})
        self.assertTrue(test_row["chosen"])
        self.assertTrue(test_row["rejected"])
        self.assertTrue(test_row["id"].startswith("test_"))

        chosen_ids = {item["concept_id"] for item in test_row["chosen"]}
        rejected_ids = {item["concept_id"] for item in test_row["rejected"]}
        self.assertFalse(chosen_ids & rejected_ids)

    def test_full_and_public_test_ids_align(self):
        _, output_root = build_full_internal_datasets()

        track1_public_test = load_jsonl(REPO_ROOT / "track1" / "test_input.jsonl")
        track2_public_test = load_jsonl(REPO_ROOT / "track2" / "test_input.jsonl")
        track1_full_test = load_jsonl(output_root / "track1_full" / "test.jsonl")
        track2_full_test = load_jsonl(output_root / "track2_full" / "test.jsonl")

        self.assertEqual(
            [row["id"] for row in track1_public_test],
            [row["id"] for row in track1_full_test],
        )
        self.assertEqual(
            [row["id"] for row in track2_public_test],
            [row["id"] for row in track2_full_test],
        )


if __name__ == "__main__":
    unittest.main()
