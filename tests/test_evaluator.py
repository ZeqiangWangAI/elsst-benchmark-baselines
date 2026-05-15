import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Track1EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.reference_rows = [
            {
                "id": "val_a",
                "retrieval_labels": {"positive_ids": ["c1"], "hard_negative_ids": []},
            },
            {
                "id": "val_b",
                "retrieval_labels": {"positive_ids": ["c2", "c3"], "hard_negative_ids": []},
            },
        ]
        self.concept_pool = {
            f"c{index}": {"term": f"Concept {index}", "definition": "Definition."}
            for index in range(1, 13)
        }

    def test_scores_valid_ranking_submission(self):
        from elsst_baselines.evaluator import track1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "val_a",
                        "ranked_ids": ["c1", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12"],
                    },
                    {
                        "id": "val_b",
                        "ranked_ids": ["c4", "c2", "c5", "c3", "c6", "c7", "c8", "c9", "c10", "c11"],
                    },
                ],
            )

            result = track1.score_submission(
                submission_path=path,
                reference_rows=self.reference_rows,
                concept_pool=self.concept_pool,
                split="val",
            )

        self.assertTrue(result.valid)
        self.assertEqual(result.primary_metric, "NDCG@10")
        self.assertAlmostEqual(result.metrics["MRR"], 0.75)
        self.assertAlmostEqual(result.metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(result.metrics["Recall@10"], 1.0)
        self.assertGreater(result.metrics["NDCG@10"], 0.8)
        self.assertEqual(result.diagnostics["row_count"], 2)

    def test_scores_gold_ranking_submission_at_full_metrics(self):
        from elsst_baselines.evaluator import track1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gold.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "val_a",
                        "ranked_ids": ["c1", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12"],
                    },
                    {
                        "id": "val_b",
                        "ranked_ids": ["c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"],
                    },
                ],
            )

            result = track1.score_submission(
                submission_path=path,
                reference_rows=self.reference_rows,
                concept_pool=self.concept_pool,
                split="val",
            )

        self.assertAlmostEqual(result.metrics["MRR"], 1.0)
        self.assertAlmostEqual(result.metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(result.metrics["Recall@10"], 1.0)
        self.assertAlmostEqual(result.metrics["NDCG@10"], 1.0)

    def test_rejects_missing_extra_duplicate_and_unknown_concepts(self):
        from elsst_baselines.evaluator import track1
        from elsst_baselines.evaluator.validation import SubmissionValidationError

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "val_a", "ranked_ids": ["c1", "c1", "unknown"] + [f"c{i}" for i in range(4, 12)]},
                    {"id": "val_extra", "ranked_ids": [f"c{i}" for i in range(1, 11)]},
                ],
            )

            with self.assertRaises(SubmissionValidationError) as ctx:
                track1.score_submission(
                    submission_path=path,
                    reference_rows=self.reference_rows,
                    concept_pool=self.concept_pool,
                    split="val",
                )

        message = str(ctx.exception)
        self.assertIn("missing ids: val_b", message)
        self.assertIn("extra ids: val_extra", message)
        self.assertIn("duplicate ranked_ids", message)
        self.assertIn("unknown concept ids", message)

    def test_rejects_short_rankings(self):
        from elsst_baselines.evaluator import track1
        from elsst_baselines.evaluator.validation import SubmissionValidationError

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "short.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "val_a", "ranked_ids": [f"c{i}" for i in range(1, 10)]},
                    {"id": "val_b", "ranked_ids": [f"c{i}" for i in range(1, 11)]},
                ],
            )

            with self.assertRaises(SubmissionValidationError) as ctx:
                track1.score_submission(
                    submission_path=path,
                    reference_rows=self.reference_rows,
                    concept_pool=self.concept_pool,
                    split="val",
                )

        self.assertIn("at least 10", str(ctx.exception))


class Track2EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.reference_rows = [
            {
                "id": "val_a",
                "chosen": [
                    {"term": "Social capital", "definition": "Network resources."},
                    {"term": "Inequality", "definition": "Uneven distribution."},
                ],
            },
            {
                "id": "val_b",
                "chosen": [{"term": "Stigma", "definition": "Social devaluation."}],
            },
        ]

    def test_scores_predicted_terms_with_injected_similarity(self):
        from elsst_baselines.evaluator import track2

        def similarity(predicted_terms, gold_terms):
            return [
                [1.0 if predicted.casefold() == gold.casefold() else 0.1 for gold in gold_terms]
                for predicted in predicted_terms
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "val_a", "predicted_terms": ["Social capital", "Inequality"]},
                    {"id": "val_b", "predicted_terms": ["Stigma"]},
                ],
            )

            result = track2.score_submission(
                submission_path=path,
                reference_rows=self.reference_rows,
                split="val",
                similarity_fn=similarity,
                tau=0.85,
            )

        self.assertTrue(result.valid)
        self.assertEqual(result.primary_metric, "semantic_f1")
        self.assertAlmostEqual(result.metrics["exact_f1"], 1.0)
        self.assertAlmostEqual(result.metrics["semantic_f1"], 1.0)
        self.assertAlmostEqual(result.metrics["parse_rate"], 1.0)
        self.assertAlmostEqual(result.metrics["valid_prediction_rate"], 1.0)

    def test_exact_gold_terms_do_not_require_similarity_model(self):
        from elsst_baselines.evaluator import track2

        def fail_if_called(predicted_terms, gold_terms):
            raise AssertionError("exact gold predictions should not call similarity model")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "val_a", "predicted_terms": ["Inequality", "Social capital"]},
                    {"id": "val_b", "predicted_terms": ["Stigma"]},
                ],
            )

            result = track2.score_submission(
                submission_path=path,
                reference_rows=self.reference_rows,
                split="val",
                similarity_fn=fail_if_called,
                tau=0.85,
            )

        self.assertAlmostEqual(result.metrics["exact_f1"], 1.0)
        self.assertAlmostEqual(result.metrics["semantic_f1"], 1.0)

    def test_raw_text_fallback_and_empty_predictions_score_zero(self):
        from elsst_baselines.evaluator import track2

        def similarity(predicted_terms, gold_terms):
            return [
                [1.0 if predicted.casefold() == gold.casefold() else 0.1 for gold in gold_terms]
                for predicted in predicted_terms
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "val_a", "raw_text": "Social capital: resources in networks; Inequality: uneven outcomes;"},
                    {"id": "val_b", "predicted_terms": []},
                ],
            )

            result = track2.score_submission(
                submission_path=path,
                reference_rows=self.reference_rows,
                split="val",
                similarity_fn=similarity,
                tau=0.85,
            )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.metrics["semantic_recall"], 0.5)
        self.assertAlmostEqual(result.metrics["valid_prediction_rate"], 0.5)
        self.assertEqual(result.diagnostics["empty_prediction_count"], 1)
        self.assertEqual(result.diagnostics["raw_text_fallback_count"], 1)

    def test_rejects_duplicate_submission_rows(self):
        from elsst_baselines.evaluator import track2
        from elsst_baselines.evaluator.validation import SubmissionValidationError

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "val_a", "predicted_terms": ["Social capital"]},
                    {"id": "val_a", "predicted_terms": ["Inequality"]},
                ],
            )

            with self.assertRaises(SubmissionValidationError) as ctx:
                track2.score_submission(
                    submission_path=path,
                    reference_rows=self.reference_rows,
                    split="val",
                    similarity_fn=lambda predicted, gold: [],
                )

        self.assertIn("duplicate ids: val_a", str(ctx.exception))


class LeaderboardStoreTest(unittest.TestCase):
    def test_rate_limits_per_user_track_and_returns_best_entries(self):
        from elsst_baselines.evaluator.leaderboard import LeaderboardStore, RateLimitError

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LeaderboardStore(Path(tmpdir) / "leaderboard.sqlite")
            now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
            store.record_submission("alice", "track1", "m1", "NDCG@10", {"NDCG@10": 0.2}, "hash1", now=now)
            store.record_submission("alice", "track1", "m1", "NDCG@10", {"NDCG@10": 0.4}, "hash2", now=now + timedelta(hours=1))
            store.record_submission("alice", "track1", "m2", "NDCG@10", {"NDCG@10": 0.3}, "hash3", now=now + timedelta(hours=2))

            with self.assertRaises(RateLimitError):
                store.record_submission(
                    "alice",
                    "track1",
                    "m3",
                    "NDCG@10",
                    {"NDCG@10": 0.5},
                    "hash4",
                    now=now + timedelta(hours=3),
                )

            entries = store.top_entries(track="track1")

        self.assertEqual([entry["model_name"] for entry in entries], ["m1", "m2"])
        self.assertEqual([entry["submission_hash"] for entry in entries], ["hash2", "hash3"])

    def test_can_disable_rate_limits_for_public_anonymous_submissions(self):
        from elsst_baselines.evaluator.leaderboard import LeaderboardStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LeaderboardStore(Path(tmpdir) / "leaderboard.sqlite", daily_limit=None)
            now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
            for index in range(5):
                store.record_submission(
                    "anonymous",
                    "track1",
                    "m0" if index < 2 else f"m{index}",
                    "NDCG@10",
                    {"NDCG@10": index / 10},
                    f"hash{index}",
                    now=now + timedelta(minutes=index),
                )

            entries = store.top_entries(track="track1")

        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0]["username"], "anonymous")
        self.assertEqual(entries[0]["model_name"], "m4")
        self.assertEqual([entry for entry in entries if entry["model_name"] == "m0"][0]["submission_hash"], "hash1")


class SpaceAppImportTest(unittest.TestCase):
    def test_app_module_imports_without_launching(self):
        spec = importlib.util.spec_from_file_location("elsst_space_app", REPO_ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(callable(module.build_demo))
        self.assertTrue(callable(module.score_val_file))

    def test_validation_state_is_session_local(self):
        spec = importlib.util.spec_from_file_location("elsst_space_app", REPO_ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        state = module._validation_state()

        self.assertFalse(hasattr(state, "storage_key"))
        self.assertEqual(state.value, {"validated": False, "track": "track1"})

    def test_anonymous_test_submit_succeeds_without_request(self):
        spec = importlib.util.spec_from_file_location("elsst_space_app", REPO_ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from elsst_baselines.evaluator.leaderboard import LeaderboardStore
        from elsst_baselines.evaluator.result import EvaluationResult

        with tempfile.TemporaryDirectory() as tmpdir:
            submission = Path(tmpdir) / "submission.jsonl"
            submission.write_text("{}\n", encoding="utf-8")
            store = LeaderboardStore(Path(tmpdir) / "leaderboard.sqlite", daily_limit=None)

            with patch.object(module, "_leaderboard_store", return_value=store), patch.object(
                module,
                "_evaluate",
                return_value=EvaluationResult(
                    track="track1",
                    split="test",
                    primary_metric="NDCG@10",
                    metrics={"NDCG@10": 1.0},
                    diagnostics={"row_count": 1},
                ),
            ):
                result_markdown, rows = module._submit_test_ui(
                    "track1",
                    "anonymous-model",
                    str(submission),
                    {"validated": True, "track": "track1"},
                )

        self.assertIn("Track1 Retrieval test score", result_markdown)
        self.assertEqual(rows[0][1], "anonymous")
        self.assertEqual(rows[0][2], "anonymous-model")

    def test_anonymous_test_submit_still_requires_val_gate(self):
        spec = importlib.util.spec_from_file_location("elsst_space_app", REPO_ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with self.assertRaises(Exception) as ctx:
            module._submit_test_ui("track1", "model", "missing.jsonl", {"validated": False, "track": "track1"})

        self.assertIn("Validate the selected track on val", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
