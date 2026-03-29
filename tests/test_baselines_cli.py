import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = str(REPO_ROOT / "src")


def run_module(module_name, args):
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module_name, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


class BaselineCliDryRunTest(unittest.TestCase):
    def test_retrieval_train_and_evaluate_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_result = run_module(
                "elsst_baselines.retrieval.train",
                [
                    "--dataset-root",
                    str(REPO_ROOT / "track1"),
                    "--output-dir",
                    tmpdir,
                    "--model-name",
                    "Qwen/Qwen3-Embedding-0.6B",
                    "--preset",
                    "full_stable",
                    "--dry-run",
                ],
            )
            eval_result = run_module(
                "elsst_baselines.retrieval.evaluate",
                [
                    "--dataset-root",
                    str(REPO_ROOT / "track1"),
                    "--output-dir",
                    tmpdir,
                    "--model-name",
                    "Qwen/Qwen3-Embedding-0.6B",
                    "--dry-run",
                ],
            )

        self.assertEqual(train_result.returncode, 0, msg=train_result.stderr)
        self.assertEqual(eval_result.returncode, 0, msg=eval_result.stderr)
        self.assertIn('"mode": "dry-run"', train_result.stdout)
        self.assertIn('"mode": "dry-run"', eval_result.stdout)
        self.assertIn('"preset": "full_stable"', train_result.stdout)

    def test_retrieval_infer_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            infer_result = run_module(
                "elsst_baselines.retrieval.infer",
                [
                    "--query-file",
                    str(REPO_ROOT / "track1" / "test_input.jsonl"),
                    "--concept-pool",
                    str(REPO_ROOT / "track1" / "concept_pool.jsonl"),
                    "--output-path",
                    str(Path(tmpdir) / "predictions.jsonl"),
                    "--model-name",
                    "Qwen/Qwen3-Embedding-0.6B",
                    "--dry-run",
                ],
            )

        self.assertEqual(infer_result.returncode, 0, msg=infer_result.stderr)
        self.assertIn('"mode": "dry-run"', infer_result.stdout)

    def test_generation_train_and_evaluate_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_result = run_module(
                "elsst_baselines.generation.train_orpo",
                [
                    "--dataset-root",
                    str(REPO_ROOT / "track2"),
                    "--output-dir",
                    tmpdir,
                    "--model-name",
                    "Qwen/Qwen3.5-4B",
                    "--dry-run",
                ],
            )
            eval_result = run_module(
                "elsst_baselines.generation.evaluate",
                [
                    "--dataset-root",
                    str(REPO_ROOT / "track2"),
                    "--output-dir",
                    tmpdir,
                    "--model-name",
                    "Qwen/Qwen3.5-4B",
                    "--dry-run",
                ],
            )

        self.assertEqual(train_result.returncode, 0, msg=train_result.stderr)
        self.assertEqual(eval_result.returncode, 0, msg=eval_result.stderr)
        self.assertIn('"mode": "dry-run"', train_result.stdout)
        self.assertIn('"mode": "dry-run"', eval_result.stdout)

    def test_generation_sft_and_dpo_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_adapter_dir = Path(tmpdir) / "sft_adapter"
            sft_adapter_dir.mkdir(parents=True)
            sft_result = run_module(
                "elsst_baselines.generation.train_sft",
                [
                    "--dataset-root",
                    str(REPO_ROOT / "track2"),
                    "--output-dir",
                    str(Path(tmpdir) / "sft_output"),
                    "--model-name",
                    "Qwen/Qwen3.5-4B",
                    "--dry-run",
                ],
            )
            dpo_result = run_module(
                "elsst_baselines.generation.train_dpo",
                [
                    "--dataset-root",
                    str(REPO_ROOT / "track2"),
                    "--output-dir",
                    str(Path(tmpdir) / "dpo_output"),
                    "--model-name",
                    "Qwen/Qwen3.5-4B",
                    "--sft-adapter-dir",
                    str(sft_adapter_dir),
                    "--dry-run",
                ],
            )

        self.assertEqual(sft_result.returncode, 0, msg=sft_result.stderr)
        self.assertEqual(dpo_result.returncode, 0, msg=dpo_result.stderr)
        self.assertIn('"mode": "dry-run"', sft_result.stdout)
        self.assertIn('"mode": "dry-run"', dpo_result.stdout)

    def test_remote_runner_requires_ssh_env_or_args(self):
        result = run_module(
            "elsst_baselines.remote.run",
            [
                "sync",
                "--dry-run",
            ],
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SSH_HOST", result.stderr)

    def test_remote_orpo_full_dry_run_uses_slurm_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_module(
                "elsst_baselines.remote.run",
                [
                    "orpo-full",
                    "--ssh-host",
                    "surrey-aisurrey",
                    "--ssh-user",
                    "zw00924",
                    "--remote-root",
                    "/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/test-orpo-full",
                    "--local-sync-root",
                    tmpdir,
                    "--dry-run",
                ],
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"command": "orpo-full"', result.stdout)
        self.assertIn("sbatch", result.stdout)
        self.assertIn("sync_results", result.stdout)

    def test_remote_sft_full_dry_run_uses_slurm_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_module(
                "elsst_baselines.remote.run",
                [
                    "sft-full",
                    "--ssh-host",
                    "surrey-aisurrey",
                    "--ssh-user",
                    "zw00924",
                    "--remote-root",
                    "/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/test-sft-full",
                    "--local-sync-root",
                    tmpdir,
                    "--dry-run",
                ],
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"command": "sft-full"', result.stdout)
        self.assertIn("sbatch", result.stdout)
        self.assertIn("sync_results", result.stdout)

    def test_remote_dpo_full_dry_run_uses_slurm_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_module(
                "elsst_baselines.remote.run",
                [
                    "dpo-full",
                    "--ssh-host",
                    "surrey-aisurrey",
                    "--ssh-user",
                    "zw00924",
                    "--remote-root",
                    "/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/test-dpo-full",
                    "--local-sync-root",
                    tmpdir,
                    "--dry-run",
                ],
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"command": "dpo-full"', result.stdout)
        self.assertIn("sbatch", result.stdout)
        self.assertIn("sync_results", result.stdout)


if __name__ == "__main__":
    unittest.main()
