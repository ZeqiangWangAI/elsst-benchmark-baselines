import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elsst_baselines.common import gpu as gpu_module
from elsst_baselines.common.lora import discover_lora_target_modules
from elsst_baselines.common.gpu import retrieval_hparams_for_preset
from elsst_baselines.generation import dataset as generation_dataset
from elsst_baselines.generation.dataset import serialize_concept_list
from elsst_baselines.generation.modeling import build_generation_prompt, normalize_adapter_name
from elsst_baselines.generation.parsing import extract_predicted_terms
from elsst_baselines.generation import scoring as generation_scoring
from elsst_baselines.generation.scoring import bert_score_similarity_matrix, semantic_set_metrics_from_similarity_matrix
from elsst_baselines.generation.train_orpo import resolve_orpo_classes
from elsst_baselines.remote.run import RemoteConfig, build_remote_commands
from elsst_baselines.retrieval.dataset import build_retrieval_triplets, load_concept_pool
from elsst_baselines.retrieval.train import select_best_checkpoint


class RetrievalDatasetBuilderTest(unittest.TestCase):
    def test_build_retrieval_triplets_expands_positive_negative_pairs(self):
        concept_pool = {
            "c1": {"term": "TERM 1", "definition": "Definition one."},
            "c2": {"term": "TERM 2", "definition": "Definition two."},
            "c3": {"term": "TERM 3", "definition": "Definition three."},
            "c4": {"term": "TERM 4", "definition": "Definition four."},
        }
        rows = [
            {
                "id": "q1",
                "text": "A long sociological passage.",
                "generation_labels": [
                    {"concept_id": "c1", "term": "TERM 1", "definition": "Definition one."},
                    {"concept_id": "c2", "term": "TERM 2", "definition": "Definition two."},
                ],
                "retrieval_labels": {
                    "positive_ids": ["c1", "c2"],
                    "hard_negative_ids": ["c3", "c4"],
                },
            }
        ]

        triplets = build_retrieval_triplets(rows, concept_pool)

        self.assertEqual(len(triplets), 4)
        self.assertEqual({row["query_id"] for row in triplets}, {"q1"})
        self.assertEqual({row["positive_id"] for row in triplets}, {"c1", "c2"})
        self.assertEqual({row["negative_id"] for row in triplets}, {"c3", "c4"})
        self.assertTrue(triplets[0]["query"].startswith("Instruct: Given a long social-science passage"))
        self.assertTrue(triplets[0]["positive"].startswith("Concept: "))
        self.assertTrue(triplets[0]["negative"].startswith("Concept: "))

    def test_load_concept_pool_preserves_ids(self):
        path = REPO_ROOT / "track1" / "concept_pool.jsonl"
        concept_pool = load_concept_pool(path)
        self.assertEqual(len(concept_pool), 3433)
        self.assertIn("000e1113-ffda-4088-8278-020b6dc71e20", concept_pool)


class RetrievalPresetTest(unittest.TestCase):
    def test_full_stable_preset_uses_expected_training_defaults(self):
        hparams = retrieval_hparams_for_preset("full_stable")

        self.assertEqual(hparams["max_seq_length"], 1536)
        self.assertEqual(hparams["num_train_epochs"], 3)
        self.assertEqual(hparams["per_device_train_batch_size"], 2)
        self.assertEqual(hparams["gradient_accumulation_steps"], 16)
        self.assertEqual(hparams["save_steps"], 1000)
        self.assertEqual(hparams["eval_steps"], 1000)
        self.assertTrue(hparams["load_best_model_at_end"])
        self.assertEqual(hparams["metric_for_best_model"], "eval_elsst-val_cosine_ndcg@10")


class RetrievalCheckpointSelectionTest(unittest.TestCase):
    def test_prefers_explicit_best_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint-1000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"best_model_checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )

            selected = select_best_checkpoint(
                output_dir=root,
                metric_name="eval_elsst-val_cosine_ndcg@10",
            )

        self.assertEqual(selected, str(checkpoint))

    def test_falls_back_to_best_metric_in_log_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_1000 = root / "checkpoint-1000"
            checkpoint_2000 = root / "checkpoint-2000"
            checkpoint_1000.mkdir(parents=True)
            checkpoint_2000.mkdir(parents=True)
            (checkpoint_2000 / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "log_history": [
                            {"step": 1000, "eval_elsst-val_cosine_ndcg@10": 0.41},
                            {"step": 2000, "eval_elsst-val_cosine_ndcg@10": 0.55},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            selected = select_best_checkpoint(
                output_dir=root,
                metric_name="eval_elsst-val_cosine_ndcg@10",
            )

        self.assertEqual(selected, str(checkpoint_2000))


class LoraTargetModuleTest(unittest.TestCase):
    def test_prefers_expected_projection_suffixes(self):
        names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.down_proj",
            "model.embed_tokens",
            "lm_head",
        ]

        target_modules = discover_lora_target_modules(names)

        self.assertEqual(
            target_modules,
            ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        )

    def test_fallback_excludes_embeddings_and_output_heads(self):
        names = [
            "encoder.block.0.attention.query",
            "encoder.block.0.attention.key",
            "encoder.block.0.attention.value",
            "encoder.block.0.attention.dense",
            "encoder.embed_tokens",
            "score",
            "lm_head",
        ]

        target_modules = discover_lora_target_modules(names)

        self.assertEqual(target_modules, ["dense", "key", "query", "value"])


class GenerationUtilitiesTest(unittest.TestCase):
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
            self.last_enable_thinking = enable_thinking
            user_content = messages[0]["content"]
            if add_generation_prompt:
                return f"<user>{user_content}</user><assistant>"
            assistant_content = messages[1]["content"]
            return f"<user>{user_content}</user><assistant>{assistant_content}"

    def test_generation_stage_presets_use_expected_training_defaults(self):
        sft_hparams_for_preset = getattr(gpu_module, "generation_sft_hparams_for_preset", None)
        dpo_hparams_for_preset = getattr(gpu_module, "generation_dpo_hparams_for_preset", None)

        self.assertIsNotNone(sft_hparams_for_preset)
        self.assertIsNotNone(dpo_hparams_for_preset)

        sft_hparams = sft_hparams_for_preset("24g")
        dpo_hparams = dpo_hparams_for_preset("24g")

        self.assertEqual(sft_hparams["max_prompt_length"], 2048)
        self.assertEqual(sft_hparams["max_completion_length"], 512)
        self.assertEqual(sft_hparams["learning_rate"], 1e-5)
        self.assertEqual(sft_hparams["num_train_epochs"], 2)
        self.assertEqual(sft_hparams["save_steps"], 100)
        self.assertEqual(dpo_hparams["max_prompt_length"], 2048)
        self.assertEqual(dpo_hparams["max_completion_length"], 512)
        self.assertEqual(dpo_hparams["learning_rate"], 5e-6)
        self.assertEqual(dpo_hparams["beta"], 0.1)
        self.assertEqual(dpo_hparams["ld_alpha"], 0.0)
        self.assertTrue(dpo_hparams["precompute_ref_log_probs"])
        self.assertEqual(dpo_hparams["model_adapter_name"], "default")
        self.assertIsNone(dpo_hparams["ref_adapter_name"])

    def test_normalize_adapter_name_avoids_module_reserved_names(self):
        self.assertEqual(normalize_adapter_name("train", fallback="policy"), "policy")
        self.assertEqual(normalize_adapter_name("reference", fallback="policy"), "reference")
        self.assertEqual(normalize_adapter_name("policy-adapter", fallback="policy"), "policy_adapter")

    def test_serialize_concept_list_uses_plain_term_definition_segments(self):
        concept_list = [
            {"concept_id": "c1", "term": "A", "definition": "Alpha"},
            {"concept_id": "c2", "term": "B", "definition": "Beta"},
        ]

        payload = serialize_concept_list(concept_list)

        self.assertEqual(
            payload,
            "A: Alpha; B: Beta;",
        )

    def test_build_generation_prompt_rewrites_legacy_json_instruction(self):
        tokenizer = self.FakeTokenizer()

        prompt = build_generation_prompt(
            'Find concepts.\n\nOutput a JSON array: [{"term": "...", "definition": "..."}]',
            tokenizer,
            disable_thinking=True,
        )

        self.assertIn("term: definition;", prompt)
        self.assertIn("between 1 and 5 concepts", prompt)
        self.assertNotIn("JSON array", prompt)
        self.assertFalse(tokenizer.last_enable_thinking)

    def test_build_sft_records_uses_wrapped_prompt_and_term_definition_response(self):
        build_sft_records = getattr(generation_dataset, "build_sft_records", None)
        self.assertIsNotNone(build_sft_records)

        tokenizer = self.FakeTokenizer()
        rows = [
            {
                "id": "q1",
                "prompt": 'Explain the implicit concepts.\n\nOutput a JSON array: [{"term": "...", "definition": "..."}]',
                "chosen": [
                    {"concept_id": "c1", "term": "TERM 1", "definition": "Definition one."},
                    {"concept_id": "c2", "term": "TERM 2", "definition": "Definition two."},
                ],
                "rejected": [],
            }
        ]

        records = build_sft_records(rows, tokenizer)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "q1")
        self.assertIn("term: definition;", records[0]["prompt"])
        self.assertNotIn("JSON array", records[0]["prompt"])
        self.assertEqual(
            records[0]["response"],
            "TERM 1: Definition one.; TERM 2: Definition two.;",
        )
        self.assertEqual(records[0]["text"], records[0]["prompt"] + records[0]["response"])
        self.assertFalse(tokenizer.last_enable_thinking)

    def test_build_dpo_records_balances_negative_lengths_deterministically(self):
        build_dpo_records = getattr(generation_dataset, "build_dpo_records", None)
        self.assertIsNotNone(build_dpo_records)

        tokenizer = self.FakeTokenizer()
        rows = [
            {
                "id": "q2",
                "prompt": "Find the hidden concepts.",
                "chosen": [
                    {"concept_id": "c1", "term": "TERM 1", "definition": "Definition one."},
                    {"concept_id": "c2", "term": "TERM 2", "definition": "Definition two."},
                    {"concept_id": "c3", "term": "TERM 3", "definition": "Definition three."},
                ],
                "rejected": [
                    {"concept_id": "r1", "term": "NEG 1", "definition": "Negative one."},
                    {"concept_id": "r2", "term": "NEG 2", "definition": "Negative two."},
                    {"concept_id": "r3", "term": "NEG 3", "definition": "Negative three."},
                    {"concept_id": "r4", "term": "NEG 4", "definition": "Negative four."},
                    {"concept_id": "r5", "term": "NEG 5", "definition": "Negative five."},
                ],
            }
        ]

        records = build_dpo_records(rows, tokenizer)
        second_pass = build_dpo_records(rows, tokenizer)

        self.assertEqual(records, second_pass)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["prompt"], "<user>Find the hidden concepts.</user><assistant>")
            self.assertEqual(
                record["chosen"],
                "TERM 1: Definition one.; TERM 2: Definition two.; TERM 3: Definition three.;",
            )
            rejected_terms = extract_predicted_terms(record["rejected"]).terms
            self.assertEqual(len(rejected_terms), 3)
            self.assertTrue(set(rejected_terms).issubset({"NEG 1", "NEG 2", "NEG 3", "NEG 4", "NEG 5"}))

    def test_build_dpo_records_sampling_is_stable_under_public_id_changes(self):
        build_dpo_records = getattr(generation_dataset, "build_dpo_records", None)
        self.assertIsNotNone(build_dpo_records)

        tokenizer = self.FakeTokenizer()
        base_row = {
            "prompt": "Find the hidden concepts.",
            "chosen": [
                {"concept_id": "c1", "term": "TERM 1", "definition": "Definition one."},
                {"concept_id": "c2", "term": "TERM 2", "definition": "Definition two."},
            ],
            "rejected": [
                {"concept_id": "r1", "term": "NEG 1", "definition": "Negative one."},
                {"concept_id": "r2", "term": "NEG 2", "definition": "Negative two."},
                {"concept_id": "r3", "term": "NEG 3", "definition": "Negative three."},
                {"concept_id": "r4", "term": "NEG 4", "definition": "Negative four."},
            ],
        }

        legacy_records = build_dpo_records([{**base_row, "id": "legacy-q2"}], tokenizer)
        public_records = build_dpo_records([{**base_row, "id": "train_t00002"}], tokenizer)

        self.assertEqual(
            [record["rejected"] for record in legacy_records],
            [record["rejected"] for record in public_records],
        )

    def test_extract_predicted_terms_handles_plain_text_and_wrapped_output(self):
        strict = extract_predicted_terms("social stratification: uneven class layering; inequality: unfair resource gaps;")
        wrapped = extract_predicted_terms(
            "Here is the answer:\nsocial capital: resources embedded in networks;"
        )
        capped = extract_predicted_terms("a: one; b: two; c: three; d: four; e: five; f: six;")

        self.assertEqual(strict.terms, ["social stratification", "inequality"])
        self.assertTrue(strict.parsed)
        self.assertEqual(wrapped.terms, ["social capital"])
        self.assertTrue(wrapped.parsed)
        self.assertEqual(capped.terms, ["a", "b", "c", "d", "e"])

    def test_semantic_metrics_match_expected_pairs(self):
        metrics = semantic_set_metrics_from_similarity_matrix(
            similarity_matrix=[
                [0.95, 0.20],
                [0.30, 0.91],
            ],
            tau=0.8,
            predicted_terms=["a", "b"],
            gold_terms=["x", "y"],
        )

        self.assertEqual(metrics["matches"], 2)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_bert_score_similarity_matrix_reuses_cached_scorer(self):
        fake_module = types.ModuleType("bert_score")
        init_calls = []
        score_calls = []

        class FakeScoreValue:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

            def __float__(self):
                return self._value

        class FakeBERTScorer:
            def __init__(self, **kwargs):
                init_calls.append(kwargs)

            def score(self, cands, refs, verbose=False):
                score_calls.append((tuple(cands), tuple(refs), verbose))
                size = len(cands)
                return None, None, [FakeScoreValue(0.9)] * size

        fake_module.BERTScorer = FakeBERTScorer
        with patch.dict(sys.modules, {"bert_score": fake_module}):
            generation_scoring._BERT_SCORER_CACHE.clear()
            first = bert_score_similarity_matrix(["a", "b"], ["x"], model_type="stub")
            second = bert_score_similarity_matrix(["c"], ["y"], model_type="stub")

        self.assertEqual(first, [[0.9], [0.9]])
        self.assertEqual(second, [[0.9]])
        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(score_calls), 2)
        self.assertEqual(init_calls[0]["model_type"], "stub")

    def test_resolve_orpo_classes_falls_back_to_experimental_namespace(self):
        top_level = types.SimpleNamespace()
        experimental = types.SimpleNamespace(ORPOConfig="cfg", ORPOTrainer="trainer")

        with patch("elsst_baselines.generation.train_orpo.importlib.import_module") as import_module:
            import_module.side_effect = [top_level, experimental]
            with patch.dict(sys.modules["elsst_baselines.generation.train_orpo"].os.environ, {}, clear=True):
                orpo_config, orpo_trainer = resolve_orpo_classes()

        self.assertEqual(orpo_config, "cfg")
        self.assertEqual(orpo_trainer, "trainer")


class RemoteRunnerCommandTest(unittest.TestCase):
    def test_remote_command_builder_renders_expected_ssh_and_rsync(self):
        config = RemoteConfig(
            ssh_host="gpu.example.edu",
            ssh_user="alice",
            ssh_port=2222,
            ssh_key_path=Path("/keys/id_ed25519"),
            remote_root=Path("/srv/elsst"),
            local_root=REPO_ROOT,
            hf_home=Path("/srv/hf-cache"),
            wandb_mode="disabled",
        )

        commands = build_remote_commands(config, command_name="retrieval-smoke")

        self.assertIn("rsync", commands.sync)
        self.assertIn("tar -czf -", commands.sync)
        self.assertIn("alice@gpu.example.edu:/srv/elsst/", commands.sync)
        self.assertIn("ssh -i /keys/id_ed25519 -p 2222 alice@gpu.example.edu", commands.setup)
        self.assertIn("python -m elsst_baselines.retrieval.train", commands.run)
        self.assertIn("--max-train-samples 64", commands.run)
        self.assertIn("HF_HOME=/srv/hf-cache", commands.run)
        self.assertNotIn("api_key", commands.run.lower())

    def test_remote_full_command_builder_renders_slurm_submission_and_sync(self):
        config = RemoteConfig(
            ssh_host="surrey-aisurrey",
            ssh_user="zw00924",
            ssh_port=22,
            ssh_key_path=None,
            remote_root=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-retrieval-full-20260321-2000"),
            local_root=REPO_ROOT,
            hf_home=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-retrieval-full-20260321-2000/.cache/huggingface"),
            wandb_mode="disabled",
        )

        commands = build_remote_commands(config, command_name="retrieval-full")

        self.assertIn("sbatch", commands.run)
        self.assertIn("retrieval_full.sbatch", commands.run)
        self.assertIn("base64", commands.run)
        script_text = commands.run
        self.assertIn("sbatch", script_text)
        self.assertIn("artifacts_remote/retrieval", commands.sync_results)
        self.assertIn("scp", commands.sync_results)
        self.assertIn("CONDA_PKGS_DIRS", commands.run)
        self.assertIn('export HOME="$WORKDIR/.bootstrap/home"', commands.run)

    def test_remote_orpo_full_command_builder_renders_slurm_submission_and_sync(self):
        config = RemoteConfig(
            ssh_host="surrey-aisurrey",
            ssh_user="zw00924",
            ssh_port=22,
            ssh_key_path=None,
            remote_root=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-orpo-full-20260323-1200"),
            local_root=REPO_ROOT,
            hf_home=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-orpo-full-20260323-1200/.cache/huggingface"),
            wandb_mode="disabled",
        )

        commands = build_remote_commands(config, command_name="orpo-full")

        self.assertIn("sbatch", commands.run)
        self.assertIn("orpo_full.sbatch", commands.run)
        self.assertIn("base64", commands.run)
        self.assertIn("--dataset-root track2", commands.run)
        self.assertIn("artifacts_remote/generation", commands.sync_results)
        self.assertIn("scp", commands.sync_results)
        self.assertIn("CONDA_PKGS_DIRS", commands.run)
        self.assertIn('export HOME="$WORKDIR/.bootstrap/home"', commands.run)

    def test_remote_sft_smoke_command_builder_renders_expected_generation_launch(self):
        config = RemoteConfig(
            ssh_host="surrey-aisurrey",
            ssh_user="zw00924",
            ssh_port=22,
            ssh_key_path=None,
            remote_root=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-sft-smoke-20260325-1200"),
            local_root=REPO_ROOT,
            hf_home=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-sft-smoke-20260325-1200/.cache/huggingface"),
            wandb_mode="disabled",
        )

        commands = build_remote_commands(config, command_name="sft-smoke")

        self.assertIn("sbatch", commands.run)
        self.assertIn("train_sft", commands.run)
        self.assertIn("--dataset-root track2", commands.run)
        self.assertIn("--resume-from-checkpoint", commands.run)
        self.assertIn("fs_weka", commands.run)
        self.assertIn("artifacts_remote/generation", commands.sync_results)
        self.assertIn("scp", commands.sync_results)
        self.assertIn("CONDA_PKGS_DIRS", commands.run)
        self.assertIn('export HOME="$WORKDIR/.bootstrap/home"', commands.run)
        self.assertIn("#SBATCH --mem=64000M", commands.run)
        self.assertIn("download.pytorch.org/whl/cu124", commands.run)
        self.assertIn("torch==2.6.0", commands.run)
        self.assertIn("pip install -e . --no-deps", commands.run)
        self.assertIn("torch.cuda.is_available()", commands.run)

    def test_remote_dpo_full_command_builder_renders_expected_generation_launch(self):
        config = RemoteConfig(
            ssh_host="surrey-aisurrey",
            ssh_user="zw00924",
            ssh_port=22,
            ssh_key_path=None,
            remote_root=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-dpo-full-20260325-1200"),
            local_root=REPO_ROOT,
            hf_home=Path("/mnt/fast/nobackup/users/zw00924/codex-surrey-runs/elsst-dpo-full-20260325-1200/.cache/huggingface"),
            wandb_mode="disabled",
        )

        commands = build_remote_commands(config, command_name="dpo-full")

        self.assertIn("sbatch", commands.run)
        self.assertIn("train_dpo", commands.run)
        self.assertIn("--dataset-root track2", commands.run)
        self.assertIn("--sft-adapter-dir", commands.run)
        self.assertIn("--resume-from-checkpoint", commands.run)
        self.assertIn("fs_weka", commands.run)
        self.assertIn("artifacts_remote/generation", commands.sync_results)
        self.assertIn("scp", commands.sync_results)
        self.assertIn("CONDA_PKGS_DIRS", commands.run)
        self.assertIn('export HOME="$WORKDIR/.bootstrap/home"', commands.run)
        self.assertIn("#SBATCH --mem=64000M", commands.run)


if __name__ == "__main__":
    unittest.main()
