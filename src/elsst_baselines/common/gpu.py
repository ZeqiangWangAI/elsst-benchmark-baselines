import json
import shutil
import subprocess


def precision_flags():
    try:
        import torch
    except ImportError:
        return {"bf16": False, "fp16": False}

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def detect_gpu_memory_mb():
    if shutil.which("nvidia-smi") is None:
        return None

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    return max(values) if values else None


def resolve_preset(explicit_preset="auto"):
    if explicit_preset != "auto":
        return explicit_preset

    memory_mb = detect_gpu_memory_mb()
    if memory_mb is None:
        return "24g"
    if memory_mb >= 46_000:
        return "48g"
    return "24g"


def retrieval_hparams_for_preset(preset):
    resolved = resolve_preset(preset)
    if resolved == "full_stable":
        return {
            "max_seq_length": 1536,
            "num_train_epochs": 3,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "save_steps": 1000,
            "eval_steps": 1000,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_elsst-val_cosine_ndcg@10",
            "greater_is_better": True,
            "seed": 42,
            "logging_steps": 25,
        }

    defaults = {
        "max_seq_length": 1536,
        "num_train_epochs": 1,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 4 if resolved == "48g" else 2,
        "per_device_eval_batch_size": 4 if resolved == "48g" else 2,
    }
    if resolved == "smoke":
        defaults.update(
            {
                "max_seq_length": 512,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 5,
            }
        )
    return defaults


def generation_hparams_for_preset(preset):
    resolved = resolve_preset(preset)
    defaults = {
        "max_prompt_length": 2048,
        "max_completion_length": 512,
        "num_train_epochs": 1,
        "learning_rate": 5e-6,
        "orpo_beta": 0.1,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16 if resolved == "48g" else 32,
    }
    if resolved == "smoke":
        defaults.update(
            {
                "max_prompt_length": 2048,
                "max_completion_length": 512,
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
            }
        )
    return defaults


def generation_sft_hparams_for_preset(preset):
    resolved = resolve_preset(preset)
    defaults = {
        "max_prompt_length": 2048,
        "max_completion_length": 512,
        "num_train_epochs": 2,
        "learning_rate": 1e-5,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "logging_steps": 10,
        "seed": 42,
    }
    if resolved == "smoke":
        defaults.update(
            {
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
            }
        )
    return defaults


def generation_dpo_hparams_for_preset(preset):
    resolved = resolve_preset(preset)
    defaults = {
        "max_prompt_length": 2048,
        "max_completion_length": 512,
        "num_train_epochs": 1,
        "learning_rate": 5e-6,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "logging_steps": 10,
        "seed": 42,
        "beta": 0.1,
        "ld_alpha": 0.0,
        "loss_type": "sigmoid",
        "precompute_ref_log_probs": True,
        "model_adapter_name": "default",
        "ref_adapter_name": None,
    }
    if resolved == "smoke":
        defaults.update(
            {
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
            }
        )
    return defaults


def dry_run_summary(task_name, preset, payload):
    summary = {"task": task_name, "mode": "dry-run", "preset": resolve_preset(preset)}
    summary.update(payload)
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
