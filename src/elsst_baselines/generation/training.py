import inspect
from pathlib import Path

from elsst_baselines.common.gpu import precision_flags
from elsst_baselines.common.introspection import filter_supported_kwargs


def resolve_resume_checkpoint(output_dir, resume_from_checkpoint):
    if not resume_from_checkpoint:
        return None

    output_dir = Path(output_dir)
    if resume_from_checkpoint != "auto":
        return str(Path(resume_from_checkpoint))

    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def apply_max_steps_overrides(hparams, max_steps):
    updated = dict(hparams)
    if max_steps is None:
        return updated

    updated["max_steps"] = max_steps
    if "save_steps" in updated:
        updated["save_steps"] = min(updated["save_steps"], max_steps)
    if "eval_steps" in updated:
        updated["eval_steps"] = min(updated["eval_steps"], max_steps)
    updated["logging_steps"] = max(1, min(updated.get("logging_steps", 10), max_steps))
    return updated


def build_generation_training_arguments(training_args_cls, output_dir, hparams):
    strategy_key = None
    signature = inspect.signature(training_args_cls)
    if "eval_strategy" in signature.parameters:
        strategy_key = "eval_strategy"
    elif "evaluation_strategy" in signature.parameters:
        strategy_key = "evaluation_strategy"

    candidate_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": hparams["num_train_epochs"],
        "per_device_train_batch_size": hparams["per_device_train_batch_size"],
        "per_device_eval_batch_size": hparams["per_device_eval_batch_size"],
        "learning_rate": hparams["learning_rate"],
        "gradient_accumulation_steps": hparams["gradient_accumulation_steps"],
        "logging_steps": hparams.get("logging_steps", 10),
        "save_strategy": hparams.get("save_strategy", "epoch"),
        "save_steps": hparams.get("save_steps"),
        "save_total_limit": hparams.get("save_total_limit"),
        "gradient_checkpointing": hparams["gradient_checkpointing"],
        "remove_unused_columns": False,
        "seed": hparams.get("seed"),
        "bf16": precision_flags()["bf16"],
        "fp16": precision_flags()["fp16"],
        "report_to": "none",
    }
    if strategy_key and hparams.get("eval_strategy"):
        candidate_kwargs[strategy_key] = hparams["eval_strategy"]
        candidate_kwargs["eval_steps"] = hparams.get("eval_steps")
    if "max_steps" in hparams:
        candidate_kwargs["max_steps"] = hparams["max_steps"]

    for key in (
        "beta",
        "ld_alpha",
        "loss_type",
        "precompute_ref_log_probs",
        "max_length",
        "max_prompt_length",
        "max_completion_length",
        "model_adapter_name",
        "ref_adapter_name",
    ):
        if key in hparams:
            candidate_kwargs[key] = hparams[key]

    filtered_kwargs = filter_supported_kwargs(training_args_cls, candidate_kwargs)
    return training_args_cls(**filtered_kwargs)
