import argparse
import importlib
import json
import inspect
import os
from pathlib import Path

from elsst_baselines.common.gpu import dry_run_summary, generation_hparams_for_preset
from elsst_baselines.common.introspection import filter_supported_kwargs
from elsst_baselines.common.jsonl import write_json
from elsst_baselines.generation.dataset import build_orpo_records, generation_dataset_summary, load_track2_rows
from elsst_baselines.generation.evaluate import evaluate_generation
from elsst_baselines.generation.modeling import load_generation_train_bundle, save_generation_artifacts


def resolve_orpo_classes():
    trl_module = importlib.import_module("trl")
    if hasattr(trl_module, "ORPOConfig") and hasattr(trl_module, "ORPOTrainer"):
        return trl_module.ORPOConfig, trl_module.ORPOTrainer

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    experimental_module = importlib.import_module("trl.experimental.orpo")
    return experimental_module.ORPOConfig, experimental_module.ORPOTrainer


def train_generation_orpo(dataset_root, output_dir, model_name, preset="auto", max_train_samples=None, max_eval_samples=None, max_steps=None, merge_adapter=False):
    from datasets import Dataset

    ORPOConfig, ORPOTrainer = resolve_orpo_classes()

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = generation_hparams_for_preset(preset)
    if max_steps is not None:
        hparams["max_steps"] = max_steps

    train_rows = load_track2_rows(dataset_root / "train.jsonl", max_rows=max_train_samples)
    val_rows = load_track2_rows(dataset_root / "val.jsonl", max_rows=max_eval_samples)
    train_records = build_orpo_records(train_rows)
    val_records = build_orpo_records(val_rows)

    model, tokenizer, processing_class, target_modules = load_generation_train_bundle(model_name, qlora=True)

    config_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": hparams["learning_rate"],
        "num_train_epochs": hparams["num_train_epochs"],
        "per_device_train_batch_size": hparams["per_device_train_batch_size"],
        "per_device_eval_batch_size": hparams["per_device_eval_batch_size"],
        "gradient_accumulation_steps": hparams["gradient_accumulation_steps"],
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "max_prompt_length": hparams["max_prompt_length"],
        "max_completion_length": hparams["max_completion_length"],
        "max_length": hparams["max_prompt_length"] + hparams["max_completion_length"],
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": hparams["gradient_checkpointing"],
        "remove_unused_columns": False,
        "beta": hparams["orpo_beta"],
        "orpo_beta": hparams["orpo_beta"],
    }
    if "max_steps" in hparams:
        config_kwargs["max_steps"] = hparams["max_steps"]
    orpo_config = ORPOConfig(**filter_supported_kwargs(ORPOConfig, config_kwargs))

    trainer_kwargs = {
        "model": model,
        "args": orpo_config,
        "train_dataset": Dataset.from_list(train_records),
        "eval_dataset": Dataset.from_list(val_records),
    }
    trainer_signature = inspect.signature(ORPOTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processing_class
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = ORPOTrainer(**trainer_kwargs)
    trainer.train()

    save_generation_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        model_name=model_name,
        target_modules=target_modules,
        merge_adapter=merge_adapter,
    )
    write_json(output_dir / "train_metadata.json", {"target_modules": target_modules, "training_hparams": hparams})
    metrics = evaluate_generation(
        dataset_root=dataset_root,
        output_dir=output_dir,
        model_name=model_name,
        preset=preset,
        adapter_dir=output_dir / "adapter",
        max_eval_samples=max_eval_samples,
    )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the ORPO concept-discovery baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        summary = generation_dataset_summary(
            args.dataset_root,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
        print(dry_run_summary("generation-train", args.preset, summary))
        return 0

    metrics = train_generation_orpo(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        preset=args.preset,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        merge_adapter=args.merge_adapter,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
