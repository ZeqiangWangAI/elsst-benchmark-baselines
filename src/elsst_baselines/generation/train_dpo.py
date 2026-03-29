import argparse
import inspect
import json
from pathlib import Path

from elsst_baselines.common.gpu import dry_run_summary, generation_dpo_hparams_for_preset
from elsst_baselines.common.introspection import filter_supported_kwargs
from elsst_baselines.common.jsonl import write_json
from elsst_baselines.generation.dataset import build_dpo_records, generation_dataset_summary, load_track2_rows
from elsst_baselines.generation.evaluate import evaluate_generation
from elsst_baselines.generation.modeling import (
    load_generation_train_bundle,
    normalize_adapter_name,
    save_generation_artifacts,
)
from elsst_baselines.generation.training import (
    apply_max_steps_overrides,
    build_generation_training_arguments,
    resolve_resume_checkpoint,
)


def train_generation_dpo(
    dataset_root,
    output_dir,
    model_name,
    sft_adapter_dir,
    preset="auto",
    max_train_samples=None,
    max_eval_samples=None,
    max_steps=None,
    merge_adapter=False,
    resume_from_checkpoint=None,
):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = apply_max_steps_overrides(generation_dpo_hparams_for_preset(preset), max_steps)
    hparams["max_length"] = hparams["max_prompt_length"] + hparams["max_completion_length"]
    hparams["model_adapter_name"] = normalize_adapter_name(hparams.get("model_adapter_name"), fallback="default")
    if hparams.get("ref_adapter_name"):
        hparams["ref_adapter_name"] = normalize_adapter_name(hparams.get("ref_adapter_name"), fallback="reference")
        if hparams["ref_adapter_name"] == hparams["model_adapter_name"]:
            hparams["ref_adapter_name"] = None

    train_rows = load_track2_rows(dataset_root / "train.jsonl", max_rows=max_train_samples)
    val_rows = load_track2_rows(dataset_root / "val.jsonl", max_rows=max_eval_samples)

    model, tokenizer, processing_class, target_modules = load_generation_train_bundle(
        model_name,
        qlora=True,
        adapter_dir=sft_adapter_dir,
        adapter_name=hparams["model_adapter_name"],
        ref_adapter_name=hparams["ref_adapter_name"],
    )
    train_records = build_dpo_records(train_rows, tokenizer)
    val_records = build_dpo_records(val_rows, tokenizer)

    training_args = build_generation_training_arguments(DPOConfig, output_dir, hparams)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": Dataset.from_list(
            [{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]} for row in train_records]
        ),
        "eval_dataset": Dataset.from_list(
            [{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]} for row in val_records]
        ),
    }
    trainer_signature = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processing_class
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**filter_supported_kwargs(DPOTrainer.__init__, trainer_kwargs))
    trainer.train(
        **filter_supported_kwargs(
            trainer.train,
            {"resume_from_checkpoint": resolve_resume_checkpoint(output_dir, resume_from_checkpoint)},
        )
    )

    save_generation_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        model_name=model_name,
        target_modules=target_modules,
        merge_adapter=merge_adapter,
        adapter_name=hparams["model_adapter_name"],
    )
    write_json(
        output_dir / "train_metadata.json",
        {
            "stage": "dpo",
            "source_adapter_dir": str(sft_adapter_dir),
            "target_modules": target_modules,
            "training_hparams": hparams,
        },
    )
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
    parser = argparse.ArgumentParser(description="Train the DPO concept-discovery baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sft-adapter-dir", required=True)
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
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
        print(dry_run_summary("generation-dpo-train", args.preset, summary))
        return 0

    metrics = train_generation_dpo(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        sft_adapter_dir=args.sft_adapter_dir,
        preset=args.preset,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        merge_adapter=args.merge_adapter,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
