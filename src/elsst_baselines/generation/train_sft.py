import argparse
import json
from pathlib import Path

from elsst_baselines.common.gpu import dry_run_summary, generation_sft_hparams_for_preset
from elsst_baselines.common.jsonl import write_json
from elsst_baselines.generation.dataset import build_sft_records, generation_dataset_summary, load_track2_rows
from elsst_baselines.generation.evaluate import evaluate_generation
from elsst_baselines.generation.modeling import load_generation_train_bundle, save_generation_artifacts
from elsst_baselines.generation.training import (
    apply_max_steps_overrides,
    build_generation_training_arguments,
    resolve_resume_checkpoint,
)


def _tokenize_sft_record(record, tokenizer, max_prompt_length, max_completion_length):
    prompt_ids = tokenizer(
        record["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]
    response_ids = tokenizer(
        record["response"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_completion_length,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        if len(response_ids) >= max_completion_length:
            response_ids = response_ids[: max_completion_length - 1] + [tokenizer.eos_token_id]
        elif not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def train_generation_sft(
    dataset_root,
    output_dir,
    model_name,
    preset="auto",
    max_train_samples=None,
    max_eval_samples=None,
    max_steps=None,
    merge_adapter=False,
    resume_from_checkpoint=None,
):
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = apply_max_steps_overrides(generation_sft_hparams_for_preset(preset), max_steps)

    train_rows = load_track2_rows(dataset_root / "train.jsonl", max_rows=max_train_samples)
    val_rows = load_track2_rows(dataset_root / "val.jsonl", max_rows=max_eval_samples)

    model, tokenizer, _, target_modules = load_generation_train_bundle(model_name, qlora=True)
    train_records = build_sft_records(train_rows, tokenizer)
    val_records = build_sft_records(val_rows, tokenizer)

    train_dataset = Dataset.from_list(
        [
            _tokenize_sft_record(record, tokenizer, hparams["max_prompt_length"], hparams["max_completion_length"])
            for record in train_records
        ]
    )
    eval_dataset = Dataset.from_list(
        [
            _tokenize_sft_record(record, tokenizer, hparams["max_prompt_length"], hparams["max_completion_length"])
            for record in val_records
        ]
    )

    training_args = build_generation_training_arguments(TrainingArguments, output_dir, hparams)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.train(
        resume_from_checkpoint=resolve_resume_checkpoint(output_dir, resume_from_checkpoint),
    )

    save_generation_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        model_name=model_name,
        target_modules=target_modules,
        merge_adapter=merge_adapter,
    )
    write_json(
        output_dir / "train_metadata.json",
        {
            "stage": "sft",
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
    parser = argparse.ArgumentParser(description="Train the SFT concept-discovery baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
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
        print(dry_run_summary("generation-sft-train", args.preset, summary))
        return 0

    metrics = train_generation_sft(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
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
