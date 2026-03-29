import argparse
import inspect
import json
from pathlib import Path

from elsst_baselines.common.gpu import dry_run_summary, precision_flags, retrieval_hparams_for_preset
from elsst_baselines.common.introspection import filter_supported_kwargs
from elsst_baselines.common.jsonl import write_json
from elsst_baselines.retrieval.dataset import (
    build_ir_evaluation_payload,
    build_retrieval_triplets,
    load_concept_pool,
    load_track_rows,
    retrieval_dataset_summary,
)
from elsst_baselines.retrieval.evaluate import evaluate_retrieval
from elsst_baselines.retrieval.modeling import (
    load_retrieval_inference_model,
    load_retrieval_train_bundle,
    save_retrieval_artifacts,
)


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


def build_loss(model, losses_module):
    kwargs = filter_supported_kwargs(
        losses_module.MultipleNegativesRankingLoss,
        {
            "model": model,
            "hardness_mode": "hard_negatives",
            "hardness_strength": 1.0,
        },
    )
    return losses_module.MultipleNegativesRankingLoss(**kwargs)


def build_training_arguments(training_args_cls, output_dir, hparams):
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
        "batch_sampler": hparams.get("batch_sampler"),
        "save_strategy": hparams.get("save_strategy", "epoch"),
        "save_steps": hparams.get("save_steps"),
        "save_total_limit": hparams.get("save_total_limit"),
        "logging_steps": hparams.get("logging_steps", 10),
        "gradient_checkpointing": hparams["gradient_checkpointing"],
        "remove_unused_columns": False,
        "weight_decay": hparams.get("weight_decay"),
        "warmup_ratio": hparams.get("warmup_ratio"),
        "lr_scheduler_type": hparams.get("lr_scheduler_type"),
        "load_best_model_at_end": hparams.get("load_best_model_at_end"),
        "metric_for_best_model": hparams.get("metric_for_best_model"),
        "greater_is_better": hparams.get("greater_is_better"),
        "seed": hparams.get("seed"),
        "bf16": precision_flags()["bf16"],
        "fp16": precision_flags()["fp16"],
    }
    if "max_steps" in hparams:
        candidate_kwargs["max_steps"] = hparams["max_steps"]
    if strategy_key and hparams.get("eval_steps"):
        candidate_kwargs[strategy_key] = "steps"
        candidate_kwargs["eval_steps"] = hparams["eval_steps"]

    filtered_kwargs = filter_supported_kwargs(training_args_cls, candidate_kwargs)
    return training_args_cls(**filtered_kwargs)


def json_safe_hparams(hparams):
    payload = {}
    for key, value in hparams.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def select_best_checkpoint(output_dir, metric_name, greater_is_better=True):
    output_dir = Path(output_dir)
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    trainer_state_paths = [checkpoint / "trainer_state.json" for checkpoint in reversed(checkpoints)]
    trainer_state_path = next((path for path in trainer_state_paths if path.exists()), None)
    if trainer_state_path is None:
        return None

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    best_model_checkpoint = trainer_state.get("best_model_checkpoint")
    if best_model_checkpoint:
        return str(Path(best_model_checkpoint))

    candidate_entries = []
    for record in trainer_state.get("log_history", []):
        if metric_name not in record:
            continue
        step = record.get("step")
        if step is None:
            continue
        checkpoint_dir = output_dir / f"checkpoint-{step}"
        if checkpoint_dir.exists():
            candidate_entries.append((float(record[metric_name]), checkpoint_dir))

    if not candidate_entries:
        return None

    best_value, best_checkpoint = max(candidate_entries, key=lambda item: item[0]) if greater_is_better else min(candidate_entries, key=lambda item: item[0])
    _ = best_value
    return str(best_checkpoint)


def train_retrieval(dataset_root, output_dir, model_name, preset="auto", max_train_samples=None, max_eval_samples=None, max_steps=None, merge_adapter=False, resume_from_checkpoint=None):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers import losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = retrieval_hparams_for_preset(preset)
    if max_steps is not None:
        hparams["max_steps"] = max_steps
        if "save_steps" in hparams:
            hparams["save_steps"] = min(hparams["save_steps"], max_steps)
        if "eval_steps" in hparams:
            hparams["eval_steps"] = min(hparams["eval_steps"], max_steps)
        hparams["logging_steps"] = max(1, min(hparams.get("logging_steps", 10), max_steps))

    concept_pool = load_concept_pool(dataset_root / "concept_pool.jsonl")
    train_rows = load_track_rows(dataset_root / "train.jsonl")
    val_rows = load_track_rows(dataset_root / "val.jsonl", max_rows=max_eval_samples)
    triplets = build_retrieval_triplets(train_rows, concept_pool)
    if max_train_samples is not None:
        triplets = triplets[:max_train_samples]

    model, target_modules = load_retrieval_train_bundle(
        model_name=model_name,
        max_seq_length=hparams["max_seq_length"],
    )
    train_dataset = Dataset.from_list(
        [
            {
                "anchor": row["query"],
                "positive": row["positive"],
                "negative": row["negative"],
            }
            for row in triplets
        ]
    )
    queries, corpus, relevant_docs = build_ir_evaluation_payload(val_rows, concept_pool)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="elsst-val",
        show_progress_bar=False,
    )
    hparams["batch_sampler"] = BatchSamplers.NO_DUPLICATES
    if hparams.get("eval_steps"):
        hparams["save_strategy"] = "steps"

    metric_for_best_model = hparams.get("metric_for_best_model")
    if metric_for_best_model and hasattr(evaluator, "primary_metric") and evaluator.primary_metric:
        hparams["metric_for_best_model"] = f"eval_{evaluator.primary_metric}"

    export_metric = hparams.get("metric_for_best_model")
    export_greater_is_better = hparams.get("greater_is_better", True)
    hparams["load_best_model_at_end"] = False

    args = build_training_arguments(
        SentenceTransformerTrainingArguments,
        output_dir=output_dir,
        hparams=hparams,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=build_loss(model, losses),
        evaluator=evaluator,
    )
    resolved_resume_checkpoint = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    trainer.train(
        **filter_supported_kwargs(
            trainer.train,
            {"resume_from_checkpoint": resolved_resume_checkpoint},
        )
    )

    export_model = model
    best_checkpoint = select_best_checkpoint(
        output_dir=output_dir,
        metric_name=export_metric,
        greater_is_better=export_greater_is_better,
    )
    if best_checkpoint:
        export_model = load_retrieval_inference_model(
            model_name=model_name,
            max_seq_length=hparams["max_seq_length"],
            adapter_dir=best_checkpoint,
        )

    save_retrieval_artifacts(
        model=export_model,
        output_dir=output_dir,
        model_name=model_name,
        target_modules=target_modules,
        merge_adapter=merge_adapter,
    )
    write_json(
        output_dir / "train_metadata.json",
        {
            "target_modules": target_modules,
            "training_hparams": json_safe_hparams(hparams),
            "resume_from_checkpoint": resolved_resume_checkpoint,
            "best_checkpoint": best_checkpoint,
        },
    )
    metrics = evaluate_retrieval(
        dataset_root=dataset_root,
        output_dir=output_dir,
        model_name=model_name,
        preset=preset,
        adapter_dir=output_dir / "adapter",
        max_eval_samples=max_eval_samples,
    )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the retrieval baseline.")
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
        summary = retrieval_dataset_summary(
            args.dataset_root,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
        print(dry_run_summary("retrieval-train", args.preset, summary))
        return 0

    metrics = train_retrieval(
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
