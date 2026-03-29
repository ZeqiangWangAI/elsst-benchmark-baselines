import argparse
import json
from pathlib import Path

from elsst_baselines.common.gpu import dry_run_summary, generation_hparams_for_preset
from elsst_baselines.common.jsonl import write_json, write_jsonl
from elsst_baselines.generation.dataset import generation_dataset_summary, load_track2_rows
from elsst_baselines.generation.modeling import (
    build_generation_prompt,
    load_generation_inference_bundle,
)
from elsst_baselines.generation.parsing import extract_predicted_terms
from elsst_baselines.generation.scoring import exact_term_metrics, semantic_set_metrics


def generate_predictions(model, tokenizer, rows, max_prompt_length, max_completion_length):
    predictions = []
    for row in rows:
        prompt_text = build_generation_prompt(row["prompt"], tokenizer, disable_thinking=True)
        batch = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        generated_ids = model.generate(
            **batch,
            max_new_tokens=max_completion_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated_ids[0][batch["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(
            {
                "id": row["id"],
                "raw_text": raw_text,
                "raw_text_length": len(raw_text),
                "ended_with_semicolon": raw_text.rstrip().endswith(";"),
            }
        )
    return predictions


def evaluate_generation(dataset_root, output_dir, model_name, preset="auto", adapter_dir=None, max_eval_samples=None, tau=0.85):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = generation_hparams_for_preset(preset)
    rows = load_track2_rows(dataset_root / "val.jsonl", max_rows=max_eval_samples)
    model, tokenizer, _ = load_generation_inference_bundle(model_name, adapter_dir=adapter_dir, qlora=True)
    raw_predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        max_prompt_length=hparams["max_prompt_length"],
        max_completion_length=hparams["max_completion_length"],
    )

    submission_rows = []
    debug_rows = []
    parse_successes = 0
    predicted_sizes = []
    exact_metrics = []
    semantic_metrics = []

    row_by_id = {row["id"]: row for row in rows}
    for prediction in raw_predictions:
        row = row_by_id[prediction["id"]]
        parsed = extract_predicted_terms(prediction["raw_text"])
        gold_terms = [item["term"] for item in row["chosen"]]
        exact = exact_term_metrics(parsed.terms, gold_terms)
        semantic = semantic_set_metrics(parsed.terms, gold_terms, tau=tau)
        parse_successes += 1 if parsed.parsed else 0
        predicted_sizes.append(len(parsed.terms))
        exact_metrics.append(exact)
        semantic_metrics.append(semantic)
        submission_rows.append({"id": row["id"], "predicted_terms": parsed.terms[:10]})
        debug_rows.append(
            {
                "id": row["id"],
                "raw_text": prediction["raw_text"],
                "raw_text_length": prediction["raw_text_length"],
                "ended_with_semicolon": prediction["ended_with_semicolon"],
                "parsed_terms": parsed.terms,
                "predicted_term_count": len(parsed.terms),
                "gold_terms": gold_terms,
                "parsed": parsed.parsed,
                "exact": exact,
                "semantic": semantic,
            }
        )

    count = len(rows) or 1
    metrics = {
        "parse_rate": parse_successes / count,
        "json_parse_rate": parse_successes / count,
        "average_predicted_terms": sum(predicted_sizes) / count if predicted_sizes else 0.0,
        "exact_precision": sum(item["precision"] for item in exact_metrics) / count,
        "exact_recall": sum(item["recall"] for item in exact_metrics) / count,
        "exact_f1": sum(item["f1"] for item in exact_metrics) / count,
        "semantic_precision": sum(item["precision"] for item in semantic_metrics) / count,
        "semantic_recall": sum(item["recall"] for item in semantic_metrics) / count,
        "semantic_f1": sum(item["f1"] for item in semantic_metrics) / count,
        "tau": tau,
    }

    write_jsonl(output_dir / "generation_semantic.jsonl", submission_rows)
    write_jsonl(output_dir / "debug_predictions.jsonl", debug_rows)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the generation concept-discovery baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        summary = generation_dataset_summary(
            args.dataset_root,
            max_eval_samples=args.max_eval_samples,
        )
        print(dry_run_summary("generation-evaluate", args.preset, summary))
        return 0

    metrics = evaluate_generation(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        preset=args.preset,
        adapter_dir=args.adapter_dir,
        max_eval_samples=args.max_eval_samples,
        tau=args.tau,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
