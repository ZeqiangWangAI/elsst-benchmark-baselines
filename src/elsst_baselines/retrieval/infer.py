import argparse
import json
from pathlib import Path

from elsst_baselines.common.gpu import dry_run_summary, retrieval_hparams_for_preset
from elsst_baselines.common.jsonl import read_jsonl, write_jsonl
from elsst_baselines.retrieval.dataset import format_query, load_concept_pool
from elsst_baselines.retrieval.evaluate import rank_concepts
from elsst_baselines.retrieval.modeling import load_retrieval_inference_model


def inference_dataset_summary(query_file, concept_pool_path, max_query_samples=None):
    rows = read_jsonl(query_file)
    if max_query_samples is not None:
        rows = rows[:max_query_samples]
    concept_pool = load_concept_pool(concept_pool_path)
    return {
        "query_file": str(query_file),
        "concept_pool": str(concept_pool_path),
        "query_count": len(rows),
        "concept_pool_size": len(concept_pool),
        "top_k": 100,
    }


def infer_retrieval(query_file, concept_pool_path, output_path, model_name, preset="auto", adapter_dir=None, max_query_samples=None, top_k=100):
    query_file = Path(query_file)
    concept_pool_path = Path(concept_pool_path)
    output_path = Path(output_path)

    hparams = retrieval_hparams_for_preset(preset)
    concept_pool = load_concept_pool(concept_pool_path)
    rows = read_jsonl(query_file)
    if max_query_samples is not None:
        rows = rows[:max_query_samples]

    model = load_retrieval_inference_model(
        model_name=model_name,
        max_seq_length=hparams["max_seq_length"],
        adapter_dir=adapter_dir,
    )
    queries = {row["id"]: format_query(row["text"]) for row in rows}
    rankings = rank_concepts(model, queries, concept_pool, top_k=top_k)
    write_jsonl(
        output_path,
        [{"id": row_id, "ranked_ids": ranked_ids} for row_id, ranked_ids in rankings.items()],
    )
    return {
        "query_count": len(rows),
        "top_k": top_k,
        "output_path": str(output_path),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate retrieval rankings for arbitrary query files.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--concept-pool", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--max-query-samples", type=int)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        summary = inference_dataset_summary(
            args.query_file,
            args.concept_pool,
            max_query_samples=args.max_query_samples,
        )
        print(dry_run_summary("retrieval-infer", args.preset, summary))
        return 0

    payload = infer_retrieval(
        query_file=args.query_file,
        concept_pool_path=args.concept_pool,
        output_path=args.output_path,
        model_name=args.model_name,
        preset=args.preset,
        adapter_dir=args.adapter_dir,
        max_query_samples=args.max_query_samples,
        top_k=args.top_k,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
