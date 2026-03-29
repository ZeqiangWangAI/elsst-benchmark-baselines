import argparse
import json
from pathlib import Path

import numpy as np

from elsst_baselines.common.gpu import dry_run_summary, retrieval_hparams_for_preset
from elsst_baselines.common.jsonl import read_jsonl, write_json, write_jsonl
from elsst_baselines.retrieval.dataset import (
    build_ir_evaluation_payload,
    load_concept_pool,
    load_track_rows,
    retrieval_dataset_summary,
)
from elsst_baselines.retrieval.modeling import load_retrieval_inference_model


def reciprocal_rank(ranked_ids, relevant_ids):
    for index, concept_id in enumerate(ranked_ids, start=1):
        if concept_id in relevant_ids:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_ids, relevant_ids, k):
    top_ids = set(ranked_ids[:k])
    if not relevant_ids:
        return 0.0
    return len(top_ids & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevant_ids, k):
    gains = [1.0 if concept_id in relevant_ids else 0.0 for concept_id in ranked_ids[:k]]
    dcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = [1.0] * min(len(relevant_ids), k)
    idcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return float(dcg / idcg) if idcg else 0.0


def rank_concepts(model, query_rows, concept_pool, top_k=100):
    concept_ids = list(concept_pool.keys())
    corpus_texts = [
        f"Concept: {concept_pool[concept_id]['term']}\nDefinition: {concept_pool[concept_id]['definition']}"
        for concept_id in concept_ids
    ]
    query_texts = [query_rows[row_id] for row_id in query_rows]

    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_embeddings = model.encode(
        query_texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = np.matmul(query_embeddings, corpus_embeddings.T)
    rankings = {}
    query_ids = list(query_rows.keys())
    for query_index, query_id in enumerate(query_ids):
        sorted_indices = np.argsort(-scores[query_index])[:top_k]
        rankings[query_id] = [concept_ids[index] for index in sorted_indices]
    return rankings


def compute_retrieval_metrics(rankings, relevant_docs):
    mrr = []
    recall_5 = []
    recall_10 = []
    ndcg_10 = []
    for query_id, relevant_ids in relevant_docs.items():
        ranked_ids = rankings[query_id]
        mrr.append(reciprocal_rank(ranked_ids, relevant_ids))
        recall_5.append(recall_at_k(ranked_ids, relevant_ids, 5))
        recall_10.append(recall_at_k(ranked_ids, relevant_ids, 10))
        ndcg_10.append(ndcg_at_k(ranked_ids, relevant_ids, 10))
    return {
        "MRR": float(np.mean(mrr)) if mrr else 0.0,
        "Recall@5": float(np.mean(recall_5)) if recall_5 else 0.0,
        "Recall@10": float(np.mean(recall_10)) if recall_10 else 0.0,
        "NDCG@10": float(np.mean(ndcg_10)) if ndcg_10 else 0.0,
    }


def evaluate_retrieval(dataset_root, output_dir, model_name, preset="auto", adapter_dir=None, max_eval_samples=None, top_k=100):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = retrieval_hparams_for_preset(preset)
    concept_pool = load_concept_pool(dataset_root / "concept_pool.jsonl")
    val_rows = load_track_rows(dataset_root / "val.jsonl", max_rows=max_eval_samples)
    test_rows = load_track_rows(dataset_root / "test_input.jsonl", max_rows=max_eval_samples)

    model = load_retrieval_inference_model(
        model_name=model_name,
        max_seq_length=hparams["max_seq_length"],
        adapter_dir=adapter_dir,
    )

    val_queries, _, relevant_docs = build_ir_evaluation_payload(val_rows, concept_pool)
    test_queries = {row["id"]: f"Instruct: Given a long social-science passage, retrieve the most relevant ELSST concepts.\nQuery: {row['text']}" for row in test_rows}

    val_rankings = rank_concepts(model, val_queries, concept_pool, top_k=top_k)
    test_rankings = rank_concepts(model, test_queries, concept_pool, top_k=top_k)
    metrics = compute_retrieval_metrics(val_rankings, relevant_docs)

    write_jsonl(
        output_dir / "val_retrieval_ranking.jsonl",
        [{"id": row_id, "ranked_ids": ranked_ids} for row_id, ranked_ids in val_rankings.items()],
    )
    write_jsonl(
        output_dir / "test_retrieval_ranking.jsonl",
        [{"id": row_id, "ranked_ids": ranked_ids} for row_id, ranked_ids in test_rankings.items()],
    )
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the retrieval baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        summary = retrieval_dataset_summary(
            args.dataset_root,
            max_eval_samples=args.max_eval_samples,
        )
        print(dry_run_summary("retrieval-evaluate", args.preset, summary))
        return 0

    metrics = evaluate_retrieval(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        preset=args.preset,
        adapter_dir=args.adapter_dir,
        max_eval_samples=args.max_eval_samples,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
