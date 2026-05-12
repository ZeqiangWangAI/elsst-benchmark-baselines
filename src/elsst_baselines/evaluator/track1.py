from elsst_baselines.evaluator.result import EvaluationResult
from elsst_baselines.evaluator.validation import (
    SubmissionValidationError,
    first_rows_by_id,
    id_validation_errors,
    read_submission_jsonl,
)
from elsst_baselines.retrieval.evaluate import compute_retrieval_metrics


def _normalize_concept_pool(concept_pool):
    if isinstance(concept_pool, dict):
        return concept_pool
    return {
        row["concept_id"]: {"term": row["term"], "definition": row["definition"]}
        for row in concept_pool
    }


def score_submission(
    submission_path,
    reference_rows,
    concept_pool,
    split,
    min_ranked_ids=10,
    top_k=100,
):
    concept_pool = _normalize_concept_pool(concept_pool)
    rows = read_submission_jsonl(submission_path)
    expected_ids = [row["id"] for row in reference_rows]
    errors = id_validation_errors(rows, expected_ids)
    ranking_lengths = []
    truncated_rows = 0

    for row in rows:
        row_id = row.get("id", "<missing>")
        ranked_ids = row.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not all(isinstance(item, str) for item in ranked_ids):
            errors.append(f"{row_id}: ranked_ids must be a list of strings")
            continue
        ranking_lengths.append(len(ranked_ids))
        if len(ranked_ids) < min_ranked_ids:
            errors.append(f"{row_id}: ranked_ids must contain at least {min_ranked_ids} concept ids")

        duplicates = sorted({concept_id for concept_id in ranked_ids if ranked_ids.count(concept_id) > 1})
        if duplicates:
            errors.append(f"{row_id}: duplicate ranked_ids: {', '.join(duplicates[:20])}")

        unknown = sorted({concept_id for concept_id in ranked_ids if concept_id not in concept_pool})
        if unknown:
            errors.append(f"{row_id}: unknown concept ids: {', '.join(unknown[:20])}")

        if len(ranked_ids) > top_k:
            truncated_rows += 1

    diagnostics = {
        "row_count": len(rows),
        "expected_row_count": len(reference_rows),
        "min_ranked_ids": min(ranking_lengths) if ranking_lengths else 0,
        "max_ranked_ids": max(ranking_lengths) if ranking_lengths else 0,
        "truncated_row_count": truncated_rows,
        "top_k": top_k,
    }
    if errors:
        raise SubmissionValidationError(errors, diagnostics)

    rows_by_id = first_rows_by_id(rows)
    rankings = {
        row_id: rows_by_id[row_id]["ranked_ids"][:top_k]
        for row_id in expected_ids
    }
    relevant_docs = {
        row["id"]: set(row["retrieval_labels"]["positive_ids"])
        for row in reference_rows
    }
    metrics = compute_retrieval_metrics(rankings, relevant_docs)
    return EvaluationResult(
        track="track1",
        split=split,
        primary_metric="NDCG@10",
        metrics=metrics,
        diagnostics=diagnostics,
    )
