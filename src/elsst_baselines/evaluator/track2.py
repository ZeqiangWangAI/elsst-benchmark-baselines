from elsst_baselines.evaluator.result import EvaluationResult
from elsst_baselines.evaluator.validation import (
    SubmissionValidationError,
    first_rows_by_id,
    id_validation_errors,
    read_submission_jsonl,
)
from elsst_baselines.generation.parsing import extract_predicted_terms
from elsst_baselines.generation.scoring import (
    bert_score_similarity_matrix,
    exact_term_metrics,
    semantic_set_metrics_from_similarity_matrix,
)


def _normalize_term(term):
    return " ".join(str(term).strip().split()).casefold()


def _dedupe_terms(terms, max_terms):
    deduped = []
    seen = set()
    duplicate_count = 0
    for term in terms:
        normalized = _normalize_term(term)
        if not normalized:
            continue
        if normalized in seen:
            duplicate_count += 1
            continue
        seen.add(normalized)
        deduped.append(str(term).strip())
        if len(deduped) == max_terms:
            break
    return deduped, duplicate_count


def _extract_terms(row, max_terms):
    if "predicted_terms" in row:
        predicted_terms = row["predicted_terms"]
        if not isinstance(predicted_terms, list) or not all(isinstance(item, str) for item in predicted_terms):
            raise TypeError("predicted_terms must be a list of strings")
        terms, duplicate_count = _dedupe_terms(predicted_terms, max_terms=max_terms)
        return {
            "terms": terms,
            "parsed": True,
            "source": "predicted_terms",
            "duplicate_count": duplicate_count,
            "truncated": len(predicted_terms) > len(terms) + duplicate_count,
        }

    raw_text = row.get("raw_text")
    if isinstance(raw_text, str):
        parsed = extract_predicted_terms(raw_text)
        terms, duplicate_count = _dedupe_terms(parsed.terms, max_terms=max_terms)
        return {
            "terms": terms,
            "parsed": parsed.parsed,
            "source": "raw_text",
            "duplicate_count": duplicate_count,
            "truncated": len(parsed.terms) > len(terms) + duplicate_count,
        }

    raise TypeError("expected predicted_terms or raw_text")


def _default_similarity(predicted_terms, gold_terms):
    return bert_score_similarity_matrix(predicted_terms, gold_terms)


def _mean(items, key, count):
    return sum(item[key] for item in items) / count if count else 0.0


def score_submission(
    submission_path,
    reference_rows,
    split,
    similarity_fn=None,
    tau=0.85,
    max_terms=5,
):
    rows = read_submission_jsonl(submission_path)
    expected_ids = [row["id"] for row in reference_rows]
    errors = id_validation_errors(rows, expected_ids)

    parsed_by_id = {}
    raw_text_fallback_count = 0
    empty_prediction_count = 0
    duplicate_term_count = 0
    truncated_row_count = 0
    parse_success_count = 0
    valid_prediction_count = 0

    for row in rows:
        row_id = row.get("id", "<missing>")
        try:
            extracted = _extract_terms(row, max_terms=max_terms)
        except TypeError as exc:
            errors.append(f"{row_id}: {exc}")
            continue

        parsed_by_id.setdefault(row_id, extracted)
        raw_text_fallback_count += 1 if extracted["source"] == "raw_text" else 0
        empty_prediction_count += 1 if not extracted["terms"] else 0
        duplicate_term_count += extracted["duplicate_count"]
        truncated_row_count += 1 if extracted["truncated"] else 0
        parse_success_count += 1 if extracted["parsed"] else 0
        valid_prediction_count += 1 if extracted["terms"] else 0

    diagnostics = {
        "row_count": len(rows),
        "expected_row_count": len(reference_rows),
        "raw_text_fallback_count": raw_text_fallback_count,
        "empty_prediction_count": empty_prediction_count,
        "duplicate_term_count": duplicate_term_count,
        "truncated_row_count": truncated_row_count,
        "max_terms": max_terms,
        "tau": tau,
    }
    if errors:
        raise SubmissionValidationError(errors, diagnostics)

    rows_by_id = first_rows_by_id(rows)
    for row_id in expected_ids:
        if row_id not in parsed_by_id:
            parsed_by_id[row_id] = _extract_terms(rows_by_id[row_id], max_terms=max_terms)

    similarity_fn = similarity_fn or _default_similarity
    exact_metrics = []
    semantic_metrics = []
    predicted_sizes = []

    for reference_row in reference_rows:
        row_id = reference_row["id"]
        predicted_terms = parsed_by_id[row_id]["terms"]
        gold_terms = [item["term"] for item in reference_row["chosen"]]
        exact_metrics.append(exact_term_metrics(predicted_terms, gold_terms))
        if predicted_terms and gold_terms:
            matrix = similarity_fn(predicted_terms, gold_terms)
        else:
            matrix = []
        semantic_metrics.append(
            semantic_set_metrics_from_similarity_matrix(
                similarity_matrix=matrix,
                tau=tau,
                predicted_terms=predicted_terms,
                gold_terms=gold_terms,
            )
        )
        predicted_sizes.append(len(predicted_terms))

    count = len(reference_rows)
    metrics = {
        "parse_rate": parse_success_count / count if count else 0.0,
        "valid_prediction_rate": valid_prediction_count / count if count else 0.0,
        "average_predicted_terms": sum(predicted_sizes) / count if count else 0.0,
        "exact_precision": _mean(exact_metrics, "precision", count),
        "exact_recall": _mean(exact_metrics, "recall", count),
        "exact_f1": _mean(exact_metrics, "f1", count),
        "semantic_precision": _mean(semantic_metrics, "precision", count),
        "semantic_recall": _mean(semantic_metrics, "recall", count),
        "semantic_f1": _mean(semantic_metrics, "f1", count),
        "tau": tau,
    }
    return EvaluationResult(
        track="track2",
        split=split,
        primary_metric="semantic_f1",
        metrics=metrics,
        diagnostics=diagnostics,
    )
