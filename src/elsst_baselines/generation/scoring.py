import itertools
import math


_BERT_SCORER_CACHE = {}


def _precision_recall_f1(matches, predicted_count, gold_count):
    precision = matches / predicted_count if predicted_count else 0.0
    recall = matches / gold_count if gold_count else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _best_assignment(similarity_matrix):
    rows = len(similarity_matrix)
    cols = len(similarity_matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    row_indices = list(range(rows))
    col_indices = list(range(cols))
    if rows <= cols:
        best_pairs = []
        best_score = -math.inf
        for chosen_cols in itertools.permutations(col_indices, rows):
            score = sum(similarity_matrix[row][col] for row, col in zip(row_indices, chosen_cols))
            if score > best_score:
                best_score = score
                best_pairs = list(zip(row_indices, chosen_cols))
        return best_pairs

    best_pairs = []
    best_score = -math.inf
    for chosen_rows in itertools.permutations(row_indices, cols):
        score = sum(similarity_matrix[row][col] for row, col in zip(chosen_rows, col_indices))
        if score > best_score:
            best_score = score
            best_pairs = list(zip(chosen_rows, col_indices))
    return best_pairs


def semantic_set_metrics_from_similarity_matrix(similarity_matrix, tau, predicted_terms, gold_terms):
    assignments = _best_assignment(similarity_matrix)
    matches = 0
    matched_scores = []
    for row, col in assignments:
        score = similarity_matrix[row][col]
        if score >= tau:
            matches += 1
            matched_scores.append(score)

    precision, recall, f1 = _precision_recall_f1(matches, len(predicted_terms), len(gold_terms))
    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_scores": matched_scores,
    }


def exact_term_metrics(predicted_terms, gold_terms):
    pred = {term.strip().casefold() for term in predicted_terms if term.strip()}
    gold = {term.strip().casefold() for term in gold_terms if term.strip()}
    matches = len(pred & gold)
    precision, recall, f1 = _precision_recall_f1(matches, len(pred), len(gold))
    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def bert_score_similarity_matrix(predicted_terms, gold_terms, model_type=None):
    if not predicted_terms or not gold_terms:
        return []
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("bert-score is required for semantic-set evaluation") from exc

    cache_key = model_type or "__default__"
    scorer = _BERT_SCORER_CACHE.get(cache_key)
    if scorer is None:
        scorer_kwargs = {"lang": "en"}
        if model_type is not None:
            scorer_kwargs["model_type"] = model_type
        scorer = BERTScorer(**scorer_kwargs)
        _BERT_SCORER_CACHE[cache_key] = scorer

    cartesian_predictions = []
    cartesian_gold = []
    for predicted_term in predicted_terms:
        for gold_term in gold_terms:
            cartesian_predictions.append(predicted_term)
            cartesian_gold.append(gold_term)
    _, _, f1 = scorer.score(
        cands=cartesian_predictions,
        refs=cartesian_gold,
        verbose=False,
    )

    values = [float(item) for item in f1]
    matrix = []
    offset = 0
    for _ in predicted_terms:
        matrix.append(values[offset : offset + len(gold_terms)])
        offset += len(gold_terms)
    return matrix


def semantic_set_metrics(predicted_terms, gold_terms, tau=0.85, model_type=None):
    matrix = bert_score_similarity_matrix(predicted_terms, gold_terms, model_type=model_type)
    return semantic_set_metrics_from_similarity_matrix(
        similarity_matrix=matrix,
        tau=tau,
        predicted_terms=predicted_terms,
        gold_terms=gold_terms,
    )
