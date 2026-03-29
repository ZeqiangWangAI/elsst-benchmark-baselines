from pathlib import Path

from elsst_baselines.common.jsonl import read_jsonl


QUERY_INSTRUCTION = "Instruct: Given a long social-science passage, retrieve the most relevant ELSST concepts."


def format_query(text):
    return f"{QUERY_INSTRUCTION}\nQuery: {text}"


def format_concept(concept):
    return f"Concept: {concept['term']}\nDefinition: {concept['definition']}"


def load_concept_pool(path):
    rows = read_jsonl(path)
    return {
        row["concept_id"]: {"term": row["term"], "definition": row["definition"]}
        for row in rows
    }


def load_track_rows(path, max_rows=None):
    rows = read_jsonl(path)
    if max_rows is not None:
        return rows[:max_rows]
    return rows


def build_retrieval_triplets(rows, concept_pool):
    triplets = []
    for row in rows:
        query = format_query(row["text"])
        positive_map = {label["concept_id"]: label for label in row["generation_labels"]}
        for positive_id in row["retrieval_labels"]["positive_ids"]:
            positive_label = positive_map[positive_id]
            positive_text = format_concept(positive_label)
            for negative_id in row["retrieval_labels"]["hard_negative_ids"]:
                negative_text = format_concept(concept_pool[negative_id])
                triplets.append(
                    {
                        "query_id": row["id"],
                        "query": query,
                        "positive_id": positive_id,
                        "positive": positive_text,
                        "negative_id": negative_id,
                        "negative": negative_text,
                    }
                )
    return triplets


def build_ir_evaluation_payload(rows, concept_pool):
    queries = {}
    relevant_docs = {}
    for row in rows:
        queries[row["id"]] = format_query(row["text"])
        relevant_docs[row["id"]] = set(row["retrieval_labels"]["positive_ids"])
    corpus = {
        concept_id: format_concept({"term": concept["term"], "definition": concept["definition"]})
        for concept_id, concept in concept_pool.items()
    }
    return queries, corpus, relevant_docs


def retrieval_dataset_summary(dataset_root, max_train_samples=None, max_eval_samples=None):
    dataset_root = Path(dataset_root)
    concept_pool = load_concept_pool(dataset_root / "concept_pool.jsonl")
    train_rows = load_track_rows(dataset_root / "train.jsonl")
    val_rows = load_track_rows(dataset_root / "val.jsonl")
    train_triplets = build_retrieval_triplets(train_rows, concept_pool)
    eval_rows = val_rows[:max_eval_samples] if max_eval_samples else val_rows
    return {
        "dataset_root": str(dataset_root),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "concept_pool_size": len(concept_pool),
        "train_triplets_total": len(train_triplets),
        "requested_max_train_samples": max_train_samples,
        "requested_max_eval_samples": max_eval_samples,
        "effective_train_samples": min(len(train_triplets), max_train_samples) if max_train_samples else len(train_triplets),
        "effective_eval_queries": len(eval_rows),
    }
