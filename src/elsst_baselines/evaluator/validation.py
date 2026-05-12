import json
from collections import Counter
from pathlib import Path


class SubmissionValidationError(ValueError):
    def __init__(self, errors, diagnostics=None):
        self.errors = list(errors)
        self.diagnostics = diagnostics or {}
        super().__init__("; ".join(self.errors))


def read_submission_jsonl(path):
    path = Path(path)
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: expected JSON object")
                continue
            rows.append(row)
    if errors:
        raise SubmissionValidationError(errors, {"row_count": len(rows)})
    return rows


def id_validation_errors(rows, expected_ids):
    observed_ids = []
    errors = []
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            errors.append(f"line {index}: id must be a non-empty string")
            continue
        observed_ids.append(row_id)

    counts = Counter(observed_ids)
    duplicate_ids = sorted(row_id for row_id, count in counts.items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate ids: {', '.join(duplicate_ids[:20])}")

    observed_set = set(observed_ids)
    expected_set = set(expected_ids)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing:
        errors.append(f"missing ids: {', '.join(missing[:20])}")
    if extra:
        errors.append(f"extra ids: {', '.join(extra[:20])}")
    return errors


def first_rows_by_id(rows):
    by_id = {}
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id not in by_id:
            by_id[row_id] = row
    return by_id
