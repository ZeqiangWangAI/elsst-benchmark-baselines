import json
import re
from dataclasses import dataclass


@dataclass
class ParsedPrediction:
    parsed: bool
    terms: list
    normalized_terms: list


def _normalize_term(term):
    return " ".join(term.strip().split()).casefold()


def _coerce_terms(payload):
    if isinstance(payload, dict):
        for key in ("predicted_terms", "terms"):
            if key in payload:
                return _coerce_terms(payload[key])
        return []
    if isinstance(payload, list):
        terms = []
        for item in payload:
            if isinstance(item, str):
                if item.strip():
                    terms.append(item.strip())
            elif isinstance(item, dict):
                term = item.get("term") or item.get("label")
                if isinstance(term, str) and term.strip():
                    terms.append(term.strip())
        return terms
    return []


def _extract_json_array(raw_text):
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return raw_text[start : end + 1]


def _extract_plain_text_terms(raw_text):
    cleaned = raw_text.replace("```", " ").strip()
    if not cleaned:
        return []

    terms = []
    for segment in re.split(r"[;\n]+", cleaned):
        segment = segment.strip()
        if not segment:
            continue
        segment = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", segment)
        if ":" not in segment and "：" not in segment:
            continue
        term, definition = re.split(r"[:：]", segment, maxsplit=1)
        term = term.strip(" \t'\"`")
        definition = definition.strip()
        if not term or not definition:
            continue
        terms.append(term)
        if len(terms) >= 5:
            break
    return terms


def extract_predicted_terms(raw_text):
    raw_text = raw_text.strip()
    payload = None
    parsed = False

    plain_text_terms = _extract_plain_text_terms(raw_text)
    if plain_text_terms:
        payload = plain_text_terms
        parsed = True

    if payload is None:
        for candidate in (raw_text, _extract_json_array(raw_text)):
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
                parsed = True
                break
            except json.JSONDecodeError:
                continue

    if payload is None:
        return ParsedPrediction(parsed=False, terms=[], normalized_terms=[])

    terms = _coerce_terms(payload)
    normalized = []
    deduped_terms = []
    seen = set()
    for term in terms:
        normalized_term = _normalize_term(term)
        if not normalized_term or normalized_term in seen:
            continue
        seen.add(normalized_term)
        normalized.append(normalized_term)
        deduped_terms.append(term)
        if len(deduped_terms) >= 5:
            break
    return ParsedPrediction(parsed=parsed, terms=deduped_terms, normalized_terms=normalized)
