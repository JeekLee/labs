"""Inline annotation format and validators for the synthetic security-audit documents.

Documents mark entities inline as ``[[Type|surface]]``. Relations and record fields refer
to entities by surface. The validators enforce the annotation rules before any document
is used for training or evaluation.
"""

import re

from config.security_audit.ontology import Ontology

Span = tuple[int, int, str]
MARK = re.compile(r"\[\[([A-Za-z]+)\|([^\[\]|]+?)\]\]")


def parse_marked(marked: str) -> tuple[str, list[Span]]:
    """Strip ``[[Type|surface]]`` marks; return plain text and (start, end, type) spans."""
    text, spans, cursor = [], [], 0
    length = 0
    for match in MARK.finditer(marked):
        before = marked[cursor:match.start()]
        text.append(before)
        length += len(before)
        surface = match.group(2)
        spans.append((length, length + len(surface), match.group(1)))
        text.append(surface)
        length += len(surface)
        cursor = match.end()
    text.append(marked[cursor:])
    return "".join(text), spans


RECORD_SEPARATOR = "\n"


def record_text(doc: dict, record: dict) -> str:
    """Plain text of a record unit; shared by validation and training conversion."""
    return RECORD_SEPARATOR.join(parse_marked(doc["paragraphs"][i])[0] for i in record["text_paragraphs"])


def _paragraph_surfaces(doc: dict) -> list[dict[str, str]]:
    """Per paragraph: surface -> type."""
    result = []
    for marked in doc["paragraphs"]:
        text, spans = parse_marked(marked)
        result.append({text[s:e]: t for s, e, t in spans})
    return result


def validate_document(doc: dict, ontology: Ontology) -> list[str]:
    errors = []
    did = doc.get("doc_id", "?")
    extract = ontology.extract_labels()
    surfaces = _paragraph_surfaces(doc)
    for i, marked in enumerate(doc["paragraphs"]):
        text, spans = parse_marked(marked)
        if "[[" in text or "]]" in text:
            errors.append(f"{did} p{i}: malformed markup")
        for s, e, t in spans:
            if t not in extract:
                errors.append(f"{did} p{i}: unknown or non-extract type {t} ({text[s:e]})")
    for rel in doc.get("relations", []):
        i = rel["paragraph"]
        local = surfaces[i] if 0 <= i < len(surfaces) else {}
        head_t, tail_t = local.get(rel["head"]), local.get(rel["tail"])
        if head_t is None or tail_t is None:
            errors.append(f"{did} p{i}: relation endpoint not marked in paragraph: {rel}")
        elif not ontology.allowed(rel["type"], head_t, tail_t):
            errors.append(f"{did} p{i}: relation not allowed {head_t} -{rel['type']}-> {tail_t}")
    for rec in doc.get("records", []):
        if rec["record"] not in ontology.records:
            errors.append(f"{did}: unknown record {rec['record']}")
            continue
        text = record_text(doc, rec)
        fields = {f.name: f for f in ontology.records[rec["record"]].fields}
        for name, value in rec["fields"].items():
            spec = fields.get(name)
            if spec is None:
                errors.append(f"{did}: unknown field {rec['record']}.{name}")
            elif spec.choices:
                if value not in spec.choices:
                    errors.append(f"{did}: {rec['record']}.{name} value {value!r} not in choices")
            else:
                for v in value if isinstance(value, list) else [value]:
                    if v not in text:
                        errors.append(f"{did}: {rec['record']}.{name} value {v!r} not in record text")
    return errors


def validate_corpus(docs: list[dict], ontology: Ontology) -> list[str]:
    errors = [e for doc in docs for e in validate_document(doc, ontology)]
    seen: dict[str, tuple[str, str]] = {}
    for doc in docs:
        for local in _paragraph_surfaces(doc):
            for surface, t in local.items():
                first = seen.setdefault(surface, (t, doc.get("doc_id", "?")))
                if first[0] != t:
                    errors.append(f"surface {surface!r} typed {first[0]} in {first[1]} but {t} in {doc.get('doc_id')}")
    return errors
