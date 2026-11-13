"""Store documents as plain Markdown and their gold answers as separate label JSON.

    documents/<split>/<doc_id>.md     what GLiNER reads (no annotation marks)
    labels/<split>/<doc_id>.json      gold: entities (file offsets), units, relations, records

A *unit* is what one model input is built from: a paragraph for narrative documents, and
"info line + header + separator + one data row" for ledgers (the Markdown keeps a single header).
Units are stored as lists of file spans joined with newlines, so the loader rebuilds exactly the
text the model sees and maps every entity offset into it.

Internally the rest of the code works on the annotated "document dict" (marked paragraphs,
relations by paragraph, records by text paragraphs). ``doc_to_files`` and ``files_to_doc``
convert between the two.
"""

import json
from pathlib import Path

from config.security_audit.annotation import parse_marked

META_KEYS = ("doc_id", "scenario_id", "split", "doc_type", "doc_kind", "variant_id", "area_tags")


class _Writer:
    def __init__(self):
        self.text = ""

    def add(self, chunk: str) -> tuple[int, int]:
        start = len(self.text)
        self.text += chunk
        return start, len(self.text)


def _entities(segment_text: str, spans, offset: int) -> list[dict]:
    return [{"start": offset + s, "end": offset + e, "type": t, "text": segment_text[s:e]} for s, e, t in spans]


def doc_to_files(doc: dict) -> tuple[str, dict]:
    w = _Writer()
    entities, units = [], []
    w.add(f"# {doc['doc_type']}\n\n")
    if doc.get("doc_kind") == "ledger":
        first = doc["paragraphs"][0].split("\n")
        info_text, info_spans = parse_marked(first[0])
        info = w.add(info_text)
        entities += _entities(info_text, info_spans, info[0])
        w.add("\n\n")
        header = w.add(first[1])
        w.add("\n")
        sep = w.add(first[2])
        for paragraph in doc["paragraphs"]:
            row_text, row_spans = parse_marked(paragraph.split("\n")[3])
            w.add("\n")
            row = w.add(row_text)
            entities += _entities(row_text, row_spans, row[0])
            units.append({"segments": [list(info), list(header), list(sep), list(row)]})
        w.add("\n")
    else:
        for i, paragraph in enumerate(doc["paragraphs"]):
            text, spans = parse_marked(paragraph)
            if i:
                w.add("\n\n")
            span = w.add(text)
            entities += _entities(text, spans, span[0])
            units.append({"segments": [list(span)]})
        w.add("\n")
    labels = {k: doc[k] for k in META_KEYS if k in doc}
    labels.update({
        "entities": entities, "units": units,
        "relations": [{"unit": r["paragraph"], "head": r["head"], "type": r["type"], "tail": r["tail"]}
                      for r in doc.get("relations", [])],
        "records": [{"record": r["record"], "units": r["text_paragraphs"], "fields": r["fields"]}
                    for r in doc.get("records", [])],
    })
    return w.text, labels


def _mark(text: str, spans: list[tuple[int, int, str]]) -> str:
    out, cursor = [], 0
    for s, e, t in sorted(spans):
        out += [text[cursor:s], f"[[{t}|{text[s:e]}]]"]
        cursor = e
    return "".join(out) + text[cursor:]


def unit_text_and_spans(md: str, unit: dict, entities: list[dict]) -> tuple[str, list[tuple[int, int, str]]]:
    """Text of one unit (segments joined by newlines) and the entity spans mapped into it."""
    parts, spans, offset = [], [], 0
    for s, e in unit["segments"]:
        parts.append(md[s:e])
        for ent in entities:
            if s <= ent["start"] and ent["end"] <= e:
                spans.append((offset + ent["start"] - s, offset + ent["end"] - s, ent["type"]))
        offset += (e - s) + 1
    return "\n".join(parts), spans


def files_to_doc(md: str, labels: dict) -> dict:
    paragraphs = []
    for unit in labels["units"]:
        text, spans = unit_text_and_spans(md, unit, labels["entities"])
        paragraphs.append(_mark(text, spans))
    doc = {k: labels[k] for k in META_KEYS if k in labels}
    doc.update({
        "paragraphs": paragraphs,
        "relations": [{"paragraph": r["unit"], "head": r["head"], "type": r["type"], "tail": r["tail"]}
                      for r in labels["relations"]],
        "records": [{"record": r["record"], "text_paragraphs": r["units"], "fields": r["fields"]}
                    for r in labels["records"]],
    })
    return doc


def write_doc(doc: dict, data_dir: Path) -> None:
    md, labels = doc_to_files(doc)
    md_path = Path(data_dir) / "documents" / doc["split"] / f"{doc['doc_id']}.md"
    label_path = Path(data_dir) / "labels" / doc["split"] / f"{doc['doc_id']}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    label_path.write_text(json.dumps(labels, ensure_ascii=False, indent=1) + "\n")


def load_docs(data_dir: Path, splits=("train", "eval")) -> list[dict]:
    docs = []
    for split in splits:
        for label_path in sorted((Path(data_dir) / "labels" / split).glob("*.json")):
            md_path = Path(data_dir) / "documents" / split / f"{label_path.stem}.md"
            docs.append(files_to_doc(md_path.read_text(), json.loads(label_path.read_text())))
    return docs
