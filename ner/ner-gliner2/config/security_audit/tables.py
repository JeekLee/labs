"""Render ledger rows as annotated Markdown and record their gold answers.

Each data row becomes one "paragraph" of the v1 document JSON:

    <info line: title / unit / meta fields>
    | header | header | ... |
    |---|---|...|
    | cell | cell | ... |

Entities are marked inline as ``[[Type|surface]]`` so the existing parser, validator, data
conversion and scoring work unchanged. Every row also gets one record (its ledger record type)
and the persistent relations that its columns express (spec: same fact, same gold, whatever
the layout).
"""

from functools import lru_cache

from config.security_audit.forms import Form, Variant
from config.security_audit.ontology import load_ontology

Value = dict  # {"surface": str, "type": str | None}
RowValues = dict[str, Value]

DATE_KINDS = {"date", "datetime"}

# (record, head source, relation, tail source); sources are "row:<column key>" or "meta:<field>"
RELATION_RULES = [
    ("loan", "row:대출자", "BELONGS_TO", "row:소속"),
    ("reading", "row:열람자", "BELONGS_TO", "row:소속"),
    ("clearance", "row:인가 대상자", "BELONGS_TO", "row:소속"),
    ("custodian", "row:보관책임자", "BELONGS_TO", "row:소속"),
    ("entry", "row:출입자", "BELONGS_TO", "row:소속"),
    ("system_entry", "row:취급자", "BELONGS_TO", "row:소속"),
    ("clearance", "row:인가 대상자", "HOLDS", "row:인가번호"),
    ("medium_entry", "meta:관리책임자", "RESPONSIBLE_FOR", "row:등록번호"),
    ("medium_transfer", "meta:관리책임자", "RESPONSIBLE_FOR", "row:관리번호"),
    ("entry", "meta:관리책임자", "RESPONSIBLE_FOR", "meta:구역"),
    ("document_entry", "meta:보관책임자", "RESPONSIBLE_FOR", "row:건명"),
    ("loan", "row:보관책임자", "RESPONSIBLE_FOR", "row:건명"),
    ("system_entry", "row:취급자", "RESPONSIBLE_FOR", "row:관리번호"),
    ("system_entry", "row:관리번호", "LOCATED_AT", "row:설치장소"),
]


@lru_cache(maxsize=1)
def _ontology():
    return load_ontology()


def format_date(value: str, fmt: str) -> str:
    """ISO 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' -> the variant's date style."""
    date, _, time = value.partition(" ")
    y, m, d = date.split("-")
    if fmt == "dot":
        out = f"{y}. {int(m)}. {int(d)}."
    elif fmt == "dash":
        out = f"{y}-{m}-{d}"
    elif fmt == "short":
        out = f"{y[2:]}-{m}-{d}"
    elif fmt == "korean":
        out = f"{y}년 {int(m)}월 {int(d)}일"
    else:
        raise ValueError(f"unknown date format {fmt}")
    return f"{out} {time}" if time else out


def _blank(value: Value | None) -> bool:
    return value is None or not value.get("surface")


def _mark(value: Value | None) -> str:
    if _blank(value):
        return ""
    return f"[[{value['type']}|{value['surface']}]]" if value.get("type") else value["surface"]


def _display(value: Value | None, kind: str, fmt: str) -> Value | None:
    if _blank(value) or kind not in DATE_KINDS:
        return value
    return {"surface": format_date(value["surface"], fmt), "type": value.get("type")}


def _cells(form: Form, variant: Variant, row: RowValues) -> tuple[list[str], list[str]]:
    kinds = {c.key: c.kind for c in form.columns}
    grouped = {key: group for group in variant.grouped for key in group}
    headers, cells, done = [], [], set()
    for key in variant.order:
        if key in done:
            continue
        group = grouped.get(key, [key])
        done.update(group)
        headers.append("·".join(variant.headers[k] for k in group))
        parts = [_mark(_display(row.get(k), kinds[k], variant.date_format)) for k in group]
        cells.append(" ".join(p for p in parts if p))
    return headers, cells


def _info_line(form: Form, meta: RowValues) -> str:
    parts = [form.title]
    if not _blank(meta.get("부대")):
        parts.append(_mark(meta["부대"]))
    for name in form.meta_fields:
        if not _blank(meta.get(name)):
            parts.append(f"{name}: {_mark(meta[name])}")
    return " / ".join(parts)


def _source(spec: str, row: RowValues, meta: RowValues) -> Value | None:
    where, key = spec.split(":", 1)
    return (row if where == "row" else meta).get(key)


def _record_fields(form: Form, variant: Variant, row: RowValues, meta: RowValues) -> dict:
    specs = {f.name: f for f in _ontology().records[form.record].fields}
    kinds = {c.field: c.kind for c in form.columns if c.field}
    values = {c.field: row.get(c.key) for c in form.columns if c.field}
    values.update({name: meta.get(name) for name in form.meta_fields})
    fields = {}
    for name, value in values.items():
        if _blank(value):
            continue
        surface = _display(value, kinds.get(name, ""), variant.date_format)["surface"]
        fields[name] = [surface] if specs[name].dtype == "list" else surface
    return fields


def render_ledger(form: Form, variant: Variant, meta: RowValues, rows: list[RowValues], *,
                  doc_id: str, scenario_id: str, split: str, area: str) -> dict:
    info = _info_line(form, meta)
    paragraphs, relations, records = [], [], []
    for i, row in enumerate(rows):
        headers, cells = _cells(form, variant, row)
        paragraphs.append("\n".join([info, "| " + " | ".join(headers) + " |",
                                     "|" + "---|" * len(headers), "| " + " | ".join(cells) + " |"]))
        seen = set()
        for record, head_spec, kind, tail_spec in RELATION_RULES:
            if record != form.record:
                continue
            head, tail = _source(head_spec, row, meta), _source(tail_spec, row, meta)
            if _blank(head) or _blank(tail) or (head["surface"], kind, tail["surface"]) in seen:
                continue
            seen.add((head["surface"], kind, tail["surface"]))
            relations.append({"paragraph": i, "head": head["surface"], "type": kind, "tail": tail["surface"]})
        records.append({"record": form.record, "text_paragraphs": [i],
                        "fields": _record_fields(form, variant, row, meta)})
    return {"doc_id": doc_id, "scenario_id": scenario_id, "split": split, "doc_type": form.title,
            "doc_kind": "ledger", "variant_id": variant.id, "area_tags": [area],
            "paragraphs": paragraphs, "relations": relations, "records": records}
