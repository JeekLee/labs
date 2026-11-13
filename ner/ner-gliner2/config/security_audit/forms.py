"""Ledger form definitions: canonical columns per ledger and per-unit rendering variants.

Columns follow public Korean security-work forms (see the v1 design doc's reference section).
Variants model how different (fictional) units label, order, group and date the same columns.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

FORMS_DIR = Path(__file__).resolve().parents[2] / "experiments/data/security_audit/forms"


@dataclass(frozen=True)
class Column:
    key: str               # stable column id used by variants
    field: str | None      # record field this column fills (None: row number etc.)
    kind: str              # value generator kind (person, unit, date, choice, text, ...)


@dataclass(frozen=True)
class Variant:
    id: str
    eval_only: bool
    headers: dict[str, str]      # column key -> header text
    order: list[str]             # column keys, left to right
    grouped: list[list[str]]     # column keys rendered together in one cell
    date_format: str             # dot | dash | short | korean


@dataclass(frozen=True)
class Form:
    record: str
    title: str
    meta_fields: list[str]       # record fields shown in the info line above the table
    columns: list[Column]
    variants: list[Variant]


def load_form(path: Path) -> Form:
    data = yaml.safe_load(Path(path).read_text())
    columns = [Column(c["key"], c.get("field"), c["kind"]) for c in data["columns"]]
    variants = [Variant(v["id"], bool(v.get("eval_only", False)), dict(v["headers"]), list(v["order"]),
                        [list(g) for g in v.get("grouped", [])], v["date_format"]) for v in data["variants"]]
    return Form(data["record"], data["title"], list(data.get("meta_fields", [])), columns, variants)


def load_forms(directory: Path = FORMS_DIR) -> dict[str, Form]:
    return {form.record: form for form in map(load_form, sorted(Path(directory).glob("*.yaml")))}
