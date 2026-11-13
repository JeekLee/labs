"""Loader for the fictional military security-audit ontology (experiments/04).

The YAML follows create-context-graph's ontology keys plus experiment-only keys
(description_ko, extract, records). Everything the extractor needs — GLiNER labels,
the JointIE relation schema, structured-record schemas — is derived from it.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "domains" / "military-security-audit.yaml"


@dataclass(frozen=True)
class EntityType:
    label: str
    pole_type: str
    subtype: str | None
    description_ko: str
    extract: bool


@dataclass(frozen=True)
class RelationType:
    type: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    description_ko: str
    extract: bool
    scopes: tuple[str, ...] = ("narrative",)


@dataclass(frozen=True)
class RecordField:
    name: str
    dtype: str
    choices: tuple[str, ...] | None
    description_ko: str


@dataclass(frozen=True)
class RecordType:
    name: str
    fields: tuple[RecordField, ...]


@dataclass
class Ontology:
    entity_types: dict[str, EntityType]
    relations: list[RelationType]
    records: dict[str, RecordType] = field(default_factory=dict)

    def extract_labels(self) -> dict[str, str]:
        return {t.label: t.description_ko for t in self.entity_types.values() if t.extract}

    def relation(self, name: str) -> RelationType | None:
        return next((r for r in self.relations if r.type == name), None)

    def allowed(self, relation: str, head_type: str, tail_type: str) -> bool:
        r = self.relation(relation)
        return r is not None and head_type in r.sources and tail_type in r.targets

    def joint_schema(self, engine, scope: str = "narrative"):
        schema = engine.create_schema()
        for label, description in self.extract_labels().items():
            schema = schema.entity(label, description)
        for r in self.relations:
            if r.extract and scope in r.scopes:
                # Relation training examples carry no description, so none is used at inference either.
                schema = schema.relation(r.type, list(r.sources), list(r.targets))
        return schema

    def record_schema(self, model, record_name: str):
        builder = model.create_schema().structure(record_name)
        for f in self.records[record_name].fields:
            builder = builder.field(f.name, dtype=f.dtype, choices=list(f.choices) if f.choices else None,
                                    description=f.description_ko)
        return builder


def load_ontology(path: Path = DEFAULT_PATH) -> Ontology:
    data = yaml.safe_load(Path(path).read_text())
    entity_types = {
        e["label"]: EntityType(e["label"], e["pole_type"], e.get("subtype"), e.get("description_ko", ""),
                               bool(e.get("extract", True)))
        for e in data["entity_types"]
    }
    relations = [
        RelationType(r["type"], tuple(r["source"]), tuple(r["target"]), r.get("description_ko", ""),
                     bool(r.get("extract", True)), tuple(r.get("scopes", ["narrative"])))
        for r in data["relationships"]
    ]
    records = {
        name: RecordType(name, tuple(RecordField(f["name"], f["dtype"],
                                                 tuple(f["choices"]) if f.get("choices") else None,
                                                 f.get("description_ko", ""))
                                     for f in fields))
        for name, fields in data.get("records", {}).items()
    }
    return Ontology(entity_types, relations, records)
