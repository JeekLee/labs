"""Convert validated security-audit documents into gold items and GLiNER2 training examples.

Entities and relations are taught per paragraph; structured records per record unit.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

from config.security_audit.annotation import parse_marked, record_text
from functools import lru_cache

from config.extractors import kiwi_word_splitter
from config.security_audit.ontology import Ontology

Span = tuple[int, int, str]


@lru_cache(maxsize=None)
def _boundaries(text: str) -> tuple[frozenset, frozenset]:
    tokens = list(kiwi_word_splitter(text, lower=False))
    return frozenset(s for _, s, _ in tokens), frozenset(e for _, _, e in tokens)


def is_aligned(text: str, start: int, end: int) -> bool:
    """Span starts and ends on a token boundary of the splitter used for training (kiwi_word_splitter)."""
    starts, ends = _boundaries(text)
    return start in starts and end in ends
Triple = tuple[str, str, str]
LONG_RECORD_CHARS = 1500

# (text, values) -> values that would not token-align during training
Unalignable = Callable[[str, list[str]], set[str]]


@dataclass(frozen=True)
class ParagraphGold:
    text: str
    spans: list[Span]
    relations: list[Triple]


@dataclass(frozen=True)
class RecordGold:
    record: str
    text: str
    fields: dict


def paragraph_examples(doc: dict, ontology: Ontology) -> list[ParagraphGold]:
    golds = []
    for i, marked in enumerate(doc["paragraphs"]):
        text, spans = parse_marked(marked)
        relations = [(r["head"], r["type"], r["tail"]) for r in doc.get("relations", []) if r["paragraph"] == i]
        golds.append(ParagraphGold(text, spans, relations))
    return golds


def record_examples(doc: dict, ontology: Ontology) -> list[RecordGold]:
    golds = []
    for rec in doc.get("records", []):
        golds.append(RecordGold(rec["record"], record_text(doc, rec), dict(rec["fields"])))
    return golds


def to_training_examples(paragraphs: list[ParagraphGold], records: list[RecordGold], ontology: Ontology,
                         unalignable: Unalignable | None = None) -> tuple[list, dict]:
    from gliner2.training.data import ChoiceField, InputExample, Relation, Structure

    descriptions = ontology.extract_labels()
    labels = list(descriptions)
    stats = {"relations_trained_by_type": {}, "paragraph_examples": 0, "record_examples": 0, "misaligned": 0, "relations_dropped": 0,
             "fields_dropped": 0, "duplicate_surfaces": 0, "long_records": 0}
    examples = []
    for p in paragraphs:
        entities = {label: [] for label in labels}
        for s, e, label in p.spans:
            if is_aligned(p.text, s, e):
                entities[label].append({"text": p.text[s:e], "start": s, "end": e})
            else:
                stats["misaligned"] += 1
        bad = unalignable(p.text, [v for h, _, t in p.relations for v in (h, t)]) if unalignable else set()
        relations = []
        for head, kind, tail in p.relations:
            if head in bad or tail in bad:
                stats["relations_dropped"] += 1
                continue
            stats["duplicate_surfaces"] += (p.text.count(head) > 1) + (p.text.count(tail) > 1)
            relations.append(Relation(kind, head=head, tail=tail))
            stats["relations_trained_by_type"][kind] = stats["relations_trained_by_type"].get(kind, 0) + 1
        examples.append(InputExample(text=p.text, entities=entities, entity_descriptions=dict(descriptions),
                                     relations=relations))
        stats["paragraph_examples"] += 1
    for r in records:
        if len(r.text) > LONG_RECORD_CHARS:
            stats["long_records"] += 1
        spec = {f.name: f for f in ontology.records[r.record].fields}
        values = [v for name, value in r.fields.items() if not spec[name].choices
                  for v in (value if isinstance(value, list) else [value]) if v]
        bad = unalignable(r.text, values) if unalignable else set()
        fields = {}
        for name, value in r.fields.items():
            if spec[name].choices:
                fields[name] = ChoiceField(value, list(spec[name].choices))
                continue
            items = value if isinstance(value, list) else [value]
            kept = [v for v in items if v and v not in bad]
            stats["fields_dropped"] += sum(1 for v in items if v) - len(kept)
            if kept:
                fields[name] = kept if spec[name].dtype == "list" else kept[0]
        field_descriptions = {name: spec[name].description_ko for name in fields}
        examples.append(InputExample(text=r.text, structures=[Structure(r.record, _descriptions=field_descriptions,
                                                                        **fields)]))
        stats["record_examples"] += 1
    return examples, stats


def split_docs(docs: list[dict], dev_ratio: float = 0.15, seed: int = 42) -> dict[str, list]:
    """train/dev split by scenario among train docs; eval docs stay eval."""
    train_scenarios = sorted({d["scenario_id"] for d in docs if d["split"] == "train"})
    rng = random.Random(seed)
    n_dev = max(1, round(len(train_scenarios) * dev_ratio))
    dev = set(rng.sample(train_scenarios, n_dev))
    return {
        "train": [d for d in docs if d["split"] == "train" and d["scenario_id"] not in dev],
        "dev": [d for d in docs if d["split"] == "train" and d["scenario_id"] in dev],
        "eval": [d for d in docs if d["split"] == "eval"],
    }
