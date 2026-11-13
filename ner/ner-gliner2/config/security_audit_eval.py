"""Prediction and scoring for the security-audit SFT experiment (experiments/04)."""

from collections import Counter
from time import perf_counter

from config.klue import score
from config.security_audit import Ontology
from config.security_audit_data import ParagraphGold, RecordGold

Span = tuple[int, int, str]
Triple = tuple[str, str, str]


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def score_entities(preds: list[set[Span]], golds: list[ParagraphGold], labels: list[str]) -> dict:
    return score(preds, [(g.text, g.spans) for g in golds], labels)


def score_relations(preds: list[set[Triple]], golds: list[ParagraphGold]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    for predicted, gold in zip(preds, golds, strict=True):
        gold_set = set(gold.relations)
        for _, kind, _ in predicted & gold_set:
            tp[kind] += 1
        for _, kind, _ in predicted - gold_set:
            fp[kind] += 1
        for _, kind, _ in gold_set - predicted:
            fn[kind] += 1
    kinds = sorted(set(tp) | set(fp) | set(fn))
    result = {kind: _prf(tp[kind], fp[kind], fn[kind]) for kind in kinds}
    result["overall"] = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return result


def _as_set(value) -> set[str]:
    if value is None or value == "":
        return set()
    return {v for v in (value if isinstance(value, list) else [value]) if v}


def score_records(preds: list[dict], golds: list[RecordGold], ontology: Ontology) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    correct, total = Counter(), Counter()
    for predicted, gold in zip(preds, golds, strict=True):
        for f in ontology.records[gold.record].fields:
            if f.choices:
                if f.name in gold.fields:
                    key = f"{gold.record}.{f.name}"
                    total[key] += 1
                    correct[key] += predicted.get(f.name) == gold.fields[f.name]
                continue
            key = f"{gold.record}.{f.name}"
            p, g = _as_set(predicted.get(f.name)), _as_set(gold.fields.get(f.name))
            tp[key] += len(p & g)
            fp[key] += len(p - g)
            fn[key] += len(g - p)
    names = sorted(set(tp) | set(fp) | set(fn))
    fields = {name: _prf(tp[name], fp[name], fn[name]) for name in names if tp[name] + fp[name] + fn[name]}
    return {"fields": fields, "overall": _prf(sum(tp.values()), sum(fp.values()), sum(fn.values())),
            "choice_accuracy": {name: correct[name] / total[name] for name in total},
            "choice_overall": sum(correct.values()) / sum(total.values()) if total else 0.0}


def score_entities_by_seen(preds: list[set[Span]], golds: list[ParagraphGold],
                           train_surfaces: set[tuple[str, str]], labels: list[str]) -> dict:
    """Entity scores split by whether the gold (surface, type) also occurs in training data.

    A prediction counts toward the bucket of the surface it names, so precision is per bucket too.
    """
    buckets = {"seen": ([], []), "unseen": ([], [])}
    seen_gold = unseen_gold = 0
    for predicted, gold in zip(preds, golds, strict=True):
        is_seen = lambda span: (gold.text[span[0]:span[1]], span[2]) in train_surfaces
        for name, keep in (("seen", True), ("unseen", False)):
            spans = [s for s in gold.spans if is_seen(s) == keep]
            buckets[name][0].append({s for s in predicted if is_seen(s) == keep})
            buckets[name][1].append(ParagraphGold(gold.text, spans, []))
        seen_gold += sum(is_seen(s) for s in gold.spans)
        unseen_gold += sum(not is_seen(s) for s in gold.spans)
    result = {name: score_entities(p, g, labels) for name, (p, g) in buckets.items()}
    result["seen_gold"], result["unseen_gold"] = seen_gold, unseen_gold
    return result


def choice_literal_rate(golds: list[RecordGold], ontology: Ontology) -> dict:
    """How often the gold choice value is written literally in the record text."""
    hits, total = Counter(), Counter()
    for gold in golds:
        for f in ontology.records[gold.record].fields:
            if f.choices and f.name in gold.fields:
                key = f"{gold.record}.{f.name}"
                total[key] += 1
                hits[key] += gold.fields[f.name] in gold.text
    return {key: hits[key] / total[key] for key in total}


def incident_description_scores(model, docs: list[dict]) -> dict:
    """Approach A (descriptive-span extraction) on every paragraph of the eval incident reports."""
    from config.annotation import parse_marked

    label = {"incident_description": "사건 행위를 서술한 절(주어를 제외한 서술부)"}
    tp = fp = fn = overlap = 0
    for doc in docs:
        if not doc.get("incident_description"):
            continue
        gold_by_par: dict[int, set] = {}
        for par, start, end in doc["incident_description"]:
            gold_by_par.setdefault(par, set()).add((start, end))
        for par, marked in enumerate(doc["paragraphs"]):
            text = parse_marked(marked)[0]
            result = model.extract_entities(text, label, include_spans=True)
            pred = {(i["start"], i["end"]) for i in result["entities"].get("incident_description", [])}
            gold = gold_by_par.get(par, set())
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            overlap += sum(any(ps < ge and gs < pe for ps, pe in pred) for gs, ge in gold)
    support = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / support if support else 0.0
    return {"exact_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "exact_precision": precision, "exact_recall": recall,
            "overlap_recall": overlap / support if support else 0.0, "support": support,
            "predicted_spans": tp + fp}


def predict_entities(model, golds: list[ParagraphGold], labels: dict[str, str],
                     threshold: float = 0.5) -> tuple[list[set[Span]], float]:
    started = perf_counter()
    preds = []
    for g in golds:
        result = model.extract_entities(g.text, labels, threshold=threshold, include_spans=True)
        preds.append({(i["start"], i["end"], label) for label, items in result["entities"].items() for i in items})
    return preds, perf_counter() - started


def predict_relations(model, golds: list[ParagraphGold], ontology: Ontology,
                      entity_threshold: float = 0.5) -> tuple[list[set[Triple]], float]:
    from gliner2.joint_ie import JointIEConfig, JointIEEngine

    engine = JointIEEngine(model, device="cpu")
    schema = ontology.joint_schema(engine)
    config = JointIEConfig(entity_threshold=entity_threshold)
    started = perf_counter()
    preds = []
    for g in golds:
        result = engine.extract(g.text, schema, config=config).to_dict()
        names = {e["id"]: e["text"] for e in result["entities"]}
        preds.append({(names[r["head"]], r["type"], names[r["tail"]]) for r in result["relations"]})
    return preds, perf_counter() - started


def predict_records(model, golds: list[RecordGold], ontology: Ontology) -> tuple[list[dict], float]:
    schemas = {name: ontology.record_schema(model, name) for name in ontology.records}
    started = perf_counter()
    preds = []
    for g in golds:
        instances = model.extract(g.text, schemas[g.record]).get(g.record) or [{}]
        first = instances[0] if isinstance(instances, list) else instances
        preds.append({k: v for k, v in (first or {}).items() if v not in (None, "", [])})
    return preds, perf_counter() - started
