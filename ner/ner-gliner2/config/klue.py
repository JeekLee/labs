"""KLUE-NER loading, conversion and exact-match scoring shared by the experiments.

The logic mirrors notebooks/02_korean_fine_tuning.ipynb so both measure the same thing.
"""

import json
import random
from collections import Counter
from functools import lru_cache
from time import perf_counter

from config.extractors import kiwi, kiwi_splitter

KLUE_TO_LABEL = {"PS": "person", "OG": "organization", "LC": "location"}
LABELS = list(KLUE_TO_LABEL.values())

Span = tuple[int, int, str]
Example = tuple[str, list[Span]]


def load_klue(split: str) -> tuple[list[dict], list[str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("klue/klue", f"ner/{split}-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path)
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    return table.to_pylist(), features["ner_tags"]["feature"]["names"]


def to_example(row: dict, tag_names: list[str]) -> Example:
    """Character BIO tags -> (text, [(start, end, label)])."""
    text = "".join(row["tokens"])
    tags = [tag_names[t] for t in row["ner_tags"]] + ["O"]
    spans, start, kind = [], None, None
    for index, tag in enumerate(tags):
        if start is not None and not tag.startswith("I-"):
            if kind in KLUE_TO_LABEL:
                spans.append((start, index, KLUE_TO_LABEL[kind]))
            start = None
        if tag.startswith("B-"):
            start, kind = index, tag[2:]
    return text, spans


def sample_train(train_rows: list[dict], size: int = 2000, seed: int = 42) -> list[dict]:
    """Same draw as notebook 02: seed, then sample train first."""
    random.seed(seed)
    return random.sample(train_rows, size)


@lru_cache(maxsize=None)
def _boundaries(text: str) -> tuple[frozenset, frozenset]:
    tokens = list(kiwi_splitter(text, lower=False))
    return frozenset(s for _, s, _ in tokens), frozenset(e for _, _, e in tokens)


def is_aligned(text: str, start: int, end: int) -> bool:
    starts, ends = _boundaries(text)
    return start in starts and end in ends


def aligned_ratio(data: list[Example]) -> tuple[int, int]:
    total = sum(len(spans) for _, spans in data)
    aligned = sum(is_aligned(text, s, e) for text, spans in data for s, e, _ in spans)
    return aligned, total


def to_input_example(text: str, spans: list[Span]):
    from gliner2.training.data import InputExample

    entities = {label: [] for label in LABELS}  # empty list = this type is absent
    for start, end, label in spans:
        if is_aligned(text, start, end):
            entities[label].append({"text": text[start:end], "start": start, "end": end})
    return InputExample(text=text, entities=entities)


def has_particle(name: str) -> bool:
    tokens = kiwi().tokenize(name)
    return bool(tokens) and tokens[-1].tag.startswith("J")


def score(predictions: list[set[Span]], data: list[Example]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    particles = 0
    for predicted, (text, spans) in zip(predictions, data, strict=True):
        gold = set(spans)
        particles += sum(has_particle(text[s:e]) for s, e, _ in predicted)
        for _, _, label in predicted & gold:
            tp[label] += 1
        for _, _, label in predicted - gold:
            fp[label] += 1
        for _, _, label in gold - predicted:
            fn[label] += 1
    scores = {}
    for label in [*LABELS, "overall"]:
        t = sum(tp.values()) if label == "overall" else tp[label]
        p = sum(fp.values()) if label == "overall" else fp[label]
        n = sum(fn.values()) if label == "overall" else fn[label]
        precision = t / (t + p) if t + p else 0.0
        recall = t / (t + n) if t + n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {"precision": precision, "recall": recall, "f1": f1}
    scores["particles"] = particles
    return scores


def classify_errors(predictions: list[set[Span]], data: list[Example]) -> dict[str, int]:
    """Each wrong prediction counted once: type > wider > narrower > spurious.

    Gold spans not exactly predicted and not matched by a typed/wider/narrower error are missed.
    """
    counts = {"wider": 0, "narrower": 0, "type": 0, "spurious": 0, "missed": 0}
    for predicted, (_, spans) in zip(predictions, data, strict=True):
        gold = set(spans)
        unmatched = gold - predicted
        for s, e, label in sorted(predicted - gold):
            same_span = next((g for g in unmatched if g[0] == s and g[1] == e), None)
            wider = next((g for g in unmatched if g[2] == label and s <= g[0] and g[1] <= e), None)
            narrower = next((g for g in unmatched if g[2] == label and g[0] <= s and e <= g[1]), None)
            if same_span:
                counts["type"] += 1
                unmatched.discard(same_span)
            elif wider:
                counts["wider"] += 1
                unmatched.discard(wider)
            elif narrower:
                counts["narrower"] += 1
                unmatched.discard(narrower)
            else:
                counts["spurious"] += 1
        counts["missed"] += len(unmatched)
    return counts


def gliner_predict(model, texts: list[str], threshold: float = 0.5) -> tuple[list[set[Span]], float]:
    started = perf_counter()
    predictions = []
    for text in texts:
        result = model.extract_entities(text, LABELS, threshold=threshold, include_spans=True)
        predictions.append({(item["start"], item["end"], label)
                            for label, items in result["entities"].items() for item in items})
    return predictions, perf_counter() - started
