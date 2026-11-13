"""Experiment 04 (v2): SFT a GLiNER2 LoRA adapter on the fictional security-audit ontology.

    uv run python scripts/security_audit/run_sft.py                                  # full run
    uv run python scripts/security_audit/run_sft.py --limit-scenarios 3 --max-steps 4  # smoke run

Ledger rows and narrative paragraphs are trained together in one adapter. Results are split by
document kind (ledger / narrative), by ledger record type, by whether the ledger's column variant
was seen in training, and by whether an entity surface was seen in training.
"""

import argparse
import json
import os
import platform
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

from huggingface_hub.utils import logging as hf_logging  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

hf_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

from config.extractors import kiwi_word_splitter  # noqa: E402
from config.security_audit.annotation import validate_corpus  # noqa: E402
from config.security_audit.data import paragraph_examples, record_examples, split_docs, to_training_examples  # noqa: E402
from config.security_audit.evaluate import (choice_literal_split, predict_entities, predict_records,  # noqa: E402
                                            predict_relations, score_by_group, score_entities,
                                            score_entities_by_seen, score_records, score_relations)
from config.security_audit.ontology import load_ontology  # noqa: E402
from config.security_audit.storage import load_docs as load_stored  # noqa: E402

BASE_MODEL = "fastino/gliner2.5-multi-v1"
DATA = ROOT / "experiments/data/security_audit"
EPOCHS, BATCH_SIZE, SEED = 2, 8, 42


def log(message: str) -> None:
    print(message, flush=True)


def load_docs() -> list[dict]:
    return load_stored(DATA)


def item_meta(doc: dict) -> dict:
    variant = doc.get("variant_id", "")
    return {"doc_id": doc["doc_id"], "doc_kind": doc.get("doc_kind", "narrative"), "doc_type": doc["doc_type"],
            "variant": "n/a" if not variant else ("unseen" if variant.endswith("_eval") else "seen")}


def golds_for(docs: list[dict], ontology) -> dict:
    paragraphs, records = [], []
    for doc in docs:
        meta = item_meta(doc)
        paragraphs += [{"gold": p, **meta} for p in paragraph_examples(doc, ontology)]
        records += [{"gold": r, **meta} for r in record_examples(doc, ontology)]
    return {"paragraphs": paragraphs, "records": records}


def load_model(adapter: Path | None = None):
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(BASE_MODEL)
    if adapter is not None:
        model.load_adapter(str(adapter))
    model.set_word_splitter(kiwi_word_splitter)
    return model


def predict(model, golds: dict, ontology) -> dict:
    """Entity, relation (scope per document kind) and record predictions, aligned with golds."""
    paragraphs = golds["paragraphs"]
    ents, ent_seconds = predict_entities(model, [p["gold"] for p in paragraphs], ontology.extract_labels())
    rels, rel_seconds = [None] * len(paragraphs), 0.0
    for kind, scope in (("ledger", "ledger"), ("narrative", "narrative")):
        idx = [i for i, p in enumerate(paragraphs) if p["doc_kind"] == kind]
        if idx:
            preds, seconds = predict_relations(model, [paragraphs[i]["gold"] for i in idx], ontology, scope=scope)
            rel_seconds += seconds
            for i, pred in zip(idx, preds):
                rels[i] = pred
    recs, rec_seconds = predict_records(model, [r["gold"] for r in golds["records"]], ontology)
    return {"entities": ents, "relations": rels, "records": recs,
            "seconds": {"entities": ent_seconds, "relations": rel_seconds, "records": rec_seconds}}


def score(golds: dict, preds: dict, ontology, train_surfaces: set) -> dict:
    labels = list(ontology.extract_labels())
    para = [{"gold": g["gold"], "pred_e": e, "pred_r": r, **{k: g[k] for k in ("doc_kind", "variant", "doc_type")}}
            for g, e, r in zip(golds["paragraphs"], preds["entities"], preds["relations"])]
    recs = [{"gold": g["gold"], "pred": p, "record": g["gold"].record, **{k: g[k] for k in ("doc_kind", "variant")}}
            for g, p in zip(golds["records"], preds["records"])]

    def ents(items):
        return score_entities([i["pred_e"] for i in items], [i["gold"] for i in items], labels)

    def rels(items):
        return score_relations([i["pred_r"] for i in items], [i["gold"] for i in items])

    def records(items):
        return score_records([i["pred"] for i in items], [i["gold"] for i in items], ontology)

    ledger_rows = [i for i in para if i["doc_kind"] == "ledger"]
    ledger_recs = [i for i in recs if i["doc_kind"] == "ledger"]
    return {
        "entities": ents(para),
        "entities_by_seen": score_entities_by_seen(preds["entities"], [g["gold"] for g in golds["paragraphs"]],
                                                   train_surfaces, labels),
        "entities_by_doc_kind": score_by_group(ents, para, "doc_kind"),
        "entities_by_variant": score_by_group(ents, ledger_rows, "variant"),
        "relations": rels(para),
        "relations_by_doc_kind": score_by_group(rels, para, "doc_kind"),
        "relations_by_variant": score_by_group(rels, ledger_rows, "variant"),
        "records": records(recs),
        "records_by_doc_kind": score_by_group(records, recs, "doc_kind"),
        "records_by_record": score_by_group(records, recs, "record"),
        "records_by_variant": score_by_group(records, ledger_recs, "variant"),
        "choice_literal_split": choice_literal_split([i["pred"] for i in recs], [i["gold"] for i in recs], ontology),
        "seconds": preds["seconds"],
        "counts": {"paragraphs": len(para), "ledger_rows": len(ledger_rows), "records": len(recs)},
    }


def example_rows(docs, golds: dict, preds: dict, kind: str, limit: int) -> list[dict]:
    out = []
    for i, g in enumerate(golds["paragraphs"]):
        if g["doc_kind"] != kind or len(out) >= limit:
            continue
        gold = g["gold"]
        rec_idx = [j for j, r in enumerate(golds["records"]) if r["doc_id"] == g["doc_id"] and r["gold"].text == gold.text]
        out.append({
            "doc_id": g["doc_id"], "variant": g["variant"], "text": gold.text,
            "gold": {"entities": sorted({(gold.text[s:e], t) for s, e, t in gold.spans}),
                     "relations": sorted(set(gold.relations)),
                     "records": [golds["records"][j]["gold"].fields for j in rec_idx]},
            "predicted": {"entities": sorted({(gold.text[s:e], t) for s, e, t in preds["entities"][i]}),
                          "relations": sorted(preds["relations"][i]),
                          "records": [preds["records"][j] for j in rec_idx]},
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-scenarios", type=int, default=None, help="smoke run: N scenarios per split")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    ontology = load_ontology()
    docs = load_docs()
    errors = validate_corpus(docs, ontology)
    if errors:
        raise SystemExit(f"{len(errors)} validation errors, e.g. {errors[:3]}")
    splits = split_docs(docs)
    if args.limit_scenarios:
        keep = lambda ds: [d for d in ds if d["scenario_id"] in sorted({x["scenario_id"] for x in ds})[: args.limit_scenarios]]
        splits = {k: keep(v) for k, v in splits.items()}
    out_dir = ROOT / ("outputs/security_audit_lora_smoke" if args.limit_scenarios else "outputs/security_audit_lora")
    result_path = out_dir / "result.json" if args.limit_scenarios else ROOT / "experiments/results/security_audit_sft.json"
    golds = {name: golds_for(split, ontology) for name, split in splits.items()}
    train_surfaces = {(p["gold"].text[s:e], t) for p in golds["train"]["paragraphs"] for s, e, t in p["gold"].spans}
    log("[data] docs " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    model = load_model()
    processor = model.processor

    def unalignable(text, values):
        if not values:
            return set()
        return {item["value"] for item in processor.find_unalignable_entities(text, {"value": list(values)})}

    examples, data_stats = to_training_examples([p["gold"] for p in golds["train"]["paragraphs"]],
                                                [r["gold"] for r in golds["train"]["records"]], ontology, unalignable)
    dataset = TrainingDataset(examples)
    dataset.validate(raise_on_error=True)
    log(f"[data] {data_stats}")
    config = TrainingConfig(output_dir=str(out_dir), num_epochs=EPOCHS, batch_size=BATCH_SIZE, use_lora=True,
                            save_adapter_only=True, eval_strategy="no", num_workers=0, pin_memory=False,
                            seed=SEED, logging_steps=50, **({"max_steps": args.max_steps} if args.max_steps else {}))
    summary = ExtractorTrainer(model, config).train(train_data=dataset)
    training = {k: summary.get(k) for k in ("total_time_seconds", "samples_per_second", "total_steps")}
    log(f"[train] {training}")

    conditions, examples_out = {}, {}
    for name, adapter in (("zero_shot", None), ("adapter", out_dir / "final")):
        model = load_model(adapter)
        conditions[name] = {}
        for split in ("dev", "eval"):
            preds = predict(model, golds[split], ontology)
            scores = score(golds[split], preds, ontology, train_surfaces)
            conditions[name][split] = scores
            if split == "eval" and name == "adapter":
                examples_out = {"ledger": example_rows(splits["eval"], golds["eval"], preds, "ledger", 4),
                                "narrative": example_rows(splits["eval"], golds["eval"], preds, "narrative", 4)}
            log(f"[eval] {name}/{split}: ent {scores['entities']['overall']['f1']:.3f} "
                f"rel {scores['relations']['overall']['f1']:.3f} rec {scores['records']['overall']['f1']:.3f}")

    result = {
        "run": {"date": date.today().isoformat(), "base_model": BASE_MODEL,
                "device": f"{platform.system()} {platform.machine()} CPU", "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "seed": SEED,
                "docs": {k: len(v) for k, v in splits.items()},
                "scenarios": {k: len({d['scenario_id'] for d in v}) for k, v in splits.items()},
                "ledger_rows": {k: sum(len(d["paragraphs"]) for d in v if d.get("doc_kind") == "ledger")
                                for k, v in splits.items()}},
        "data_stats": data_stats, "training": training, "conditions": conditions, "examples": examples_out,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=list) + "\n")
    log(f"saved {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
