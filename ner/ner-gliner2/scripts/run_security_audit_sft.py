"""Experiment 04 (phase 1 pilot): SFT a GLiNER2 LoRA adapter on the fictional security-audit ontology.

    uv run python scripts/run_security_audit_sft.py              # pilot run
    uv run python scripts/run_security_audit_sft.py --limit 5 --max-steps 4   # smoke run
"""

import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

from config.annotation import validate_corpus  # noqa: E402
from config.extractors import kiwi_splitter  # noqa: E402
from config.security_audit import load_ontology  # noqa: E402
from config.security_audit_data import (paragraph_examples, record_examples, split_docs,  # noqa: E402
                                        to_training_examples)
from config.security_audit_eval import (choice_literal_rate, incident_description_scores,  # noqa: E402
                                        predict_entities, predict_records, predict_relations, score_entities,
                                        score_entities_by_seen, score_records, score_relations)

BASE_MODEL = "fastino/gliner2.5-multi-v1"
DOCS = ROOT / "experiments/data/security_audit/docs"
HELD_OUT_DOC_TYPE = "보안대책 회의록"
EPOCHS, BATCH_SIZE, SEED = 2, 8, 42


def log(message: str) -> None:
    print(message, flush=True)


def load_docs() -> list[dict]:
    return [json.loads(line) for path in sorted(DOCS.glob("*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def golds_for(docs: list[dict], ontology):
    paragraphs, records, p_meta, r_meta = [], [], [], []
    for doc in docs:
        meta = {"doc_id": doc["doc_id"], "doc_type": doc["doc_type"], "area": (doc.get("area_tags") or ["?"])[0]}
        for p in paragraph_examples(doc, ontology):
            paragraphs.append(p)
            p_meta.append(meta)
        for r in record_examples(doc, ontology):
            records.append(r)
            r_meta.append(meta)
    return paragraphs, records, p_meta, r_meta


def load_model(adapter: Path | None = None):
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(BASE_MODEL)
    if adapter is not None:
        model.load_adapter(str(adapter))
    model.set_word_splitter(kiwi_splitter)
    return model


def evaluate(model, golds, ontology, train_surfaces: set) -> tuple[dict, dict]:
    paragraphs, records, _, _ = golds
    labels = ontology.extract_labels()
    ent_preds, ent_seconds = predict_entities(model, paragraphs, labels)
    rel_preds, rel_seconds = predict_relations(model, paragraphs, ontology)
    rec_preds, rec_seconds = predict_records(model, records, ontology)
    scores = {
        "entities": score_entities(ent_preds, paragraphs, list(labels)),
        "entities_by_seen": score_entities_by_seen(ent_preds, paragraphs, train_surfaces, list(labels)),
        "relations": score_relations(rel_preds, paragraphs),
        "records": score_records(rec_preds, records, ontology),
        "seconds": {"entities": ent_seconds, "relations": rel_seconds, "records": rec_seconds},
        "paragraphs": len(paragraphs), "records_n": len(records),
    }
    return scores, {"entities": ent_preds, "relations": rel_preds, "records": rec_preds}


def grouped(golds, preds, ontology, key: str) -> dict:
    paragraphs, records, p_meta, r_meta = golds
    labels = list(ontology.extract_labels())
    out = defaultdict(dict)
    for value in sorted({m[key] for m in p_meta + r_meta}):
        pi = [i for i, m in enumerate(p_meta) if m[key] == value]
        ri = [i for i, m in enumerate(r_meta) if m[key] == value]
        if pi:
            out[value]["entities_f1"] = score_entities([preds["entities"][i] for i in pi],
                                                       [paragraphs[i] for i in pi], labels)["overall"]["f1"]
            out[value]["relations_f1"] = score_relations([preds["relations"][i] for i in pi],
                                                         [paragraphs[i] for i in pi])["overall"]["f1"]
        if ri:
            rec = score_records([preds["records"][i] for i in ri], [records[i] for i in ri], ontology)
            out[value]["record_fields_f1"] = rec["overall"]["f1"]
            out[value]["choice_accuracy"] = rec["choice_overall"]
        out[value]["paragraphs"], out[value]["records"] = len(pi), len(ri)
    return dict(out)


def graph_examples(docs, ontology, model, limit: int = 3) -> list[dict]:
    from config.security_audit_eval import predict_relations as rel

    examples = []
    for doc in [d for d in docs if d["doc_type"] == "보안사고 경위서"][:limit]:
        paragraphs = paragraph_examples(doc, ontology)
        ent, _ = predict_entities(model, paragraphs, ontology.extract_labels())
        rels, _ = rel(model, paragraphs, ontology)
        examples.append({
            "doc_id": doc["doc_id"], "doc_type": doc["doc_type"],
            "text": "\n".join(p.text for p in paragraphs),
            "gold": {"entities": sorted({(p.text[s:e], t) for p in paragraphs for s, e, t in p.spans}),
                     "relations": sorted({r for p in paragraphs for r in p.relations}),
                     "records": [r.fields for r in record_examples(doc, ontology)]},
            "predicted": {"entities": sorted({(p.text[s:e], t) for p, ps in zip(paragraphs, ent) for s, e, t in ps}),
                          "relations": sorted({r for rs in rels for r in rs}),
                          "records": predict_records(model, record_examples(doc, ontology), ontology)[0]},
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="smoke run: N docs per split")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    ontology = load_ontology()
    docs = load_docs()
    errors = validate_corpus(docs, ontology)
    if errors:
        raise SystemExit(f"{len(errors)} validation errors, e.g. {errors[:3]}")
    splits = split_docs(docs)
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    out_dir = ROOT / ("outputs/security_audit_lora_smoke" if args.limit else "outputs/security_audit_lora_pilot")
    result_path = (out_dir / "result.json") if args.limit else ROOT / "experiments/results/security_audit_sft_pilot.json"
    golds = {name: golds_for(split, ontology) for name, split in splits.items()}
    train_surfaces = {(p.text[s:e], t) for p in golds["train"][0] for s, e, t in p.spans}
    log(f"[data] docs " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    model = load_model()
    processor = model.processor

    def unalignable(text, values):
        if not values:
            return set()
        return {item["value"] for item in processor.find_unalignable_entities(text, {"value": list(values)})}

    train_p, train_r, _, _ = golds["train"]
    examples, data_stats = to_training_examples(train_p, train_r, ontology, unalignable)
    dataset = TrainingDataset(examples)
    dataset.validate(raise_on_error=True)
    log(f"[data] {data_stats}")
    config = TrainingConfig(output_dir=str(out_dir), num_epochs=EPOCHS, batch_size=BATCH_SIZE, use_lora=True,
                            save_adapter_only=True, eval_strategy="no", num_workers=0, pin_memory=False,
                            seed=SEED, logging_steps=50, **({"max_steps": args.max_steps} if args.max_steps else {}))
    summary = ExtractorTrainer(model, config).train(train_data=dataset)
    training = {k: summary.get(k) for k in ("total_time_seconds", "samples_per_second", "total_steps")}
    log(f"[train] {training}")

    conditions, preds_eval = {}, {}
    for name, adapter in (("zero_shot", None), ("adapter", out_dir / "final")):
        model = load_model(adapter)
        conditions[name] = {}
        for split in ("dev", "eval"):
            scores, preds = evaluate(model, golds[split], ontology, train_surfaces)
            conditions[name][split] = scores
            if split == "eval":
                preds_eval[name] = preds
            log(f"[eval] {name}/{split}: ent {scores['entities']['overall']['f1']:.3f} "
                f"rel {scores['relations']['overall']['f1']:.3f} rec {scores['records']['overall']['f1']:.3f} "
                f"choice {scores['records']['choice_overall']:.3f}")
        conditions[name]["incident_description"] = incident_description_scores(model, splits["eval"])

    eval_golds = golds["eval"]
    held_idx_p = [i for i, m in enumerate(eval_golds[2]) if m["doc_type"] == HELD_OUT_DOC_TYPE]
    held_idx_r = [i for i, m in enumerate(eval_golds[3]) if m["doc_type"] == HELD_OUT_DOC_TYPE]
    held_out = {}
    for name in conditions:
        p, r = preds_eval[name], eval_golds
        held_out[name] = {
            "entities_f1": score_entities([p["entities"][i] for i in held_idx_p], [r[0][i] for i in held_idx_p],
                                          list(ontology.extract_labels()))["overall"]["f1"] if held_idx_p else None,
            "relations_f1": score_relations([p["relations"][i] for i in held_idx_p],
                                            [r[0][i] for i in held_idx_p])["overall"]["f1"] if held_idx_p else None,
            "records": score_records([p["records"][i] for i in held_idx_r], [r[1][i] for i in held_idx_r],
                                     ontology) if held_idx_r else None,
            "records_n": len(held_idx_r), "paragraphs": len(held_idx_p)}

    result = {
        "run": {"date": date.today().isoformat(), "base_model": BASE_MODEL, "device": f"{platform.system()} "
                f"{platform.machine()} CPU", "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED,
                "docs": {k: len(v) for k, v in splits.items()},
                "scenarios": {k: len({d['scenario_id'] for d in v}) for k, v in splits.items()}},
        "data_stats": data_stats, "training": training, "conditions": conditions,
        "choice_literal_rate": {split: choice_literal_rate(golds[split][1], ontology) for split in golds},
        "eval_seen_gold": {"seen": conditions["adapter"]["eval"]["entities_by_seen"]["seen_gold"],
                           "unseen": conditions["adapter"]["eval"]["entities_by_seen"]["unseen_gold"]},
        "by_doc_type": {name: grouped(eval_golds, preds_eval[name], ontology, "doc_type") for name in conditions},
        "by_area": {name: grouped(eval_golds, preds_eval[name], ontology, "area") for name in conditions},
        "held_out_doc_type": {"doc_type": HELD_OUT_DOC_TYPE, **held_out},
        "examples": graph_examples(splits["eval"], ontology, load_model(out_dir / "final")),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=list) + "\n")
    log(f"saved {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
