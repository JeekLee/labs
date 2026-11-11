"""Experiment 03: distill gpt-5.4-mini labels into a GLiNER2 LoRA adapter and compare on KLUE dev.

Stages are cached under outputs/distill/ (or outputs/distill_smoke/ with --limit), so an
interrupted run resumes without repeating API calls or training.

    uv run python scripts/run_llm_distillation.py --price-input 0.75 --price-output 4.5 \
        --price-source https://developers.openai.com/api/docs/pricing
"""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
import os
import platform
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

from dotenv import load_dotenv  # noqa: E402
from huggingface_hub.utils import logging as hf_logging  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

hf_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

from config.extractors import kiwi_splitter  # noqa: E402
from config.klue import (LABELS, classify_errors, gliner_predict, is_aligned, load_klue,  # noqa: E402
                         sample_train, score, to_example, to_input_example)
from config.teacher import TeacherLabeler, locate_spans  # noqa: E402

BASE_MODEL = "fastino/gliner2.5-multi-v1"
KLUE_ADAPTER = ROOT / "outputs/klue_ner_lora/final"
TRAIN_SIZE, EPOCHS, BATCH_SIZE, SEED = 2000, 2, 8, 42
CONDITIONS = ("base", "klue_lora", "distill_lora")


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


def stage_label(texts: list[str], out: Path, model: str, concurrency: int) -> list[dict]:
    cache = out / "train_labels.jsonl"
    cached = {json.loads(line)["text"] for line in cache.read_text().splitlines()} if cache.exists() else set()
    todo = len(set(texts) - cached)
    log(f"[label] {len(texts)} sentences, {len(texts) - todo} cached, {todo} new sentences to call")
    records = asyncio.run(TeacherLabeler(model, concurrency=concurrency).label_many(texts, cache))
    for record in records:  # re-place spans with the current locate rule; no API call needed
        spans, record["dropped"] = locate_spans(record["text"], record["entities"])
        record["spans"] = [list(span) for span in spans]
    return records


def labeling_time(records: list[dict], out: Path) -> tuple[float, int]:
    """Wall seconds and API attempts measured from per-record timestamps.

    Records written before timestamps existed fall back to the legacy labeling_meta.json.
    """
    runs: dict[str, list[dict]] = {}
    for record in records:
        if record.get("run_id"):
            runs.setdefault(record["run_id"], []).append(record)
    wall = sum(max(r["finished_at"] for r in rs) - min(r["started_at"] for r in rs) for rs in runs.values())
    if any(not r.get("run_id") for r in records):
        wall += read_json(out / "labeling_meta.json", {"wall_seconds": 0.0})["wall_seconds"]
    return wall, sum(r.get("attempts", 1) for r in records)


def label_hash(records: list[dict]) -> str:
    payload = json.dumps([[r["text"], r["spans"]] for r in records if r["error"] is None], ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()


def stage_train(records: list[dict], out: Path) -> dict:
    meta_path = out / "training_meta.json"
    labels = label_hash(records)
    if (out / "lora/final").exists() and read_json(meta_path, {}).get("label_hash") == labels:
        log("[train] cached adapter")
        return read_json(meta_path, {})
    from gliner2 import AutoExtractor
    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    usable = [r for r in records if r["error"] is None]
    examples = [to_input_example(r["text"], [tuple(s) for s in r["spans"]]) for r in usable]
    dataset = TrainingDataset(examples)
    dataset.validate(raise_on_error=True)
    model = AutoExtractor.from_pretrained(BASE_MODEL)
    model.set_word_splitter(kiwi_splitter)
    config = TrainingConfig(output_dir=str(out / "lora"), num_epochs=EPOCHS, batch_size=BATCH_SIZE,
                            use_lora=True, save_adapter_only=True, eval_strategy="no",
                            num_workers=0, pin_memory=False, seed=SEED, logging_steps=50)
    log(f"[train] {len(examples)} examples")
    summary = ExtractorTrainer(model, config).train(train_data=dataset)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {"examples": len(examples), "seconds": summary.get("total_time_seconds"),
            "samples_per_second": summary.get("samples_per_second"),
            "total_steps": summary.get("total_steps"), "trainable_params": trainable or None,
            "label_hash": labels}
    write_json(meta_path, meta)
    return meta


def stage_predict(eval_texts: list[str], out: Path, distill_hash: str) -> dict:
    from gliner2 import AutoExtractor

    path = out / "predictions.json"
    cache = read_json(path, {})
    adapters = {"base": None, "klue_lora": KLUE_ADAPTER, "distill_lora": out / "lora/final"}
    for name in CONDITIONS:
        stale = name == "distill_lora" and cache.get(name, {}).get("label_hash") != distill_hash
        if name in cache and not stale:
            log(f"[predict] {name}: cached")
            continue
        model = AutoExtractor.from_pretrained(BASE_MODEL)
        if adapters[name] is not None:
            model.load_adapter(str(adapters[name]))
        model.set_word_splitter(kiwi_splitter)
        predictions, seconds = gliner_predict(model, eval_texts)
        cache[name] = {"predictions": [sorted(map(list, p)) for p in predictions], "seconds": seconds,
                       "label_hash": distill_hash if name == "distill_lora" else None}
        write_json(path, cache)
        log(f"[predict] {name}: {seconds:.0f}s")
    return {name: ([{tuple(s) for s in p} for p in cache[name]["predictions"]], cache[name]["seconds"])
            for name in CONDITIONS}


def surfaces(text: str, spans) -> list[list[str]]:
    return [[text[s:e], label] for s, e, label in sorted(spans)]


def pick_examples(eval_data, predicted: dict, limit: int = 8) -> list[dict]:
    """Sentences where KLUE and distilled adapters disagree: half distill right, half distill wrong."""
    distill_right, distill_wrong = [], []
    for i, (text, gold) in enumerate(eval_data):
        gold_set = set(gold)
        klue, distill = predicted["klue_lora"][0][i], predicted["distill_lora"][0][i]
        if klue == distill or not gold or len(text) > 90:
            continue
        (distill_right if distill == gold_set else distill_wrong if klue == gold_set else []).append(i)
    chosen = distill_right[: limit // 2] + distill_wrong[: limit - min(len(distill_right), limit // 2)]
    return [{"text": eval_data[i][0], "gold": surfaces(eval_data[i][0], eval_data[i][1]),
             **{name: surfaces(eval_data[i][0], predicted[name][0][i]) for name in CONDITIONS}}
            for i in chosen[:limit]]


def disagreements(predictions, data) -> tuple[Counter, Counter]:
    """(label, surface) counts of predictions not in gold (fp) and gold not predicted (fn)."""
    fp, fn = Counter(), Counter()
    for predicted, (text, gold) in zip(predictions, data, strict=True):
        gold = set(gold)
        fp.update((label, text[s:e]) for s, e, label in predicted - gold)
        fn.update((label, text[s:e]) for s, e, label in gold - predicted)
    return fp, fn


def inherited_share(predictions, data, teacher_fp: Counter, teacher_fn: Counter) -> dict:
    """Share of error instances whose (label, surface) the teacher also got wrong on train."""
    fp, fn = disagreements(predictions, data)
    def share(errors: Counter, reference: Counter) -> float:
        total = sum(errors.values())
        return sum(n for key, n in errors.items() if key in reference) / total if total else 0.0
    return {"fp": sum(fp.values()), "fp_inherited": share(fp, teacher_fp),
            "fn": sum(fn.values()), "fn_inherited": share(fn, teacher_fn)}


def top(counter: Counter, n: int = 15) -> list[list]:
    return [[label, surface, count] for (label, surface), count in counter.most_common(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="smoke run with N train and N dev sentences")
    parser.add_argument("--price-input", type=float, required=True, help="USD per 1M input tokens")
    parser.add_argument("--price-output", type=float, required=True, help="USD per 1M output tokens")
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model_name = os.environ["LLM_MODEL"]
    out = ROOT / ("outputs/distill_smoke" if args.limit else "outputs/distill")
    result_path = out / "llm_distillation.json" if args.limit else ROOT / "experiments/results/llm_distillation.json"

    train_rows, tag_names = load_klue("train")
    eval_rows, _ = load_klue("validation")
    train_data = [to_example(row, tag_names) for row in sample_train(train_rows, TRAIN_SIZE, SEED)]
    eval_data = [to_example(row, tag_names) for row in eval_rows]
    if args.limit:
        train_data, eval_data = train_data[: args.limit], eval_data[: args.limit]

    records = stage_label([t for t, _ in train_data], out, model_name, args.concurrency)
    wall_seconds, api_attempts = labeling_time(records, out)
    teacher_train = [{tuple(s) for s in r["spans"]} for r in records]
    ok = [r for r in records if r["error"] is None]
    input_tokens = sum(r["input_tokens"] for r in records)
    output_tokens = sum(r["output_tokens"] for r in records)
    cost = (input_tokens * args.price_input + output_tokens * args.price_output) / 1e6
    total_spans = sum(len(r["spans"]) for r in ok)
    labeling = {
        "sentences": len(records), "failed": len(records) - len(ok),
        "dropped": sum(r["dropped"] for r in records), "total_spans": total_spans,
        "aligned": sum(is_aligned(r["text"], s, e) for r in ok for s, e, _ in r["spans"]),
        "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost,
        "cost_per_sentence_usd": cost / len(records),
        "api_attempts": api_attempts, "wall_seconds": wall_seconds,
        "throughput_seconds_per_sentence": wall_seconds / len(records),
        "mean_call_seconds": sum(r["seconds"] for r in records) / len(records),
    }
    log(f"[label] cost ${cost:.4f}, failed {labeling['failed']}, dropped {labeling['dropped']}")

    training = stage_train(records, out)
    predicted = stage_predict([t for t, _ in eval_data], out, label_hash(records))
    teacher_fp, teacher_fn = disagreements(teacher_train, train_data)

    conditions = {name: {**score(preds, eval_data), "seconds": seconds,
                         "seconds_per_sentence": seconds / len(eval_data)}
                  for name, (preds, seconds) in predicted.items()}
    result = {
        "run": {"date": date.today().isoformat(), "model": model_name, "base_model": BASE_MODEL,
                "price": {"input_per_mtok": args.price_input, "output_per_mtok": args.price_output,
                          "source": args.price_source},
                "device": f"{platform.system()} {platform.machine()} CPU",
                "train_size": len(train_data), "eval_size": len(eval_data),
                "eval_entities": sum(len(s) for _, s in eval_data), "labels": LABELS},
        "teacher_vs_human_train": score(teacher_train, train_data),
        "labeling": labeling,
        "training": training,
        "conditions": conditions,
        "errors": {"klue_lora": classify_errors(predicted["klue_lora"][0], eval_data),
                   "distill_lora": classify_errors(predicted["distill_lora"][0], eval_data),
                   "teacher_train": classify_errors(teacher_train, train_data)},
        "teacher_disagreements": {"fp_top": top(teacher_fp), "fn_top": top(teacher_fn),
                                  "fp": sum(teacher_fp.values()), "fn": sum(teacher_fn.values())},
        # Control: the KLUE adapter's share shows how often surfaces overlap by chance.
        "inherited_errors": {name: inherited_share(predicted[name][0], eval_data, teacher_fp, teacher_fn)
                             for name in ("klue_lora", "distill_lora")},
        "examples": pick_examples(eval_data, predicted),
        # One-time labeling cost divided by the LLM cost of one inference sentence:
        # after this many sentences, running the distilled GLiNER instead of the LLM pays off.
        "breakeven_sentences": round(cost / labeling["cost_per_sentence_usd"]),
    }
    write_json(result_path, result)
    for name, scores in conditions.items():
        log(f"{name:>13}: F1 {scores['overall']['f1']:.3f}  P {scores['overall']['precision']:.3f}"
            f"  R {scores['overall']['recall']:.3f}  {scores['seconds']:.0f}s")
    log(f"teacher(train): F1 {result['teacher_vs_human_train']['overall']['f1']:.3f}")
    log(f"saved {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
