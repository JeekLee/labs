"""Render every scenario's ledgers to Markdown documents and label JSON (gold recorded at render time).

    uv run python scripts/security_audit/render_ledgers.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.security_audit.forms import FORMS_DIR, load_forms  # noqa: E402
from config.security_audit.storage import write_doc  # noqa: E402
from config.security_audit.tables import render_ledger  # noqa: E402

DATA = ROOT / "experiments/data/security_audit"


def main() -> None:
    forms = load_forms(FORMS_DIR)
    scenarios = [json.loads(line) for line in (DATA / "scenarios.jsonl").read_text().splitlines() if line.strip()]
    by_split: dict[str, list[dict]] = {}
    for s in scenarios:
        for n, ledger in enumerate(s["ledgers"], 1):
            form = forms[ledger["record"]]
            variant = next(v for v in form.variants if v.id == ledger["variant_id"])
            doc = render_ledger(form, variant, ledger["meta"], ledger["rows"], doc_id=f"{s['scenario_id']}-L{n}",
                                scenario_id=s["scenario_id"], split=s["split"], area=s["area"])
            by_split.setdefault(s["split"], []).append(doc)
    for split, docs in by_split.items():
        for doc in docs:
            write_doc(doc, DATA)
        print(f"{split}: {len(docs)} ledgers, {sum(len(d['paragraphs']) for d in docs)} rows -> documents/{split}/*.md + labels/{split}/*.json")

if __name__ == "__main__":
    main()
