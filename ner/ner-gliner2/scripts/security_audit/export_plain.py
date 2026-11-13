"""Export the plain text GLiNER reads (annotation marks removed) as one Markdown file per document.

    uv run python scripts/security_audit/export_plain.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.annotation import parse_marked  # noqa: E402

DATA = ROOT / "experiments/data/security_audit"


def main() -> None:
    out = DATA / "plain"
    count = 0
    for path in sorted((DATA / "docs").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            target = out / doc["split"] / f"{doc['doc_id']}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            body = "\n\n".join(parse_marked(p)[0] for p in doc["paragraphs"])
            target.write_text(
                f"# {doc['doc_type']}\n\n"
                f"> 가상 문서 · 시나리오 `{doc['scenario_id']}` · {doc['split']} · "
                f"영역 {', '.join(doc.get('area_tags', []))} · 원본 `{path.relative_to(ROOT)}`\n\n"
                f"{body}\n")
            count += 1
    print(f"exported {count} documents to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
