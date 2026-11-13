"""Convert writer output (annotated JSONL) into Markdown documents + label JSON, then remove the JSONL.

    uv run python scripts/security_audit/import_written.py experiments/data/security_audit/documents/*/narrative_*.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.security_audit.storage import write_doc  # noqa: E402

DATA = ROOT / "experiments/data/security_audit"


def main() -> None:
    for path in map(Path, sys.argv[1:]):
        docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for doc in docs:
            doc.setdefault("doc_kind", "narrative")
            write_doc(doc, DATA)
        path.unlink()
        print(f"{path.name}: {len(docs)} documents imported")


if __name__ == "__main__":
    main()
