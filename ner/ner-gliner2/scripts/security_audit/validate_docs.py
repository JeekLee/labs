"""Validate documents and labels.

    uv run python scripts/security_audit/validate_docs.py                 # documents/ + labels/
    uv run python scripts/security_audit/validate_docs.py <file.jsonl>    # authoring JSONL (writers)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.security_audit.annotation import validate_corpus  # noqa: E402
from config.security_audit.ontology import load_ontology  # noqa: E402
from config.security_audit.storage import load_docs  # noqa: E402

DATA = ROOT / "experiments/data/security_audit"


def main() -> None:
    if sys.argv[1:]:
        docs = []
        for path in sys.argv[1:]:
            for n, line in enumerate(Path(path).read_text().splitlines(), 1):
                if line.strip():
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(f"{path}:{n}: invalid JSON: {exc}")
                        sys.exit(1)
    else:
        docs = load_docs(DATA)
    errors = validate_corpus(docs, load_ontology())
    for error in errors:
        print(error)
    print(f"{len(docs)} documents, {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
