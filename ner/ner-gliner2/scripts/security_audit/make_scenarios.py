"""Generate fictional security-audit scenario graphs (the gold graph each document is written from).

    uv run python scripts/security_audit/make_scenarios.py --n 30 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.security_audit import Ontology, load_ontology  # noqa: E402

DATA = ROOT / "experiments/data/security_audit"

# incident type -> (area, finding area, corrective action type, type-specific roles)
INCIDENTS = {
    "저장매체 무단 반출": ("정보통신보안", "정보통신보안", "매체 회수", ["medium", "document"]),
    "비인가 출입": ("시설보안", "시설보안", "시설 보완", ["facility_access"]),
    "비밀문서 관리 소홀": ("문서보안", "문서보안", "재교육", ["document"]),
    "계정·권한 관리 위반": ("정보통신보안", "정보통신보안", "권한 회수", ["system", "credential_account"]),
    "보안교육 미이수": ("인원보안", "인원보안", "재교육", []),
    "인가 만료자 접근": ("인원보안", "인원보안", "권한 회수", ["credential_clearance", "document"]),
}
REGULATION = {
    "저장매체 무단 반출": "저장매체 통제 지침", "비인가 출입": "출입 통제 규정",
    "비밀문서 관리 소홀": "비밀문서 관리 규정", "계정·권한 관리 위반": "정보체계 계정 관리 지침",
    "보안교육 미이수": "보안교육 시행 지침", "인가 만료자 접근": "비밀취급인가 관리 규정",
}
POSITION = {
    "저장매체 무단 반출": "정보보호 담당관", "비인가 출입": "출입통제 담당관", "비밀문서 관리 소홀": "비밀관리책임자",
    "계정·권한 관리 위반": "정보체계 관리자", "보안교육 미이수": "보안담당관", "인가 만료자 접근": "보안담당관",
}


def load_dictionary(path: Path = DATA / "dictionary.json") -> dict:
    return json.loads(Path(path).read_text())


def _scenario(index: int, split: str, incident_type: str, rng: random.Random, d: dict, used: set) -> dict:
    area, finding_area, action_type, roles = INCIDENTS[incident_type]
    entities, relations = [], []

    def add(kind: str, surface: str, **attrs) -> str:
        eid = f"e{len(entities) + 1}"
        entities.append({"id": eid, "type": kind, "surface": surface, "attrs": attrs})
        return eid

    def rel(head: str, kind: str, tail: str) -> None:
        relations.append({"head": head, "type": kind, "tail": tail})

    def fresh_name() -> str:
        name = rng.choice([n for n in d["names"] if n not in used])
        used.add(name)
        return name

    main_rank, officer_rank = rng.choice(d["enlisted_ranks"]), rng.choice(d["officer_ranks"])
    main_name, officer_name = fresh_name(), fresh_name()
    main = add("Person", f"{main_name} {main_rank}", name=main_name, rank=main_rank)
    officer = add("Person", f"{officer_name} {officer_rank}", name=officer_name, rank=officer_rank)
    unit = add("Unit", rng.choice(d["units"]))
    position = add("Position", POSITION[incident_type])
    facility = add("Facility", rng.choice(d["facilities"]))
    regulation = add("Regulation", REGULATION[incident_type])
    quarter = rng.randint(1, 4)
    inspection = add("Inspection", f"2026년 {quarter}분기 정기 보안점검")
    rel(main, "BELONGS_TO", unit)
    rel(officer, "BELONGS_TO", unit)
    rel(officer, "HOLDS_POSITION", position)
    rel(position, "POSITION_OF", unit)
    rel(officer, "RESPONSIBLE_FOR", facility)
    rel(inspection, "INSPECTED", unit)

    incident = {"유형": incident_type, "관련자": [main], "매체": [], "대상 문서": [], "장소": [facility],
                "위반 규정": [regulation], "일자": f"2026. {quarter * 3 - 2}. {rng.randint(1, 28)}."}
    targets = [main]
    if "medium" in roles:
        medium = add("StorageMedium", rng.choice(d["media"]).format(n=100 + index))
        rel(main, "CARRIED_OUT", medium)
        rel(officer, "RESPONSIBLE_FOR", medium)
        incident["매체"].append(medium)
        targets.append(medium)
    if "document" in roles:
        document = add("ClassifiedDocument", rng.choice(d["documents"]))
        if "medium" in roles:
            rel(document, "STORED_ON", medium)
        else:
            rel(main, "ACCESSED", document)
        incident["대상 문서"].append(document)
        targets.append(document)
    if "facility_access" in roles:
        rel(main, "ACCESSED", facility)
    if "system" in roles:
        system = add("InformationSystem", rng.choice(d["systems"]).format(n=10 + index))
        rel(main, "ACCESSED", system)
        rel(system, "LOCATED_AT", facility)
        rel(officer, "RESPONSIBLE_FOR", system)
        incident["장소"] = [system]
        targets.append(system)
    if "credential_account" in roles:
        credential = add("Credential", d["credentials"][2].format(id=f"u{2000 + index}"))
        rel(main, "HOLDS", credential)
    if "credential_clearance" in roles:
        credential = add("Credential", d["credentials"][0])
        rel(main, "HOLDS", credential)
    if not roles:
        targets.append(position)

    doc_types = ["보안사고 경위서", "보안점검 결과보고서"]
    pool = [t for t in d["doc_types_train"] if t not in doc_types]
    doc_types += rng.sample(pool, rng.randint(2, 3))
    if split == "eval":
        doc_types.append(d["doc_type_eval_only"])
    return {
        "scenario_id": f"{split}-{index:03d}", "split": split, "area": area, "incident_type": incident_type,
        "entities": entities, "relations": relations, "incident": incident,
        "findings": [{"영역": finding_area, "대상": targets, "위반 규정": [regulation]}],
        "actions": [{"조치 유형": action_type, "담당": [officer], "기한": "2026. 12. 31."}],
        "doc_types": doc_types,
    }


def make_scenarios(n: int, seed: int, dictionary: dict, ontology: Ontology) -> list[dict]:
    rng = random.Random(seed)
    types = list(INCIDENTS)
    n_eval = max(1, n // 5)
    used: set = set()
    scenarios = []
    for i in range(n):
        split = "eval" if i >= n - n_eval else "train"
        scenarios.append(_scenario(i, split, types[i % len(types)], rng, dictionary, used))
    choices = {f.name: f for f in ontology.records["incident"].fields}["유형"].choices
    assert all(s["incident_type"] in choices for s in scenarios)
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    scenarios = make_scenarios(args.n, args.seed, load_dictionary(), load_ontology())
    out = DATA / "scenarios.jsonl"
    out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scenarios))
    print(f"wrote {len(scenarios)} scenarios to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
