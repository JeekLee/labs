"""Generate fictional security-audit scenarios (v2): incident graph + ledger rows per scenario.

    uv run python scripts/security_audit/make_scenarios.py --train 45 --eval 15 --seed 42

Train scenarios draw every surface from the train dictionary and only training column variants.
Eval scenarios draw about half their surfaces from the eval dictionary and render about half
their ledgers with the eval-only variant, so seen and unseen buckets can both be measured.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.security_audit.forms import FORMS_DIR, Form, load_forms  # noqa: E402
from config.security_audit.ontology import Ontology, load_ontology  # noqa: E402

DATA = ROOT / "experiments/data/security_audit"

INCIDENTS = {  # incident type -> (area, ledgers, regulation index, position)
    "저장매체 무단 반출": ("정보통신보안", ["medium_transfer", "medium_entry"], 0, "정보보호 담당관"),
    "비인가 출입": ("시설보안", ["entry", "daily_check"], 1, "출입통제 담당관"),
    "비밀문서 관리 소홀": ("문서보안", ["document_entry", "loan", "reading"], 2, "비밀관리책임자"),
    "계정·권한 관리 위반": ("정보통신보안", ["system_entry", "medium_entry"], 3, "정보체계 관리자"),
    "보안교육 미이수": ("인원보안", ["custodian", "clearance"], 4, "보안담당관"),
    "인가 만료자 접근": ("인원보안", ["clearance", "reading", "loan"], 5, "보안담당관"),
}
EXTRA_LEDGERS = ["daily_check", "system_entry", "custodian", "entry"]
ACTION_TYPE = {"저장매체 무단 반출": "매체 회수", "비인가 출입": "시설 보완", "비밀문서 관리 소홀": "재교육",
               "계정·권한 관리 위반": "권한 회수", "보안교육 미이수": "재교육", "인가 만료자 접근": "권한 회수"}
OFFICER_ROLES = {"확인자", "입회자", "보관책임자", "처리자", "수령 확인관", "반납 확인관", "점검자"}
INCIDENT_BLANKABLE = {"확인자", "입회자", "반납일자", "반납 확인관", "퇴청시간", "해제일"}


def load_dict(split: str) -> dict:
    return json.loads((DATA / "dictionary" / f"{split}.json").read_text())


def load_dicts() -> dict:
    return {name: load_dict(name) for name in ("train", "eval", "common")}


def V(surface, kind=None):
    return {"surface": surface, "type": kind}


class Sampler:
    """Draws surfaces for one scenario; eval scenarios mix eval and train dictionaries."""

    def __init__(self, rng: random.Random, dicts: dict, split: str):
        self.rng, self.d, self.split = rng, dicts, split

    def source(self) -> dict:
        if self.split == "eval" and self.rng.random() < 0.5:
            return self.d["eval"]
        return self.d["train"]

    def person(self, officer: bool) -> dict:
        name = self.rng.choice(self.source()["names"])
        rank = self.rng.choice(self.d["common"]["officer_ranks" if officer else "enlisted_ranks"])
        surface = f"{name} {rank}" if self.rng.random() < 0.7 else f"{rank} {name}"
        return V(surface, "Person")

    def pick(self, kind: str, entity_type: str) -> dict:
        return V(self.rng.choice(self.source()[kind]), entity_type)

    def numbered(self, prefix_key: str, number_key: str, entity_type: str, width: int = 0) -> dict:
        src = self.source()
        low, high = src[number_key]
        number = str(self.rng.randint(low, high)).zfill(width)
        return V(f"{self.rng.choice(src[prefix_key])}{number}", entity_type)

    def credential(self) -> dict:
        src = self.source()
        low, high = src["credential_numbers"]
        return V(f"{src['credential_prefix']}{self.rng.randint(low, high)}호", "Credential")


def _date(rng: random.Random, month: int) -> str:
    return f"2026-{month:02d}-{rng.randint(1, 28):02d}"


def _time(rng: random.Random) -> str:
    return f"{rng.randint(8, 18):02d}:{rng.choice(['00', '10', '20', '30', '40', '50'])}"


def _row(form: Form, rng: random.Random, s: Sampler, ctx: dict, ontology: Ontology, incident: bool) -> dict:
    common, choices = s.d["common"], {f.name: f.choices for f in ontology.records[form.record].fields}
    row = {}
    for col in form.columns:
        kind, key = col.kind, col.key
        if kind == "person":
            if key in OFFICER_ROLES:
                row[key] = ctx["officer"] if incident or rng.random() < 0.6 else rng.choice(ctx["officers"])
            else:
                row[key] = ctx["actor"] if incident else rng.choice(ctx["users"])
        elif kind == "unit":
            row[key] = ctx["unit"] if incident or rng.random() < 0.7 else rng.choice(ctx["units"])
        elif kind == "date":
            row[key] = V(_date(rng, ctx["month"]))
        elif kind == "datetime":
            row[key] = V(f"{_date(rng, ctx['month'])} {_time(rng)}")
        elif kind == "time":
            row[key] = V(_time(rng))
        elif kind == "doc_id":
            row[key] = V(f"26-{rng.randint(1, 99):02d}")
        elif kind == "doc_title":
            row[key] = ctx["document"] if incident else rng.choice(ctx["documents"])
        elif kind == "grade":
            row[key] = V(rng.choice(["Ⅰ급", "Ⅱ급", "Ⅲ급"]))
        elif kind == "facility":
            row[key] = ctx["facility"] if incident else rng.choice(ctx["facilities"])
        elif kind == "choice":
            row[key] = V(rng.choice(choices[col.field]))
        elif kind == "clearance_id":
            row[key] = s.credential()
        elif kind == "medium_id":
            row[key] = ctx["medium"] if incident else rng.choice(ctx["media"])
        elif kind == "medium_kind":
            row[key] = V(rng.choice(common["medium_kinds"]))
        elif kind == "system":
            row[key] = ctx["system"] if incident else rng.choice(ctx["systems"])
        elif kind == "system_kind":
            row[key] = V(rng.choice(common["system_kinds"]))
        elif kind == "text_purpose":
            row[key] = V(rng.choice(common["purposes"]))
        elif kind == "text_release":
            row[key] = V(rng.choice(common["release_reasons"]) if rng.random() < 0.4 else "")
        elif kind == "text_check":
            row[key] = V(rng.choice(common["check_notes"]))
        elif kind == "text_dispose":
            row[key] = V(rng.choice(common["dispose_methods"]))
        elif kind == "text_inspect":
            row[key] = V(rng.choice(common["inspect_notes"]))
        elif kind == "text":
            row[key] = V(rng.choice(common["remarks"]))
        elif kind == "number":
            row[key] = V("")
        else:
            raise ValueError(f"unknown column kind {kind}")
        if incident and key in INCIDENT_BLANKABLE and rng.random() < 0.5:
            row[key] = V("")
    # Keep dates (and times) chronological in column order, e.g. 인가일 <= 해제일, 출입 <= 퇴청.
    for kinds in ({"date", "datetime"}, {"time"}):
        keys = [c.key for c in form.columns if c.kind in kinds and row[c.key]["surface"]]
        for key, surface in zip(keys, sorted(row[k]["surface"] for k in keys)):
            row[key] = V(surface)
    return row


def _meta(form: Form, ctx: dict) -> dict:
    meta = {"부대": ctx["unit"]}
    for name in form.meta_fields:
        meta[name] = {"관리책임자": ctx["officer"], "보관책임자": ctx["officer"], "구역": ctx["facility"],
                      "관리번호": V("26-07"), "건명": ctx["document"]}[name]
    return meta


def _scenario(index: int, split: str, incident_type: str, rng: random.Random, dicts: dict,
              forms: dict[str, Form], ontology: Ontology) -> dict:
    s = Sampler(rng, dicts, split)
    area, ledger_names, regulation_index, position = INCIDENTS[incident_type]
    ctx = {"month": rng.randint(1, 11)}
    ctx["unit"] = s.pick("units", "Unit")
    ctx["units"] = [s.pick("units", "Unit") for _ in range(3)]
    ctx["actor"], ctx["officer"] = s.person(False), s.person(True)
    ctx["users"] = [s.person(False) for _ in range(5)]
    ctx["officers"] = [s.person(True) for _ in range(2)]
    ctx["facility"] = s.pick("facilities", "Facility")
    ctx["facilities"] = [s.pick("facilities", "Facility") for _ in range(3)]
    ctx["document"] = s.pick("documents", "ClassifiedDocument")
    ctx["documents"] = [s.pick("documents", "ClassifiedDocument") for _ in range(4)]
    ctx["medium"] = s.numbered("medium_prefixes", "medium_numbers", "StorageMedium")
    ctx["media"] = [s.numbered("medium_prefixes", "medium_numbers", "StorageMedium") for _ in range(4)]
    ctx["system"] = s.numbered("system_prefixes", "system_numbers", "InformationSystem", 2)
    ctx["systems"] = [s.numbered("system_prefixes", "system_numbers", "InformationSystem", 2) for _ in range(3)]
    regulation = V((s.source())["regulations"][regulation_index], "Regulation")

    names = list(ledger_names)
    extra = [n for n in EXTRA_LEDGERS if n not in names]
    if len(names) < 4 and rng.random() < 0.5:
        names.append(rng.choice(extra))
    ledgers = []
    for name in names:
        form = forms[name]
        train_variants = [v for v in form.variants if not v.eval_only]
        eval_variant = next(v for v in form.variants if v.eval_only)
        variant = eval_variant if split == "eval" and rng.random() < 0.5 else rng.choice(train_variants)
        rows = [_row(form, rng, s, ctx, ontology, incident=False) for _ in range(rng.randint(5, 7))]
        for _ in range(rng.randint(1, 2)):
            rows.insert(rng.randint(0, len(rows)), _row(form, rng, s, ctx, ontology, incident=True))
        ledgers.append({"record": name, "variant_id": variant.id, "meta": _meta(form, ctx), "rows": rows})

    entities, seen = [], set()
    def collect(value):
        if value and value.get("type") and value["surface"] and value["surface"] not in seen:
            seen.add(value["surface"])
            entities.append({"surface": value["surface"], "type": value["type"]})
    for ledger in ledgers:
        for value in [*ledger["meta"].values(), *(v for row in ledger["rows"] for v in row.values())]:
            collect(value)
    collect(regulation)
    collect(V(position, "Position"))

    date = _date(rng, ctx["month"])
    incident = {"유형": incident_type, "관련자": [ctx["actor"]["surface"]], "장소": [ctx["facility"]["surface"]],
                "위반 규정": [regulation["surface"]], "일자": date,
                "매체": [ctx["medium"]["surface"]] if "medium" in " ".join(names) else [],
                "대상 문서": [ctx["document"]["surface"]] if {"document_entry", "loan", "reading"} & set(names) else []}
    return {
        "scenario_id": f"{split}-{index:03d}", "split": split, "area": area, "incident_type": incident_type,
        "unit": ctx["unit"]["surface"], "officer": ctx["officer"]["surface"], "officer_position": position,
        "actor": ctx["actor"]["surface"], "incident": incident,
        "findings": [{"영역": area, "대상": [ctx["actor"]["surface"]], "위반 규정": [regulation["surface"]]}],
        "actions": [{"조치 유형": ACTION_TYPE[incident_type], "담당": [ctx["officer"]["surface"]], "기한": "2026. 12. 31."}],
        "ledgers": ledgers, "entities": entities,
    }


def make_scenarios(n_train: int, n_eval: int, seed: int, dicts: dict, ontology: Ontology) -> list[dict]:
    rng = random.Random(seed)
    forms = load_forms(FORMS_DIR)
    types = list(INCIDENTS)
    out = []
    for i in range(n_train + n_eval):
        split = "train" if i < n_train else "eval"
        out.append(_scenario(i, split, types[i % len(types)], rng, dicts, forms, ontology))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=45)
    parser.add_argument("--eval", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    scenarios = make_scenarios(args.train, args.eval, args.seed, load_dicts(), load_ontology())
    out = DATA / "scenarios.jsonl"
    out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scenarios))
    print(f"wrote {len(scenarios)} scenarios to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
