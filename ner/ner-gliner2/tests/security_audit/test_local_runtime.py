"""Prediction resumes reuse exact inputs and reject mismatched checkpoints."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/security_audit'))
from local_runtime import Runtime


class Model:
    calls = 0

    def extract_entities(self, text, *args, **kwargs):
        self.calls += 1
        return {'entities': {'Person': [{'start': 0, 'end': len(text)}]}}

    def extract(self, text, schema):
        self.calls += 1
        return {'entry': [{'name': text}]}


def test_prediction_resume_and_input_guard(tmp_path, monkeypatch):
    import gliner2.joint_ie as joint

    class Engine:
        def __init__(self, *args, **kwargs): pass
        def extract(self, *args, **kwargs):
            return SimpleNamespace(to_dict=lambda: {'entities': [], 'relations': []})

    monkeypatch.setattr(joint, 'JointIEEngine', Engine)
    ontology = SimpleNamespace(joint_schema=lambda *a, **k: None, records={'entry': None},
                              record_schema=lambda *a: None, extract_labels=lambda: {'Person': 'person'})
    paragraph = SimpleNamespace(text='Kim')
    record = SimpleNamespace(text='Kim', record='entry')
    golds = {'paragraphs': [{'doc_id': 'a', 'doc_kind': 'ledger', 'gold': paragraph}],
             'records': [{'doc_id': 'a', 'gold': record}]}
    runtime = Runtime.__new__(Runtime)
    runtime.directory, runtime.device, runtime.failure = tmp_path, 'cpu', None
    model = Model()
    first = runtime.predict(model, golds, ontology, 'base', 'eval')
    assert model.calls == 2
    second = runtime.predict(model, golds, ontology, 'base', 'eval')
    assert first == second
    assert model.calls == 2
    paragraph.text = 'Lee'
    with pytest.raises(RuntimeError, match='input mismatch'):
        runtime.predict(model, golds, ontology, 'base', 'eval')
