"""Local experiment monitoring and durable item-level prediction checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import torch


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=list) + '\n')
    tmp.replace(path)


class Runtime:
    def __init__(self, directory, device):
        self.directory, self.device = Path(directory), device
        self.directory.mkdir(parents=True, exist_ok=True)
        self.progress = {'stage': 'initializing', 'pid': os.getpid()}
        self.failure = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)

    def monitor(self):
        import subprocess
        try:
            with (self.directory / 'resources.jsonl').open('a', buffering=1) as f:
                while not self.stop.is_set():
                    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                    available = int(mem['MemAvailable'].split()[0])
                    row = {'time': time.time(), **self.progress, 'available_kib': available,
                           'swap_free_kib': int(mem['SwapFree'].split()[0]), 'load': os.getloadavg(),
                           'cuda_allocated': torch.cuda.memory_allocated(), 'cuda_reserved': torch.cuda.memory_reserved()}
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
                    row['gpu_utilization_temperature'] = result.stdout.strip()
                    f.write(json.dumps(row) + '\n')
                    write_json(self.directory / 'progress.json', row)
                    if available < 12 * 1024**2:
                        print('[resource warning] available memory below 12 GiB', flush=True)
                    if available < 8 * 1024**2:
                        self.failure = 'Available memory below 8 GiB; stop without killing other workloads'
                    print(f"[progress] {self.progress}", flush=True)
                    self.stop.wait(15)
        except Exception as exc:
            self.failure = f'Resource monitor failed: {exc}'

    def predict(self, model, golds, ontology, condition, split):
        from gliner2.joint_ie import JointIEConfig, JointIEEngine
        folder = self.directory / 'predictions' / condition / split
        folder.mkdir(parents=True, exist_ok=True)
        engine = JointIEEngine(model, device=self.device)
        schemas = {k: ontology.joint_schema(engine, scope=k) for k in ('ledger', 'narrative')}
        record_schemas = {k: ontology.record_schema(model, k) for k in ontology.records}
        out, seconds = {}, {}
        for task in ('entities', 'relations', 'records'):
            items = golds['records' if task == 'records' else 'paragraphs']
            path = folder / f'{task}.jsonl'
            cached = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            if len(cached) > len(items): raise RuntimeError('Checkpoint contains excess items')
            values = []
            with path.open('a', buffering=1) as f, torch.inference_mode():
                for i, item in enumerate(items):
                    self.check()
                    g = item['gold']
                    key = hashlib.sha256((item['doc_id'] + ':' + str(i) + ':' + g.text +
                                          (g.record if task == 'records' else '')).encode()).hexdigest()
                    self.progress = {'stage': f'{condition}/{split}/{task}', 'completed': i, 'total': len(items), 'pid': os.getpid()}
                    if i < len(cached):
                        row = cached[i]
                        if row['id'] != key: raise RuntimeError('Prediction checkpoint input mismatch')
                    else:
                        start = time.perf_counter()
                        if task == 'entities':
                            raw = model.extract_entities(g.text, ontology.extract_labels(), threshold=0.5, include_spans=True)
                            pred = sorted({(v['start'], v['end'], k) for k, vs in raw['entities'].items() for v in vs})
                        elif task == 'relations':
                            raw = engine.extract(g.text, schemas[item['doc_kind']], config=JointIEConfig(entity_threshold=0.5)).to_dict()
                            names = {v['id']: v['text'] for v in raw['entities']}
                            pred = sorted({(names[r['head']], r['type'], names[r['tail']]) for r in raw['relations']})
                        else:
                            raw = model.extract(g.text, record_schemas[g.record]).get(g.record) or [{}]
                            raw = raw[0] if isinstance(raw, list) else raw
                            pred = {k: v for k, v in (raw or {}).items() if v not in (None, '', [])}
                        torch.cuda.synchronize() if self.device == 'cuda' else None
                        row = {'id': key, 'doc_id': item['doc_id'], 'index': i, 'prediction': pred,
                               'seconds': time.perf_counter() - start}
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    values.append(row['prediction'] if task == 'records' else {tuple(v) for v in row['prediction']})
            out[task] = values
            seconds[task] = sum(row['seconds'] for row in [json.loads(s) for s in path.read_text().splitlines()])
            print(f'[task] {condition}/{split}/{task}: {len(values)} items, {seconds[task]:.1f}s', flush=True)
        out['seconds'] = seconds
        return out
