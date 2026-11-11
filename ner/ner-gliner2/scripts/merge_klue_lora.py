"""Merge full-KLUE adapter, reload artifact, and benchmark full validation."""
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from gliner2 import AutoExtractor
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from config.extractors import kiwi_splitter

torch.cuda.init()
print('CUDA initialized', flush=True)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
RUN = ROOT / 'outputs/klue_ner_lora_full'
DEST = RUN / 'merged'
nb = json.loads((ROOT / 'notebooks/02_korean_fine_tuning.ipynb').read_text())
def cell(i):
    exec(compile(''.join(nb['cells'][i]['source']), f'notebook-cell-{i}', 'exec'), globals())
for i in (3, 7): cell(i)
TRAIN_SIZE, EVAL_SIZE = 21008, 5000
for i in (9, 17): cell(i)
base = AutoExtractor.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, str(RUN / 'final'), torch_device='cpu').to('cuda').eval()
model.set_word_splitter(kiwi_splitter)
probes = [text for text, _ in eval_data[:20]]
with torch.inference_mode():
    before = [model.extract_entities(t, LABELS, include_spans=True) for t in probes]
torch.cuda.synchronize()
start = perf_counter()
merged = model.merge_and_unload(safe_merge=True)
torch.cuda.synchronize()
merge_seconds = perf_counter() - start
assert not any(isinstance(m, LoraLayer) for m in merged.modules())
with torch.inference_mode():
    after = [merged.extract_entities(t, LABELS, include_spans=True) for t in probes]
assert before == after, 'Merge changed probe predictions'
start = perf_counter()
merged.to('cpu').save_pretrained(str(DEST))
save_seconds = perf_counter() - start
del merged, model, base
gc.collect()
torch.cuda.empty_cache()
reloaded = AutoExtractor.from_pretrained(str(DEST)).to('cuda').eval()
reloaded.set_word_splitter(kiwi_splitter)
with torch.inference_mode():
    restored = [reloaded.extract_entities(t, LABELS, include_spans=True) for t in probes]
assert after == restored, 'Reload changed probe predictions'
torch.cuda.synchronize()
with torch.inference_mode():
    scores = evaluate(reloaded, eval_data)
torch.cuda.synchronize()
report = {'merge_seconds': merge_seconds, 'save_seconds': save_seconds,
          'probe_count': len(probes), 'probe_predictions_identical': True,
          'evaluation': scores}
(RUN / 'merged_evaluation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
