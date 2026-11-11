"""Execute notebook 02 with full KLUE splits on CUDA, recording resources."""
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
OUT = ROOT / 'outputs/klue_ner_lora_full'
OUT.mkdir(parents=True, exist_ok=True)

def monitor():
    with (OUT / 'resources.jsonl').open('a', buffering=1) as f:
        while True:
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            f.write(json.dumps({'time': time.time(), 'available_kib': int(mem['MemAvailable'].split()[0]),
                'load': os.getloadavg(), 'cuda_allocated': torch.cuda.memory_allocated(),
                'cuda_reserved': torch.cuda.memory_reserved()}) + '\n')
            time.sleep(15)

threading.Thread(target=monitor, daemon=True).start()
nb = json.loads((ROOT / 'notebooks/02_korean_fine_tuning.ipynb').read_text())

def cell(i):
    exec(compile(''.join(nb['cells'][i]['source']), f'notebook-cell-{i}', 'exec'), globals())

print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('Expected GB10 CUDA; refusing accidental CPU training')
x = torch.ones(8, device='cuda')
print('CUDA probe', (x @ x).item(), flush=True)
for i in (3, 5, 7): cell(i)
TRAIN_SIZE, EVAL_SIZE = 21008, 5000
ADAPTER_DIR = OUT
for i in (9, 12, 15, 17): cell(i)
assert len(train_data) == len(train_rows) and len(eval_data) == len(eval_rows)
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
model = AutoExtractor.from_pretrained(BASE_MODEL)
model.set_word_splitter(kiwi_splitter)
config = TrainingConfig(output_dir=str(OUT), num_epochs=EPOCHS, batch_size=BATCH_SIZE,
    use_lora=True, save_adapter_only=True, eval_strategy='no', logging_steps=50,
    num_workers=0, pin_memory=False, seed=SEED, fp16=False, bf16=True)
(OUT / 'run_config.json').write_text(json.dumps({'model': BASE_MODEL, 'train_size': TRAIN_SIZE,
    'eval_size': EVAL_SIZE, 'labels': LABELS, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
    'seed': SEED, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0)}, indent=2))
trainer = ExtractorTrainer(model, config)
summary = trainer.train(train_data=dataset)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
del trainer, model
gc.collect()
torch.cuda.empty_cache()
results = {}
for name, adapter in [('base / kiwi', False), ('LoRA / kiwi', True)]:
    extractor = AutoExtractor.from_pretrained(BASE_MODEL)
    if adapter: extractor.load_adapter(str(OUT / 'final'))
    extractor.set_word_splitter(kiwi_splitter)
    extractor.to('cuda').eval()
    with torch.inference_mode():
        results[name] = evaluate(extractor, eval_data)
    (OUT / 'evaluation.json').write_text(json.dumps(results, indent=2))
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
show(results)
(OUT / 'COMPLETED').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
