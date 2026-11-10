"""Evaluate the frozen base and selected adapter on the official held-out split."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trialmatch.workflow import evaluation_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--confirm-frozen', action='store_true')
    args = parser.parse_args()
    if not args.confirm_frozen:
        parser.error('Confirm the selected checkpoint and protocol are frozen before opening test data.')
    import torch
    import transformers
    import peft
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA runtime for the full held-out evaluation.')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidate = json.loads((ROOT / 'configs/model_candidate.json').read_text())
    for name, expected in candidate['adapter_file_sha256'].items():
        if hashlib.sha256((args.adapter_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Wrong selected adapter file: {name}')
    manifest = json.loads((ROOT / 'data/manifests/split_manifest.json').read_text())
    path = ROOT / 'data/processed/test.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    test = manifest['splits']['test']
    if (len(rows) != test['annotation_count'] or len({r['annotation_id'] for r in rows}) != len(rows)
            or dict(Counter(r['patient_id'] for r in rows)) != test['annotations_per_patient']):
        raise ValueError('Test rows differ from the fixed manifest')
    if set(test['patient_ids']) & set(manifest['splits']['train']['patient_ids'] + manifest['splits']['validation']['patient_ids']):
        raise ValueError('Patient leakage across splits')
    spec = candidate['base_model']
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported() else torch.float32
    settings = {'batch_size': 4, 'max_new_tokens': 192, 'max_prompt_tokens': 2048, 'prompt_version': 'v2'}
    provenance = {'split': 'test', 'candidate': candidate, 'settings': settings,
                  'backend': 'sdpa', 'decoding': 'greedy', 'test_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'dataset_revision': manifest['dataset_revision'], 'gpu': torch.cuda.get_device_name(),
                  'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'dtype': str(dtype)}
    run_path = output / 'run.json'
    if run_path.exists() and json.loads(run_path.read_text()) != provenance:
        raise ValueError('Existing output belongs to a different evaluation protocol.')
    write_json(run_path, provenance)
    write_json(output / 'split_manifest.json', manifest)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=False)
    def load_base():
        return transformers.AutoModelForCausalLM.from_pretrained(spec['repository'], revision=spec['revision'],
            torch_dtype=dtype, attn_implementation='sdpa', trust_remote_code=False).to(device).eval()
    base = load_base()
    baseline = evaluation_run(base, tokenizer, rows, output / 'baseline', settings)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    adapted = peft.PeftModel.from_pretrained(load_base(), args.adapter_dir, is_trainable=False).eval()
    metrics = evaluation_run(adapted, tokenizer, rows, output / 'adapted', settings)
    write_json(output / 'comparison.json', {'split':'test','examples':len(rows),'baseline':baseline,'adapted':metrics})
    print('Held-out evaluation complete. Share comparison.json and the run archive.')


if __name__ == '__main__':
    main()
