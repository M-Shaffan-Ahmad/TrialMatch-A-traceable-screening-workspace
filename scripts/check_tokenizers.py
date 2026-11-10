"""Download tokenizer files only and pin the selected model commits, never model weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trialmatch.sft import tokenizer_compatibility


def main():
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/tokenizer-preflight')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pin_path = ROOT / 'configs/model_revisions.json'
    selected = {'target': 'Qwen/Qwen2.5-1.5B-Instruct', 'draft': 'Qwen/Qwen2.5-0.5B-Instruct'}
    if pin_path.exists():
        pins = json.loads(pin_path.read_text())
        if any(pins[role]['repository'] != repo for role, repo in selected.items()):
            raise ValueError('Existing model pins differ from selected models')
    else:
        api = HfApi()
        pins = {role: {'repository': repo, 'revision': api.model_info(repo).sha}
                for role, repo in selected.items()}
    tokenizers = {role: AutoTokenizer.from_pretrained(spec['repository'], revision=spec['revision'],
                    cache_dir=ROOT / '.cache/huggingface', trust_remote_code=False)
                  for role, spec in pins.items()}
    result = tokenizer_compatibility(tokenizers['target'], tokenizers['draft'])
    (args.output_dir / 'tokenizer_compatibility.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'model_revisions.json').write_text(json.dumps(pins, indent=2) + '\n')
    if not result['compatible']:
        raise RuntimeError('Tokenizers are incompatible; inspect the saved checks')
    pin_path.write_text(json.dumps(pins, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('Model revisions pinned. No model weights downloaded.')


if __name__ == '__main__':
    main()
