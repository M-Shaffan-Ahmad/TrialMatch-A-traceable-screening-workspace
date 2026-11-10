"""GPU feasibility check only, not full training, evaluation or decoding benchmarks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trialmatch.data import build_messages
from trialmatch.sft import OverlengthExample, encode_supervised, one_optimizer_step, tokenizer_compatibility


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(p.name != 'console.log' for p in output.iterdir()):
        raise ValueError('Use a new, empty output directory for each run')
    output.mkdir(parents=True, exist_ok=True)
    status = {'status': 'started', 'purpose': 'one-step feasibility only',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'test_set_used': False}
    write_json(output / 'status.json', status)
    try:
        import torch
        import transformers
        import peft
        import numpy as np
        from huggingface_hub import HfApi
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(42)
        np.random.seed(42)
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA GPU. In Colab choose a GPU runtime and rerun.')
        device = torch.device('cuda:0')
        properties = torch.cuda.get_device_properties(device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        environment = {
            'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'peft': peft.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu_name': properties.name, 'gpu_total_gib': properties.total_memory / 2**30,
            'compute_capability': list(torch.cuda.get_device_capability()),
            'dtype': str(dtype), 'seed': 42,
        }
        write_json(output / 'environment.json', environment)
        print(json.dumps(environment, indent=2), flush=True)
        if 'A100' not in properties.name:
            print('Assigned GPU is not A100. Continuing with conservative settings; inspect memory.', flush=True)
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                                text=True, check=True)
        (output / 'requirements-resolved.txt').write_text(freeze.stdout)
        manifest_path = ROOT / 'data/manifests/split_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        write_json(output / 'split_manifest.json', manifest)
        # This script intentionally opens only train.jsonl; no test/validation scoring.
        examples = [json.loads(line) for line in (ROOT / 'data/processed/train.jsonl').read_text().splitlines()]
        expected = manifest['splits']['train']
        if len(examples) != expected['annotation_count']:
            raise ValueError('Training annotation count differs from the manifest')
        if set(x['patient_id'] for x in examples) != set(expected['patient_ids']):
            raise ValueError('Training patient IDs differ from the manifest')
        if len({x['annotation_id'] for x in examples}) != len(examples):
            raise ValueError('Duplicate training annotation IDs')

        api = HfApi()
        selected = {'target': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'draft': 'Qwen/Qwen2.5-0.5B-Instruct'}
        pin_path = ROOT / 'configs/model_revisions.json'
        if pin_path.exists():
            pins = json.loads(pin_path.read_text())
            if any(pins[role]['repository'] != repo for role, repo in selected.items()):
                raise ValueError('Pinned models differ from the selected pair')
        else:
            pins = {role: {'repository': repo, 'revision': api.model_info(repo).sha}
                    for role, repo in selected.items()}
        write_json(output / 'model_revisions.json', pins)
        tokenizers = {
            role: AutoTokenizer.from_pretrained(spec['repository'], revision=spec['revision'],
                                               trust_remote_code=False)
            for role, spec in pins.items()
        }
        compatibility = tokenizer_compatibility(tokenizers['target'], tokenizers['draft'])
        write_json(output / 'tokenizer_compatibility.json', compatibility)
        if not compatibility['compatible']:
            raise RuntimeError('Tokenizer compatibility failed; inspect tokenizer_compatibility.json')
        print('Tokenizer compatibility passed.', flush=True)
        tokenizer = tokenizers['target']
        lengths, rejected, chosen = [], [], None
        for example in examples:
            try:
                encoded = encode_supervised(example, tokenizer, args.max_length)
                lengths.append(len(encoded['input_ids']))
                if chosen is None:
                    chosen = (example['annotation_id'], encoded)
            except OverlengthExample as exc:
                rejected.append({'annotation_id': example['annotation_id'], 'reason': str(exc)})
        if chosen is None:
            raise RuntimeError('No complete training example fits; review the training-only length report')
        write_json(output / 'train_preparation.json', {
            'source': 'train only', 'max_length': args.max_length, 'truncated_examples': 0,
            'accepted': len(lengths), 'rejected': rejected,
            'accepted_length_percentiles': dict(zip(['min', 'median', 'p95', 'max'],
                                                   map(float, np.percentile(lengths, [0, 50, 95, 100])))),
            'label_counts': dict(Counter(e['target']['label'] for e in examples)),
            'smoke_annotation_id': chosen[0],
        })

        def load_base(role):
            spec = pins[role]
            model = AutoModelForCausalLM.from_pretrained(
                spec['repository'], revision=spec['revision'], trust_remote_code=False,
                torch_dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True,
            ).to(device)
            if max(tokenizers[role].get_vocab().values()) >= model.get_input_embeddings().num_embeddings:
                raise ValueError(f'{role} tokenizer can index outside model embeddings')
            return model

        synthetic = {'input': {'note': '0. The patient is 50 years old.\n1. Diabetes is not documented.',
                              'criterion_type': 'inclusion', 'criterion_text': 'Age at least 18 years.'}}
        prompt_ids = tokenizer.apply_chat_template(build_messages(synthetic), tokenize=True,
                                                   add_generation_prompt=True, return_tensors='pt').to(device)
        prompt_mask = torch.ones_like(prompt_ids)

        @torch.inference_mode()
        def smoke_generate(model):
            model.eval()
            generated = model.generate(
                input_ids=prompt_ids, attention_mask=prompt_mask, do_sample=False,
                max_new_tokens=16, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            return generated[0, prompt_ids.shape[-1]:].tolist()

        print('Loading both selected models on the GPU.', flush=True)
        target = load_base('target')
        draft = load_base('draft')
        write_json(output / 'model_loading.json', {
            role: {'parameters': sum(p.numel() for p in model.parameters()),
                   'layers': model.config.num_hidden_layers, 'vocab_size': model.config.vocab_size}
            for role, model in [('target', target), ('draft', draft)]
        })
        base_tokens = smoke_generate(target)
        draft_tokens = smoke_generate(draft)
        write_json(output / 'generation_smoke.json', {
            'prompt_source': 'synthetic, not held-out data',
            'note': 'Library greedy generation is only a loading check, not custom decoding implementation.',
            'base_target_token_ids': base_tokens, 'draft_token_ids': draft_tokens,
            'both_models_allocated_gib': torch.cuda.memory_allocated() / 2**30,
        })
        del draft
        gc.collect()
        torch.cuda.empty_cache()
        target.config.use_cache = False
        model = get_peft_model(target, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
            bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                       'gate_proj', 'up_proj', 'down_proj'],
        ))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print('Running one real optimizer step on one training example.', flush=True)
        update = one_optimizer_step(model, chosen[1], device)
        update['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        update['annotation_id'] = chosen[0]
        write_json(output / 'one_step.json', update)
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        expected_tokens = smoke_generate(model)
        adapter_dir = output / 'smoke-adapter'
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        (adapter_dir / 'NOT_FINAL_CHECKPOINT.md').write_text(
            '# Feasibility adapter only\n\nOne optimizer step, not the final trained model.\n'
            'Do not use as a final result or overwrite a full training checkpoint with it.\n')
        del model, target
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(load_base('target'), adapter_dir, is_trainable=False)
        actual_tokens = smoke_generate(reloaded)
        equal = expected_tokens == actual_tokens
        write_json(output / 'adapter_reload.json', {
            'synthetic_greedy_tokens_equal': equal,
            'before_reload_tokens': expected_tokens, 'after_reload_tokens': actual_tokens,
        })
        if not equal:
            raise RuntimeError('Adapter reload changed synthetic greedy tokens; investigate before training')
        status.update(status='passed', finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / 'status.json', status)
        print('PASSED: GPU/model loading, tokenizer compatibility, optimizer update, adapter save/reload.', flush=True)
        print(f'Artifacts saved to {output}', flush=True)
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / 'status.json', status)
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
