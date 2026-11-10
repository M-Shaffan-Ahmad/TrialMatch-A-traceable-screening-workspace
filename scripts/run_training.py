"""Official train/validation workflow. Deliberately provides no test-set option."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trialmatch.workflow import run_workflow, write_json


def load_split(name, manifest):
    if name not in {'train','validation'}:
        raise ValueError('This development workflow cannot open the test split')
    path = ROOT/'data/processed'/f'{name}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = manifest['splits'][name]
    if len(rows) != expected['annotation_count']:
        raise ValueError(f'{name} annotation count differs from manifest')
    if set(r['patient_id'] for r in rows) != set(expected['patient_ids']):
        raise ValueError(f'{name} patient IDs differ from manifest')
    if len({r['annotation_id'] for r in rows}) != len(rows):
        raise ValueError(f'{name} annotation IDs are not unique')
    from collections import Counter
    if dict(Counter(r['patient_id'] for r in rows)) != expected['annotations_per_patient']:
        raise ValueError(f'{name} per-patient counts differ from manifest')
    return rows,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/experiment.toml')
    parser.add_argument('--resume',action='store_true',help='Resume from the last completed epoch; preserve the same config/code/environment')
    parser.add_argument('--baseline-from',type=Path,help='Reuse a verified same-protocol untouched-base validation from another run')
    args = parser.parse_args()
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    spec = tomllib.loads(args.config.read_text())
    training,model_spec = spec['training'],spec['model']
    config = {
        'seed':spec['seed'],'epochs':training['epochs'],
        'experiment_id':spec.get('experiment',{}).get('id','original_v1'),
        'prompt_version':spec.get('prompt',{}).get('version','v1'),
        'class_weighting':training.get('class_weighting','none'),
        'class_weight_cap':training.get('class_weight_cap',3.0),
        'batch_size':training['per_device_batch_size'],
        'accumulation':training['gradient_accumulation_steps'],
        'learning_rate':training['learning_rate'],'lora_rank':training['lora_rank'],
        'lora_alpha':training['lora_alpha'],'lora_dropout':training['lora_dropout'],
        'warmup_ratio':training.get('warmup_ratio',0.1),'weight_decay':training.get('weight_decay',0.01),
        'max_length':model_spec['max_sequence_length'],
        'eval_batch_size':spec.get('evaluation',{}).get('batch_size',4),
        'max_new_tokens':spec['generation']['max_new_tokens'],
    }
    if any(config[k]<1 for k in ['epochs','batch_size','accumulation','max_length','eval_batch_size','max_new_tokens']):
        raise ValueError('Epoch, batch, accumulation and token counts must be positive')
    if config['prompt_version'] not in {'v1','v2'} or config['class_weighting'] not in {'none','inverse_sqrt_capped'}:
        raise ValueError('Unknown prompt version or class weighting method')
    if spec['generation']['do_sample'] is not False:
        raise ValueError('The common before/after workflow requires deterministic greedy decoding')
    if training['method'] != 'lora' or model_spec['overlength_policy'] != 'reject_and_log':
        raise ValueError('Unsupported training or truncation policy')
    manifest = json.loads((ROOT/'data/manifests/split_manifest.json').read_text())
    train,train_hash = load_split('train',manifest)
    validation,val_hash = load_split('validation',manifest)
    if {e['patient_id'] for e in train} & {e['patient_id'] for e in validation}:
        raise ValueError('Train/validation patient leakage')
    pins = json.loads((ROOT/'configs/model_revisions.json').read_text())
    target = pins['target']
    if target['repository'] != model_spec['target']:
        raise ValueError('Configured target differs from pinned target')
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('Full training requires a CUDA runtime. Select GPU in Colab.')
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    environment = {'torch':torch.__version__,'transformers':transformers.__version__,
                   'peft':peft.__version__,'python':sys.version,'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name(device),
                   'gpu_total_gib':torch.cuda.get_device_properties(device).total_memory/2**30,
                   'dtype':str(dtype)}
    source_paths = sorted((ROOT/'src/trialmatch').glob('*.py'))+[Path(__file__).resolve()]
    code_hash = hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in source_paths)).hexdigest()
    provenance = {'model':target,'dataset_revision':manifest['dataset_revision'],
                  'train_sha256':train_hash,'validation_sha256':val_hash,
                  'source_sha256':code_hash,'environment':environment,
                  'decoding':'Transformers library greedy; custom assignment decoders pending'}
    tokenizer = AutoTokenizer.from_pretrained(target['repository'],revision=target['revision'],trust_remote_code=False)
    def load_base():
        return AutoModelForCausalLM.from_pretrained(target['repository'],revision=target['revision'],
                    trust_remote_code=False,torch_dtype=dtype,attn_implementation='sdpa',
                    low_cpu_mem_usage=True).to(device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps(environment,indent=2),flush=True)
    # Record resolved package versions alongside the run; exclude editable absolute paths.
    freeze = subprocess.run([sys.executable,'-m','pip','freeze','--exclude-editable'],
                             capture_output=True,text=True,check=True)
    (args.output_dir/'requirements-resolved.txt').write_text(freeze.stdout)
    write_json(args.output_dir/'split_manifest.json',manifest)
    result = run_workflow(args.output_dir,tokenizer,load_base,train,validation,config,provenance,device,args.resume,baseline_from=args.baseline_from)
    print('Selected epoch:',result['selected_epoch'],flush=True)


if __name__=='__main__':
    main()
