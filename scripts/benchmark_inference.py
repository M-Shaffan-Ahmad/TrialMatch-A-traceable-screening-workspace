"""Full selected-adapter inference correctness and repeated benchmark; no benchmark dataset."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trialmatch.workflow import write_json
from trialmatch.decoding import greedy,speculative,synchronize
from trialmatch.attention import reference_attention,tiled_attention


def measure(call,device):
    import torch
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
        synchronize(device)
        baseline=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline=None
    start=time.perf_counter()
    value=call()
    synchronize(device)
    elapsed=time.perf_counter()-start
    memory={'baseline_allocated_bytes':baseline,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if baseline is not None else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if baseline is not None else None}
    if baseline is not None:
        memory['incremental_peak_allocated_bytes']=memory['peak_allocated_bytes']-baseline
    return value,elapsed,memory


def attention_suite(device,warmups,repeats,output):
    import torch
    rows=[]
    # Rectangular cache-style causal masks and noncausal shapes included in correctness.
    for nq,nk,offset,causal in [(7,7,0,True),(5,13,8,True),(11,17,0,False)]:
        generator=torch.Generator(device=device).manual_seed(42)
        q=torch.randn(2,3,nq,16,device=device,generator=generator)
        k=torch.randn(2,3,nk,16,device=device,generator=generator)
        v=torch.randn(2,3,nk,12,device=device,generator=generator)
        expected=reference_attention(q,k,v,causal,offset)
        actual=tiled_attention(q,k,v,causal,offset,3,4)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        rows.append({'kind':'correctness','shape':[2,3,nq,nk,16,12],
                     'causal':causal,'query_offset':offset,
                     'maximum_absolute_error':float((actual-expected).abs().max()),
                     'mean_absolute_error':float((actual-expected).abs().mean())})
    for length in [128,512,1024]:
        generator=torch.Generator(device=device).manual_seed(42)
        q,k,v=[torch.randn(1,4,length,64,device=device,generator=generator) for _ in range(3)]
        expected=reference_attention(q,k,v)
        actual=tiled_attention(q,k,v)
        error=(actual-expected).abs()
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        errors={'maximum_absolute_error':float(error.max()),'mean_absolute_error':float(error.mean())}
        del expected,actual,error
        for name,fn in [('dense_reference',reference_attention),('tiled_online_softmax',tiled_attention)]:
            for _ in range(warmups):
                warm=fn(q,k,v)
                del warm
            measurements=[]
            for _ in range(repeats):
                value,seconds,memory=measure(lambda:fn(q,k,v),device)
                measurements.append({'seconds':seconds,**memory})
                del value
            rows.append({'kind':'benchmark','mode':name,'length':length,'dtype':'float32',
                         'query_tile':32,'key_tile':64,'causal':True,
                         **errors,'measurements':measurements,
                         'median_seconds':statistics.median(r['seconds'] for r in measurements)})
        write_json(output/'attention.json',rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--draft-tokens',type=int,default=4)
    args=parser.parse_args()
    if args.warmups<1 or args.repeats<2 or args.draft_tokens<1:
        raise ValueError('Require warmup >=1, repeats >=2 and positive draft block')
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    if (output/'status.json').exists():
        raise ValueError('Use a new output directory; preserve prior benchmark logs')
    write_json(output/'status.json',{'status':'running','test_set_used':False})
    try:
        import torch
        import transformers
        import peft
        from peft import PeftModel
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from trialmatch.sft import tokenizer_compatibility
        from trialmatch.data import build_messages
        if not torch.cuda.is_available():
            raise RuntimeError('Select a CUDA GPU runtime in Colab')
        device=torch.device('cuda:0')
        # Require native BF16. Some runtimes report emulated BF16 on older GPUs.
        major,_=torch.cuda.get_device_capability(device)
        dtype=torch.bfloat16 if major>=8 and torch.cuda.is_bf16_supported() else torch.float32
        candidate=json.loads((ROOT/'configs/model_candidate.json').read_text())
        pins=json.loads((ROOT/'configs/model_revisions.json').read_text())
        if candidate['base_model'] != pins['target']:
            raise ValueError('Candidate base model differs from pinned target')
        for name,expected in candidate['adapter_file_sha256'].items():
            path=args.adapter_dir/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Adapter integrity check failed: {name}')
        tokenizer=AutoTokenizer.from_pretrained(args.adapter_dir,trust_remote_code=False)
        dt=AutoTokenizer.from_pretrained(pins['draft']['repository'],revision=pins['draft']['revision'])
        compatible=tokenizer_compatibility(tokenizer,dt)
        write_json(output/'tokenizer_compatibility.json',compatible)
        if not compatible['compatible']:
            raise ValueError('Tokenizers differ; speculative token comparison is unsafe')
        environment={'gpu':torch.cuda.get_device_name(device),
                     'gpu_total_gib':torch.cuda.get_device_properties(device).total_memory/2**30,
                     'torch':torch.__version__,'transformers':transformers.__version__,
                     'peft':peft.__version__,'python':sys.version,'dtype':str(dtype),
                     'cuda':torch.version.cuda,'candidate':candidate,
                     'warmups':args.warmups,'repeats':args.repeats,'draft_tokens':args.draft_tokens,
                     'memory_policy':'PyTorch allocated/reserved device memory, baseline and incremental peak; not process/driver total.',
                     'timing_policy':'CUDA synchronize before/after measurements; warmups; cache cleared outside timing; end-to-end includes Python overhead.',
                     'attention_backend':'eager for selected models; educational tiled attention tested separately, not patched into model'}
        print(json.dumps({k:v for k,v in environment.items() if k!='candidate'},indent=2),flush=True)
        write_json(output/'environment.json',environment)
        freeze=subprocess.run([sys.executable,'-m','pip','freeze','--exclude-editable'],capture_output=True,text=True,check=True)
        (output/'requirements-resolved.txt').write_text(freeze.stdout)
        torch.manual_seed(42)
        with torch.inference_mode():
            attention_suite(device,args.warmups,args.repeats,output)
        def load(role):
            spec=pins[role]
            return AutoModelForCausalLM.from_pretrained(spec['repository'],revision=spec['revision'],
                torch_dtype=dtype,attn_implementation='eager',trust_remote_code=False).to(device).eval()
        target=PeftModel.from_pretrained(load('target'),args.adapter_dir,is_trainable=False).eval()
        draft=load('draft')
        eos=target.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        # Synthetic text only: no benchmark patient examples or evaluation labels.
        text=('Educational benchmark. A synthetic patient is 50 years old. '
              'Only documented facts may support trial screening. ')*300
        pool=tokenizer.encode(text,add_special_tokens=False)
        cases=[(128,32),(512,64),(1024,128)]
        rows=[]
        for length,count in cases:
            ids=torch.tensor([pool[:length]],dtype=torch.long,device=device)
            assert ids.shape[1]==length
            calls={'uncached':lambda:greedy(target,ids,count,False),
                   'cached':lambda:greedy(target,ids,count,True),
                   'speculative':lambda:speculative(target,draft,ids,count,args.draft_tokens)}
            # Fixed-work benchmarks intentionally ignore EOS. Real prompts below stop at EOS.
            reference=calls['uncached']()['token_ids']
            for name,call in calls.items():
                for _ in range(args.warmups):
                    call()
                measurements=[]
                for repetition in range(args.repeats):
                    value,seconds,memory=measure(call,device)
                    equal=value['token_ids']==reference
                    record={'input_tokens':length,'requested_output_tokens':count,'mode':name,
                            'repetition':repetition,'tokens_equal':equal,'wall_seconds':seconds,
                            **memory,**value}
                    measurements.append(record)
                    rows.append(record)
                    write_json(output/'decoding.json',rows)
                    if not equal:
                        raise RuntimeError(f'Greedy token mismatch: {name}, {length}/{count}. See decoding.json; do not claim equivalence.')
                print(f'{length}/{count} {name}: median {statistics.median(r["total_seconds"] for r in measurements):.3f}s',flush=True)
        prompt={'input':{'note':'0. A synthetic patient is 50 years old.\n1. Diabetes is explicitly denied.',
                          'criterion_type':'inclusion','criterion_text':'Age at least 18 years.'}}
        ids=tokenizer.apply_chat_template(build_messages(prompt,'v2'),tokenize=True,
                                          add_generation_prompt=True,return_tensors='pt').to(device)
        outputs={'uncached':greedy(target,ids,192,False,eos),
                 'cached':greedy(target,ids,192,True,eos),
                 'speculative':speculative(target,draft,ids,192,args.draft_tokens,eos)}
        write_json(output/'synthetic_screening.json',outputs)
        if not all(v['token_ids']==outputs['uncached']['token_ids'] for v in outputs.values()):
            raise RuntimeError('EOS-aware screening output mismatch')
        summary=[]
        for length,count in cases:
            metrics={}
            for mode in ['uncached','cached','speculative']:
                subset=[r for r in rows if r['input_tokens']==length and r['mode']==mode]
                metrics[mode]={key:statistics.median(r[key] for r in subset)
                    for key in ['prefill_seconds','decode_seconds','total_seconds','tokens_per_second',
                                'peak_allocated_bytes','incremental_peak_allocated_bytes']}
                if mode=='speculative':
                    metrics[mode]['acceptance_rate']=statistics.mean(r['acceptance_rate'] for r in subset)
            summary.append({'input_tokens':length,'output_tokens':count,'metrics':metrics,
                'cached_speedup_vs_uncached':metrics['uncached']['total_seconds']/metrics['cached']['total_seconds'],
                'speculative_speedup_vs_cached':metrics['cached']['total_seconds']/metrics['speculative']['total_seconds']})
        write_json(output/'summary.json',summary)
        report='# Inference benchmark\n\nAll fixed-length and EOS-aware custom decoding token checks passed. Synthetic inputs only; test set unused.\n\n'
        report+='| Input/output tokens | Mode | Median seconds | Tokens/s | Prefill seconds | Decode seconds |\n|---|---|---:|---:|---:|---:|\n'
        for item in summary:
            for mode,m in item['metrics'].items():
                report+=f'| {item["input_tokens"]}/{item["output_tokens"]} | {mode} | {m["total_seconds"]:.4f} | {m["tokens_per_second"]:.2f} | {m["prefill_seconds"]:.4f} | {m["decode_seconds"]:.4f} |\n'
        report+='\nBoth models remain resident in every decoding timing for comparable memory baselines. Speculative prefill includes both models; ordinary prefill includes target only. Decode throughput, raw repetitions, acceptance and allocator peaks are in JSON. No speedup is assumed. Educational tiled attention is a standalone correctness/performance implementation, not a fused model backend.\n'
        (output/'REPORT.md').write_text(report)
        write_json(output/'status.json',{'status':'passed','test_set_used':False,'correctness_cases':cases})
    except Exception as exc:
        import traceback
        write_json(output/'status.json',{'status':'failed','test_set_used':False,'error':str(exc)})
        (output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
