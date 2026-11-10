"""Shared immutable model weights with serialized, request-local decoding caches."""
import hashlib
import json
from pathlib import Path
import statistics
import threading

from trialmatch.data import build_messages
from trialmatch.screening import result_from_output

ROOT=Path(__file__).resolve().parents[2]
MODES=('Cached greedy','Uncached greedy','Speculative greedy')


class ModelRuntime:
    def __init__(self,adapter_directory):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.lock=threading.RLock()
        self.candidate=json.loads((ROOT/'configs/model_candidate.json').read_text())
        self.pins=json.loads((ROOT/'configs/model_revisions.json').read_text())
        directory=Path(adapter_directory).resolve()
        for name,expected in self.candidate['adapter_file_sha256'].items():
            path=directory/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Selected adapter missing or different: {name}. Use the supplied selected-adapter ZIP.')
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype=(torch.bfloat16 if self.device.type=='cuda' and
                    torch.cuda.get_device_capability()[0]>=8 and torch.cuda.is_bf16_supported() else torch.float32)
        self.tokenizer=AutoTokenizer.from_pretrained(directory,trust_remote_code=False)
        target=self.pins['target']
        if target!=self.candidate['base_model']:
            raise ValueError('Candidate does not match pinned base model')
        self.target=PeftModel.from_pretrained(self._base('target'),directory,is_trainable=False).eval()
        self.draft=None
        import transformers,peft
        self.provenance={'target':target,'draft':self.pins['draft'],'prompt_version':'v2',
            'adapter_weights_sha256':self.candidate['adapter_file_sha256']['adapter_model.safetensors'],
            'selected_epoch':self.candidate['selected_epoch'],'training':'unweighted LoRA',
            'device':torch.cuda.get_device_name() if self.device.type=='cuda' else 'CPU',
            'dtype':str(self.dtype),'torch':torch.__version__,'transformers':transformers.__version__,
            'peft':peft.__version__,'attention_backend':'eager',
            'validation_accuracy':self.candidate['validation_metrics']['overall']['classification']['accuracy'],
            'validation_macro_f1':self.candidate['validation_metrics']['overall']['classification']['macro_f1'],
            'held_out_test_evaluated':False}

    def _base(self,role):
        from transformers import AutoModelForCausalLM
        spec=self.pins[role]
        return AutoModelForCausalLM.from_pretrained(spec['repository'],revision=spec['revision'],
            torch_dtype=self.dtype,trust_remote_code=False,attn_implementation='eager').to(self.device).eval()

    def _ensure_draft(self):
        if self.draft is None:
            from transformers import AutoTokenizer
            from trialmatch.sft import tokenizer_compatibility
            spec=self.pins['draft']
            tokenizer=AutoTokenizer.from_pretrained(spec['repository'],revision=spec['revision'])
            if not tokenizer_compatibility(self.tokenizer,tokenizer)['compatible']:
                raise ValueError('Draft tokenizer compatibility failed')
            self.draft=self._base('draft')

    def _input(self,note,criterion,max_new_tokens):
        ids=self.tokenizer.apply_chat_template(build_messages({'input':{'note':note,
            'criterion_type':criterion['criterion_type'],'criterion_text':criterion['criterion_text']}},'v2'),
            tokenize=True,add_generation_prompt=True,return_tensors='pt').to(self.device)
        if ids.shape[1]>2048 or ids.shape[1]+max_new_tokens>self.target.config.max_position_embeddings:
            raise ValueError('This criterion and patient record exceed the supported prompt length; shorten them without removing relevant evidence.')
        return ids

    def _decode(self,ids,mode,max_new_tokens):
        from trialmatch.decoding import greedy,speculative
        if mode not in MODES:
            raise ValueError('Unknown decoding mode')
        eos=self.target.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        if mode=='Speculative greedy':
            self._ensure_draft()
            return speculative(self.target,self.draft,ids,max_new_tokens,4,eos)
        return greedy(self.target,ids,max_new_tokens,mode=='Cached greedy',eos)

    def screen(self,note,criteria,mode,max_new_tokens=192,on_result=None):
        with self.lock:
            # Preflight every criterion before generating any decisions.
            inputs=[self._input(note,c,max_new_tokens) for c in criteria]
            results=[]
            for criterion,ids in zip(criteria,inputs):
                decoded=self._decode(ids,mode,max_new_tokens)
                text=self.tokenizer.decode(decoded['token_ids'],skip_special_tokens=True)
                row=result_from_output(criterion,note,text,decoded)
                row['decoding']['prompt_tokens']=int(ids.shape[1])
                results.append(row)
                if on_result:
                    on_result(len(results),len(criteria))
            return results

    def compare(self,note,criterion,max_new_tokens=64,repeats=3):
        if repeats not in (3,5) or not 8<=max_new_tokens<=192:
            raise ValueError('Use 3 or 5 repeats and an output budget from 8 to 192')
        import torch
        from trialmatch.decoding import synchronize
        with self.lock:
            self._ensure_draft()  # Both resident before any timing or warm-up.
            ids=self._input(note,criterion,max_new_tokens)
            for mode in MODES:
                self._decode(ids,mode,max_new_tokens)
            rows=[]
            for repetition in range(repeats):
                # Rotate execution order to reduce systematic order bias.
                order=MODES[repetition%3:]+MODES[:repetition%3]
                for mode in order:
                    synchronize(self.device)
                    if self.device.type=='cuda':
                        baseline=torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        baseline=None
                    result=self._decode(ids,mode,max_new_tokens)
                    rows.append({'mode':mode,'repetition':repetition,**result,
                        'baseline_allocated_bytes':baseline,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if baseline is not None else None})
            reference=next(r['token_ids'] for r in rows if r['mode']=='Uncached greedy')
            for row in rows:
                row['tokens_equal']=row['token_ids']==reference
            summaries=[]
            for mode in MODES:
                subset=[r for r in rows if r['mode']==mode]
                summaries.append({'mode':mode,
                    'median_seconds':statistics.median(r['total_seconds'] for r in subset),
                    'median_tokens_per_second':statistics.median(r['tokens_per_second'] for r in subset),
                    'median_prefill_seconds':statistics.median(r['prefill_seconds'] for r in subset),
                    'median_decode_seconds':statistics.median(r['decode_seconds'] for r in subset),
                    'output_tokens':len(subset[0]['token_ids']),
                    'all_tokens_equal':all(r['tokens_equal'] for r in subset),
                    'acceptance_rate':statistics.mean(r.get('acceptance_rate',0) for r in subset) if mode=='Speculative greedy' else None,
                    'peak_allocated_mib':max(r['peak_allocated_bytes'] for r in subset)/2**20 if self.device.type=='cuda' else None})
            return {'summaries':summaries,'measurements':rows,'all_tokens_equal':all(r['tokens_equal'] for r in rows),
                    'decoded_text':self.tokenizer.decode(reference,skip_special_tokens=True),
                    'note':note,'criterion':criterion,'prompt_tokens':int(ids.shape[1]),
                    'max_new_tokens':max_new_tokens,'repeats':repeats,'warmups_per_mode':1,
                    'eos_stopping':True,'both_models_resident':True,'provenance':dict(self.provenance)}
