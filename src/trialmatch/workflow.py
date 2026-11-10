"""Validation-only baseline, LoRA training, epoch checkpoints and comparison artifacts."""
from datetime import datetime, timezone
import json
import math
import shutil
from pathlib import Path
import time
import uuid

from trialmatch.evaluation import evaluate_records
from trialmatch.generation import predict_examples
from trialmatch.sft import OverlengthExample, encode_supervised
from trialmatch.training import checkpoint_score, learning_rate_factor, train_epoch


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def evaluation_run(model, tokenizer, examples, destination, generation):
    destination.mkdir(parents=True,exist_ok=True)
    metrics_path = destination/'metrics.json'
    if metrics_path.exists():
        records = [json.loads(line) for line in (destination/'predictions.jsonl').read_text().splitlines()]
        if [r['annotation_id'] for r in records] != [e['annotation_id'] for e in examples]:
            raise ValueError('Saved validation IDs differ from this run')
        metrics = evaluate_records(records)
        if metrics != json.loads(metrics_path.read_text()):
            raise ValueError('Saved validation metrics do not match predictions')
        return metrics
    start = time.perf_counter()
    count = 0
    with (destination/'predictions.partial.jsonl').open('w') as progress:
        def on_record(record):
            nonlocal count
            count += 1
            progress.write(json.dumps(record,ensure_ascii=False)+'\n')
            progress.flush()
            if count % 20 == 0 or count == len(examples):
                print(f'Validation: {count}/{len(examples)}',flush=True)
        records = predict_examples(model,tokenizer,examples,on_record=on_record,**generation)
    # Write a canonical ordering after the live progress file, including rejected prompts.
    text = ''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records)
    (destination/'predictions.jsonl').write_text(text)
    metrics = evaluate_records(records)
    write_json(destination/'timing.json', {'seconds':time.perf_counter()-start,
               'purpose':'Diagnostic validation duration, not a repeated inference benchmark'})
    write_json(metrics_path,metrics)  # Completion marker, written after predictions.
    return metrics


def write_comparison(output, baseline, adapted, best_epoch):
    comparison = {'split':'validation','test_set_used':False,'selected_epoch':best_epoch,
                  'selection_rule':'Highest validation classification macro-F1, then evidence micro-F1; earliest epoch on exact tie.',
                  'baseline':baseline,'adapted':adapted}
    write_json(output/'comparison.json',comparison)
    a,b = baseline['overall'],adapted['overall']
    def fmt(value):
        return 'undefined' if value is None else f'{value:.4f}'
    rows = [('Accuracy',a['classification']['accuracy'],b['classification']['accuracy']),
            ('Classification macro-F1',a['classification']['macro_f1'],b['classification']['macro_f1']),
            ('Evidence micro-F1',a['evidence']['micro']['f1'],b['evidence']['micro']['f1']),
            ('Evidence macro-F1',a['evidence']['macro']['f1'],b['evidence']['macro']['f1'])]
    text = '# Validation comparison\n\n'
    text += f'Selected fine-tuned epoch: {best_epoch}. Test set was not used.\n\n'
    text += '| Metric | Untouched base | Selected adapter |\n|---|---:|---:|\n'
    text += ''.join(f'| {name} | {fmt(x)} | {fmt(y)} |\n' for name,x,y in rows)
    text += f'| Invalid structured outputs | {a["invalid_outputs"]} | {b["invalid_outputs"]} |\n'
    text += '\nMacro-F1 uses the fixed six-category overall label set, including zero-support classes. '
    text += 'Per-type/per-class metrics and confusion matrices are in comparison.json. '
    text += 'These are development results, not final held-out test results.\n'
    (output/'COMPARISON.md').write_text(text)


def write_plots(output, history, epoch_metrics, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot([r['global_step'] for r in history],[r['loss'] for r in history],linewidth=1)
    axes[0].set(xlabel='Optimizer step',ylabel='Assistant-token cross-entropy',title='Training loss')
    epochs = [0]+list(range(1,len(epoch_metrics)+1))
    metrics = [baseline]+epoch_metrics
    axes[1].plot(epochs,[m['overall']['classification']['macro_f1'] for m in metrics],marker='o',label='Classification macro-F1')
    axes[1].plot(epochs,[m['overall']['evidence']['micro']['f1'] for m in metrics],marker='o',label='Evidence micro-F1')
    axes[1].set(xlabel='Epoch (0 = base)',ylabel='Validation score',title='Validation during adaptation',ylim=(0,1))
    axes[1].set_xticks(epochs)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(output/'training-validation-curves.png',dpi=160)
    fig.savefig(output/'training-validation-curves.svg')
    plt.close(fig)


def reuse_baseline(output, source_run, config, provenance):
    source_run = Path(source_run).resolve()
    output = Path(output).resolve()
    if source_run == output:
        raise ValueError('Baseline reuse source must be a different run')
    previous = json.loads((source_run/'run.json').read_text())
    keys = ('seed','max_length','eval_batch_size','max_new_tokens','prompt_version')
    defaults = {'prompt_version':'v1'}
    if any(previous['config'].get(k,defaults.get(k)) != config.get(k,defaults.get(k)) for k in keys):
        raise ValueError('Cannot reuse baseline: generation or prompt protocol differs')
    if previous['provenance'] != provenance:
        raise ValueError('Cannot reuse baseline: model, data, source or runtime provenance differs')
    if not (source_run/'baseline/metrics.json').exists():
        raise ValueError('Source baseline has not completed')
    destination = output/'baseline'
    if not (destination/'metrics.json').exists():
        destination.mkdir(exist_ok=True)
        # Copy predictions first, completion marker last.
        shutil.copyfile(source_run/'baseline/predictions.jsonl',destination/'predictions.jsonl')
        if (source_run/'baseline/timing.json').exists():
            shutil.copyfile(source_run/'baseline/timing.json',destination/'timing.json')
        shutil.copyfile(source_run/'baseline/metrics.json',destination/'metrics.json')
    write_json(output/'baseline_reuse.json',{'source_run':str(source_run),
               'note':'Only untouched-base validation reused; no adapter weights reused.'})


def run_workflow(output, tokenizer, load_base, train_examples, validation_examples,
                 config, provenance, device, resume=False, baseline_from=None):
    """load_base creates a fresh pinned target; callers own data-integrity verification."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    run_spec = {'config':config,'provenance':provenance,'test_set_used':False,
                'selection_rule':'validation macro-F1, then evidence micro-F1, then earliest epoch'}
    spec_path = output/'run.json'
    if spec_path.exists():
        if not resume:
            raise ValueError('Existing run: use --resume or choose a new output directory')
        if json.loads(spec_path.read_text()) != run_spec:
            raise ValueError('Resume refused: code, data, environment or configuration changed')
    else:
        if resume:
            raise ValueError('No existing run to resume')
        write_json(spec_path,run_spec)
    write_json(output/'status.json',{'status':'running','test_set_used':False,
                                     'updated_utc':datetime.now(timezone.utc).isoformat()})
    try:
        torch.manual_seed(config['seed'])
        prepared,rejected,accepted_examples = [],[],[]
        for example in train_examples:
            try:
                prepared.append(encode_supervised(example,tokenizer,config['max_length'],config.get('prompt_version','v1')))
                accepted_examples.append(example)
            except OverlengthExample as exc:
                rejected.append({'annotation_id':example['annotation_id'],'reason':str(exc)})
        if not prepared:
            raise ValueError('No complete training examples fit')
        write_json(output/'train_preparation.json',{'accepted':len(prepared),'rejected':rejected,
                                                   'truncated':0,'max_length':config['max_length']})
        from trialmatch.balancing import class_weights
        weights,weight_summary = class_weights(accepted_examples,config.get('class_weighting','none'),
                                               config.get('class_weight_cap',3.0))
        write_json(output/'class_weighting.json',weight_summary)
        from trialmatch.data import build_messages
        write_json(output/'prompt_protocol.json',{'version':config.get('prompt_version','v1'),
            'systems':{kind:build_messages({'input':{'note':'0. Synthetic example.',
                       'criterion_type':kind,'criterion_text':'Synthetic criterion'}},
                       config.get('prompt_version','v1'))[0]['content']
                       for kind in ['inclusion','exclusion']}})
        if baseline_from is not None:
            reuse_baseline(output,baseline_from,config,provenance)
        model = load_base()
        generation = {'batch_size':config['eval_batch_size'],
                      'max_new_tokens':config['max_new_tokens'],
                      'max_prompt_tokens':config['max_length'],'prompt_version':config.get('prompt_version','v1')}
        print('Evaluating the untouched base model on validation.',flush=True)
        baseline = evaluation_run(model,tokenizer,validation_examples,output/'baseline',generation)
        last_epoch = 0
        for epoch in range(1,config['epochs']+1):
            checkpoint = output/f'epoch-{epoch:02d}'
            if (checkpoint/'trainer_state.pt').exists():
                if epoch != last_epoch+1:
                    raise ValueError('Non-contiguous epoch checkpoints')
                last_epoch = epoch
            elif checkpoint.exists():
                raise ValueError('Incomplete checkpoint directory; preserve it and inspect the failed save')
        if last_epoch:
            model = PeftModel.from_pretrained(model,output/f'epoch-{last_epoch:02d}'/'adapter',is_trainable=True)
        else:
            model = get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=config['lora_rank'],
                lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],bias='none',
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        steps_per_epoch = math.ceil(math.ceil(len(prepared)/config['batch_size'])/config['accumulation'])
        total_steps = steps_per_epoch*config['epochs']
        warmup_steps = max(1,math.ceil(total_steps*config['warmup_ratio']))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step:learning_rate_factor(step,total_steps,warmup_steps))
        global_step = 0
        if last_epoch:
            # Only load optimizer states produced by this workflow; no remote pickle.
            state = torch.load(output/f'epoch-{last_epoch:02d}'/'trainer_state.pt',map_location='cpu',weights_only=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            global_step = state['global_step']
            torch.set_rng_state(state['rng_cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(state['rng_cuda'])
            if global_step != steps_per_epoch*last_epoch:
                raise ValueError('Checkpoint optimizer step count is inconsistent')
        history,epoch_metrics = [],[]
        best_epoch,best_metrics = None,None
        for epoch in range(1,config['epochs']+1):
            checkpoint = output/f'epoch-{epoch:02d}'
            if epoch > last_epoch:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                model.enable_input_require_grads()
                updates = []
                start = time.perf_counter()
                if device.type=='cuda':
                    torch.cuda.reset_peak_memory_stats()
                with (output/f'epoch-{epoch:02d}-progress.jsonl').open('w') as progress:
                    def on_update(record):
                        updates.append(record)
                        progress.write(json.dumps(record)+'\n')
                        progress.flush()
                        if record['global_step']%10==0:
                            print(f'Epoch {epoch}: step {record["global_step"]}/{total_steps}, loss {record["loss"]:.4f}',flush=True)
                    global_step = train_epoch(model,prepared,optimizer,scheduler,device,
                        tokenizer.pad_token_id,config['batch_size'],config['accumulation'],
                        config['seed'],epoch-1,global_step,on_update,
                        example_weights=weights if config.get('class_weighting','none')!='none' else None)
                if device.type=='cuda':
                    torch.cuda.synchronize()
                summary = {'epoch':epoch,'global_step':global_step,'seconds':time.perf_counter()-start,
                           'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                           'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None}
                # A same-filesystem directory rename marks a complete epoch checkpoint.
                staging = output/f'.epoch-{epoch:02d}-saving-{uuid.uuid4().hex}'
                staging.mkdir()
                model.save_pretrained(staging/'adapter',safe_serialization=True)
                tokenizer.save_pretrained(staging/'adapter')
                write_json(staging/'train_summary.json',summary)
                write_json(staging/'updates.json',updates)
                torch.save({'global_step':global_step,'optimizer':optimizer.state_dict(),
                            'scheduler':scheduler.state_dict(),'rng_cpu':torch.get_rng_state(),
                            'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},
                           staging/'trainer_state.pt')
                staging.rename(checkpoint)
            # Existing metrics are verified and reused, even after a later-epoch resume.
            # If interrupted after saving an epoch but before evaluation, evaluate that
            # adapter explicitly instead of accidentally using a later epoch's weights.
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            metrics_path = checkpoint/'validation'/'metrics.json'
            if epoch < last_epoch and not metrics_path.exists():
                raise ValueError('An earlier epoch lacks validation; investigate the run before resuming')
            print(f'Validation for epoch {epoch}.',flush=True)
            metrics = evaluation_run(model,tokenizer,validation_examples,checkpoint/'validation',generation)
            epoch_metrics.append(metrics)
            history.extend(json.loads((checkpoint/'updates.json').read_text()))
            if best_metrics is None or checkpoint_score(metrics)>checkpoint_score(best_metrics):
                best_epoch,best_metrics = epoch,metrics
        selected = output/f'epoch-{best_epoch:02d}'/'adapter'
        # Selection is a portable pointer to an existing adapter, never an untested merge.
        write_json(output/'selected_adapter.json',{'epoch':best_epoch,
            'relative_path':str(selected.relative_to(output)), 'base_model':provenance['model'],
            'prompt_version':config.get('prompt_version','v1'),
            'class_weighting':config.get('class_weighting','none'),
            'validation_score':list(checkpoint_score(best_metrics)), 'test_set_used':False})
        # Validate the actual selected checkpoint after reloading it on a fresh base.
        del optimizer, scheduler, model
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(load_base(),selected,is_trainable=False)
        final_metrics = evaluation_run(reloaded,tokenizer,validation_examples,
                                        output/'selected-adapter-validation',generation)
        expected_records = [json.loads(line) for line in
            (output/f'epoch-{best_epoch:02d}'/'validation'/'predictions.jsonl').read_text().splitlines()]
        actual_records = [json.loads(line) for line in
            (output/'selected-adapter-validation'/'predictions.jsonl').read_text().splitlines()]
        keys = ('annotation_id','raw_output','valid','label','evidence_sentence_ids')
        equal = (len(expected_records)==len(actual_records) and
                 all(all(a[k]==b[k] for k in keys) for a,b in zip(expected_records,actual_records)))
        write_json(output/'checkpoint_reload.json',{'validation_outputs_equal':equal,
                    'metrics_equal':final_metrics==best_metrics,'examples':len(actual_records)})
        if not equal or final_metrics != best_metrics:
            raise RuntimeError('Reloaded selected adapter changed validation outputs; investigate numerical differences')
        (output/'training_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        write_comparison(output,baseline,best_metrics,best_epoch)
        write_plots(output,history,epoch_metrics,baseline)
        write_json(output/'status.json',{'status':'passed','test_set_used':False,'selected_epoch':best_epoch,
                                         'finished_utc':datetime.now(timezone.utc).isoformat()})
        print(f'Completed. Selected epoch {best_epoch}. Validation artifacts: {output}',flush=True)
        return {'baseline':baseline,'adapted':best_metrics,'selected_epoch':best_epoch}
    except Exception as exc:
        import traceback
        write_json(output/'status.json',{'status':'failed','test_set_used':False,
                                         'error_type':type(exc).__name__,'error':str(exc)})
        (output/'failure.txt').write_text(traceback.format_exc())
        raise
