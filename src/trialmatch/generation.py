"""Library greedy baseline for validation; custom assignment decoders come later."""
import time

from trialmatch.data import build_messages
from trialmatch.evaluation import prediction_record


def predict_examples(model, tokenizer, examples, batch_size=4, max_new_tokens=192,
                     max_prompt_tokens=2048, on_record=None, prompt_version="v1"):
    import torch
    from transformers import GenerationConfig

    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    model.eval()
    tokenizer.padding_side = 'left'
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    settings = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                                use_cache=True)
    records = []
    def emit(record):
        records.append(record)
        if on_record:
            on_record(record)
    for offset in range(0, len(examples), batch_size):
        pending = examples[offset:offset+batch_size]
        accepted, encoded = [], []
        for example in pending:
            ids = tokenizer.apply_chat_template(build_messages(example, prompt_version), tokenize=True,
                                                 add_generation_prompt=True)
            if len(ids) > max_prompt_tokens or len(ids)+max_new_tokens > model.config.max_position_embeddings:
                emit(prediction_record(example, '', error='Overlength prompt rejected without truncation'))
            else:
                accepted.append(example)
                encoded.append({'input_ids':ids,'attention_mask':[1]*len(ids)})
        if not accepted:
            continue
        batch = tokenizer.pad(encoded, padding=True, return_tensors='pt').to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**batch, generation_config=settings)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        for i,example in enumerate(accepted):
            token_ids = generated[i,batch['input_ids'].shape[1]:].tolist()
            end = next((j+1 for j,t in enumerate(token_ids) if t in eos_ids),len(token_ids))
            token_ids = token_ids[:end]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            emit(prediction_record(example,text,generation={
                'prompt_tokens':sum(encoded[i]['attention_mask']), 'generated_tokens':len(token_ids),
                'hit_token_limit':len(token_ids)>=max_new_tokens and not any(t in eos_ids for t in token_ids),
                'batch_seconds_diagnostic_only':seconds,
            }))
    # Preserve identical evaluation ordering even if one prompt was rejected.
    by_id = {r['annotation_id']:r for r in records}
    return [by_id[e['annotation_id']] for e in examples]
