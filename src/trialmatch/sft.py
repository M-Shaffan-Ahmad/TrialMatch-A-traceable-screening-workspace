"""Assistant-only supervised examples and a one-step feasibility check."""
import json

from trialmatch.data import build_messages


class OverlengthExample(ValueError):
    pass


def encode_supervised(example, tokenizer, max_length, prompt_version="v1"):
    """Keep every source token or reject; never silently truncate evidence."""
    messages = build_messages(example, prompt_version)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    answer = json.dumps(example['target'], ensure_ascii=False, sort_keys=True)
    complete = tokenizer.apply_chat_template(
        messages + [{'role': 'assistant', 'content': answer}],
        tokenize=True, add_generation_prompt=False,
    )
    if complete[:len(prompt)] != prompt:
        raise ValueError('Chat template does not produce a stable assistant prefix')
    if len(complete) > max_length:
        raise OverlengthExample(f'annotation {example["annotation_id"]}: {len(complete)} > {max_length}')
    if len(complete) <= len(prompt):
        raise ValueError('No assistant target tokens')
    return {'input_ids': complete, 'attention_mask': [1] * len(complete),
            'labels': [-100] * len(prompt) + complete[len(prompt):]}


def tokenizer_compatibility(target, draft):
    """Fail closed unless the token-ID mapping and special-token semantics agree."""
    probes = [
        'Patient is 58 years old.\n0. No diabetes documented.',
        '{"label":"not enough information","evidence_sentence_ids":[0,2]}',
        'HbA1c ≥ 6.5%; creatinine 1.2 mg/dL; β-blocker.',
        '<|im_start|>assistant\n<|im_end|>',
    ]
    special_names = ('bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id')
    checks = {
        'vocabulary_equal': target.get_vocab() == draft.get_vocab(),
        'added_vocabulary_equal': target.get_added_vocab() == draft.get_added_vocab(),
        'special_token_ids_equal': all(getattr(target, key) == getattr(draft, key)
                                       for key in special_names),
        'special_tokens_map_equal': target.special_tokens_map == draft.special_tokens_map,
        'probe_encodings_equal': all(target.encode(p, add_special_tokens=False)
                                     == draft.encode(p, add_special_tokens=False) for p in probes),
        'chat_templates_equal': target.chat_template == draft.chat_template,
    }
    return {'compatible': all(checks.values()), 'checks': checks,
            'target_tokenizer_length': len(target), 'draft_tokenizer_length': len(draft)}


def one_optimizer_step(model, encoded, device, learning_rate=2e-4):
    """One real update; checks finite loss/gradients and an actual adapter change."""
    import time
    import torch

    batch = {key: torch.tensor([value], dtype=torch.long, device=device)
             for key, value in encoded.items()}
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    if not parameters or any('lora_' not in name for name, _ in parameters):
        raise ValueError('Expected LoRA-only trainable parameters')
    name, tracked = next((name, p) for name, p in parameters if 'lora_B' in name)
    before = tracked.detach().float().clone()
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=learning_rate)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch, use_cache=False)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite training loss')
    loss.backward()
    gradients = [p.grad for _, p in parameters if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('Missing or non-finite adapter gradients')
    grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0)
    optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    delta = (tracked.detach().float() - before).abs().max().item()
    if delta == 0:
        raise RuntimeError('Optimizer did not update the tracked adapter weight')
    result = {
        'optimizer_steps': 1, 'batch_size': 1,
        'sequence_tokens': len(encoded['input_ids']),
        'supervised_tokens': sum(x != -100 for x in encoded['labels']),
        'train_loss': loss.item(), 'gradient_norm_before_clipping': float(grad_norm),
        'tracked_parameter': name, 'max_parameter_change': delta,
        'step_seconds': duration,
        'peak_allocated_gib': (torch.cuda.max_memory_allocated() / 2**30
                               if device.type == 'cuda' else None),
    }
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return result
