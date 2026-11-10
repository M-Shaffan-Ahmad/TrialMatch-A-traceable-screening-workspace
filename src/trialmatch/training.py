"""Small explicit SFT loop helpers with token-weighted gradient accumulation."""


def pad_training_batch(examples, pad_token_id, device):
    import torch
    length = max(len(e['input_ids']) for e in examples)
    values = {key:[] for key in ('input_ids','attention_mask','labels')}
    for example in examples:
        padding = length-len(example['input_ids'])
        values['input_ids'].append(example['input_ids']+[pad_token_id]*padding)
        values['attention_mask'].append(example['attention_mask']+[0]*padding)
        values['labels'].append(example['labels']+[-100]*padding)
    return {key:torch.tensor(value,dtype=torch.long,device=device) for key,value in values.items()}


def epoch_batches(count, batch_size, seed, epoch):
    import numpy as np
    if count < 1 or batch_size < 1:
        raise ValueError('Training requires examples and a positive batch size')
    order = np.random.default_rng(seed+epoch).permutation(count).tolist()
    return [order[i:i+batch_size] for i in range(0,count,batch_size)]


def update_groups(batches, accumulation):
    if accumulation < 1:
        raise ValueError('Accumulation must be positive')
    return [batches[i:i+accumulation] for i in range(0,len(batches),accumulation)]


def train_epoch(model, encoded, optimizer, scheduler, device, pad_token_id,
                batch_size, accumulation, seed, epoch, global_step=0, on_update=None, example_weights=None):
    import torch
    if example_weights is not None:
        import math
        if len(example_weights) != len(encoded) or any(not math.isfinite(w) or w <= 0 for w in example_weights):
            raise ValueError('Expected one finite positive weight per training example')
    model.train()
    model.config.use_cache = False
    groups = update_groups(epoch_batches(len(encoded),batch_size,seed,epoch),accumulation)
    for group in groups:
        # Weight by target-token count, including the final incomplete group.
        target_count = sum(sum(x != -100 for x in encoded[i]['labels'][1:])
                           for indices in group for i in indices)
        if not target_count:
            raise ValueError('No shifted supervised tokens in an update group')
        denominator = (sum(example_weights[i]*sum(x != -100 for x in encoded[i]['labels'][1:])
                           for indices in group for i in indices)
                       if example_weights is not None else target_count)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = unweighted_sum = 0.0
        for indices in group:
            batch = pad_training_batch([encoded[i] for i in indices],pad_token_id,device)
            count = int((batch['labels'][:,1:] != -100).sum())
            if example_weights is None:
                output = model(**batch,use_cache=False)
                loss = output.loss
                numerator = loss*count
                raw_sum = loss.detach().item()*count
            else:
                import torch.nn.functional as functional
                # No labels are forwarded: avoid computing and discarding a second loss.
                output = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
                logits = output.logits[:,:-1,:].float()
                targets = batch['labels'][:,1:]
                token_loss = functional.cross_entropy(
                    logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),
                    ignore_index=-100,reduction='none').reshape(targets.shape)
                weights = torch.tensor([example_weights[i] for i in indices],device=device)
                numerator = (token_loss.sum(dim=1)*weights).sum()
                raw_sum = token_loss.detach().sum().item()
                loss = numerator/denominator
                del logits, targets, token_loss, weights
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            (numerator/denominator).backward()
            loss_sum += numerator.detach().item()
            unweighted_sum += raw_sum
            del output, loss, numerator, batch
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradient norm')
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        global_step += 1
        if on_update:
            on_update({'global_step':global_step,'epoch':epoch+1,
                       'loss':loss_sum/denominator,'unweighted_loss':unweighted_sum/target_count,
                       'weighted_supervised_tokens':denominator,'supervised_tokens':target_count,
                       'examples':sum(len(x) for x in group),'learning_rate':learning_rate,
                       'gradient_norm':float(norm)})
    optimizer.zero_grad(set_to_none=True)
    return global_step


def learning_rate_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step+1)/max(1,warmup_steps)
    return max(0.0,(total_steps-step)/max(1,total_steps-warmup_steps))


def checkpoint_score(metrics):
    overall = metrics['overall']
    evidence = overall['evidence']['micro']['f1']
    return (overall['classification']['macro_f1'], -1.0 if evidence is None else evidence)
