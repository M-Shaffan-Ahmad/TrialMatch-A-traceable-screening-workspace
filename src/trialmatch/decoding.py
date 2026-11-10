"""Explicit single-sequence greedy decoders; no model.generate or assisted API."""
import time
import torch
from transformers import DynamicCache


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _check(model, ids, count):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or count < 0:
        raise ValueError('Expected a nonempty unpadded single sequence and nonnegative output length')
    if ids.shape[1]+count > model.config.max_position_embeddings:
        raise ValueError('Requested sequence exceeds model context')
    model.eval()


def _forward(model, ids, cache=None):
    past = 0 if cache is None else cache.get_seq_length()
    positions = torch.arange(past,past+ids.shape[1],device=ids.device)
    return model(input_ids=ids,attention_mask=torch.ones((1,past+ids.shape[1]),
                  dtype=torch.long,device=ids.device),position_ids=positions[None,:],
                  cache_position=positions,past_key_values=cache,use_cache=cache is not None)


def _end(token,eos):
    return token in eos


def _result(tokens,start,prefill,device,**extra):
    synchronize(device)
    total=time.perf_counter()-start
    return {'token_ids':tokens,'prefill_seconds':prefill,'decode_seconds':total-prefill,
            'total_seconds':total,'tokens_per_second':len(tokens)/total if total else 0,
            'decode_tokens_per_second':max(0,len(tokens)-1)/(total-prefill) if total>prefill else 0,
            **extra}


@torch.inference_mode()
def greedy(model,input_ids,max_new_tokens=64,use_cache=True,eos_token_ids=()):
    _check(model,input_ids,max_new_tokens)
    synchronize(input_ids.device)
    start=time.perf_counter()
    if not max_new_tokens:
        return _result([],start,0,input_ids.device)
    cache=DynamicCache() if use_cache else None
    ids=input_ids
    output=_forward(model,ids,cache)
    if use_cache:
        cache=output.past_key_values
    token=int(output.logits[0,-1].argmax())
    del output
    synchronize(ids.device)
    prefill=time.perf_counter()-start
    tokens=[token]
    while len(tokens)<max_new_tokens and not _end(token,eos_token_ids):
        next_ids=torch.tensor([[token]],device=ids.device)
        if use_cache:
            output=_forward(model,next_ids,cache)
            cache=output.past_key_values
        else:
            ids=torch.cat([ids,next_ids],dim=1)
            output=_forward(model,ids)
        token=int(output.logits[0,-1].argmax())
        tokens.append(token)
        del output
    return _result(tokens,start,prefill,input_ids.device,
                   cache_tokens=cache.get_seq_length() if cache is not None else 0)


@torch.inference_mode()
def speculative(target,draft,input_ids,max_new_tokens=64,draft_tokens=4,eos_token_ids=()):
    """Target state caches the entire committed prefix, with its next-token logits.

    Verify candidate j with logits after candidate j-1 (or the saved prefix logits).
    Crop both caches to the accepted prefix, then feed the correction/bonus token
    through both models. Never retain rejected candidates in either cache.
    """
    _check(target,input_ids,max_new_tokens)
    _check(draft,input_ids,max_new_tokens)
    if draft_tokens<1:
        raise ValueError('Draft block must be positive')
    synchronize(input_ids.device)
    start=time.perf_counter()
    if not max_new_tokens:
        return _result([],start,0,input_ids.device,drafted=0,accepted=0,acceptance_rate=0,rounds=[])
    tc,dc=DynamicCache(),DynamicCache()
    out=_forward(target,input_ids,tc)
    tc,tn=out.past_key_values,out.logits[:,-1,:]
    del out
    out=_forward(draft,input_ids,dc)
    dc,dn=out.past_key_values,out.logits[:,-1,:]
    del out
    synchronize(input_ids.device)
    prefill=time.perf_counter()-start
    generated=[]
    drafted=accepted=0
    rounds=[]
    while len(generated)<max_new_tokens:
        prefix=tc.get_seq_length()
        assert prefix==dc.get_seq_length()==input_ids.shape[1]+len(generated)
        candidates=[]
        budget=min(draft_tokens,max_new_tokens-len(generated))
        for _ in range(budget):
            token=int(dn[0].argmax())
            candidates.append(token)
            out=_forward(draft,torch.tensor([[token]],device=input_ids.device),dc)
            dc,dn=out.past_key_values,out.logits[:,-1,:]
            del out
            if _end(token,eos_token_ids):
                break
        out=_forward(target,torch.tensor([candidates],device=input_ids.device),tc)
        tc=out.past_key_values
        verified=out.logits
        matched=0
        for i,token in enumerate(candidates):
            expected=int((tn if i==0 else verified[:,i-1,:])[0].argmax())
            if token!=expected:
                break
            matched+=1
        drafted+=len(candidates)
        accepted+=matched
        generated.extend(candidates[:matched])
        rounds.append({'drafted':len(candidates),'accepted':matched,
                       'rejected':matched<len(candidates),'prefix_cache_tokens':prefix})
        if (generated and _end(generated[-1],eos_token_ids)) or len(generated)>=max_new_tokens:
            # Even on return, explicitly discard any uncommitted suffix.
            tc.crop(prefix+matched)
            dc.crop(prefix+matched)
            del out,verified
            break
        correction=int((tn if matched==0 else verified[:,matched-1,:])[0].argmax())
        # When every candidate matched, verified[:, matched-1] predicts a bonus token.
        tc.crop(prefix+matched)
        dc.crop(prefix+matched)
        del out,verified
        generated.append(correction)
        token_tensor=torch.tensor([[correction]],device=input_ids.device)
        out=_forward(target,token_tensor,tc)
        tc,tn=out.past_key_values,out.logits[:,-1,:]
        del out
        out=_forward(draft,token_tensor,dc)
        dc,dn=out.past_key_values,out.logits[:,-1,:]
        del out
        if _end(correction,eos_token_ids):
            break
    return _result(generated,start,prefill,input_ids.device,drafted=drafted,accepted=accepted,
                   acceptance_rate=accepted/drafted if drafted else 0,rounds=rounds,
                   target_cache_tokens=tc.get_seq_length(),draft_cache_tokens=dc.get_seq_length())
