"""Educational dense and tiled exact attention, not a fused FlashAttention kernel."""
import math
import torch


def _validate(q,k,v,causal,query_offset):
    if q.ndim!=4 or k.ndim!=4 or v.ndim!=4:
        raise ValueError('Expected [batch, heads, sequence, channels] tensors')
    if q.shape[:2]!=k.shape[:2] or k.shape[:3]!=v.shape[:3] or q.shape[-1]!=k.shape[-1]:
        raise ValueError('Incompatible shapes (expand GQA heads before calling)')
    if min(q.shape+k.shape+v.shape)<1 or not q.is_floating_point():
        raise ValueError('Nonempty floating tensors required')
    if any(x.device!=q.device or x.dtype!=q.dtype for x in (k,v)):
        raise ValueError('Q/K/V must share dtype and device')
    if query_offset<0 or (causal and query_offset+q.shape[-2]>k.shape[-2]):
        raise ValueError('Invalid absolute causal query positions')


def _scores(q,k,qstart,kstart,causal,offset):
    scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    if causal:
        qi=torch.arange(qstart,qstart+q.shape[-2],device=q.device)+offset
        kj=torch.arange(kstart,kstart+k.shape[-2],device=q.device)
        scores=scores.masked_fill(kj[None,:]>qi[:,None],-torch.inf)
    return scores


def reference_attention(q,k,v,causal=True,query_offset=0):
    _validate(q,k,v,causal,query_offset)
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    scores=_scores(q.to(dtype),k.to(dtype),0,0,causal,query_offset)
    probabilities=torch.softmax(scores,dim=-1)
    return (probabilities@v.to(dtype)).to(q.dtype)


def tiled_attention(q,k,v,causal=True,query_offset=0,query_tile=32,key_tile=64):
    _validate(q,k,v,causal,query_offset)
    if query_tile<1 or key_tile<1:
        raise ValueError('Tile sizes must be positive')
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    original_dtype=q.dtype
    q,k,v=(x.to(dtype) for x in (q,k,v))
    result=torch.empty((*q.shape[:-1],v.shape[-1]),device=q.device,dtype=dtype)
    for qs in range(0,q.shape[-2],query_tile):
        qb=q[...,qs:qs+query_tile,:]
        maximum=torch.full(qb.shape[:-1]+(1,),-torch.inf,device=q.device,dtype=dtype)
        normalizer=torch.zeros_like(maximum)
        accumulator=torch.zeros(qb.shape[:-1]+(v.shape[-1],),device=q.device,dtype=dtype)
        for ks in range(0,k.shape[-2],key_tile):
            scores=_scores(qb,k[...,ks:ks+key_tile,:],qs,ks,causal,query_offset)
            updated=torch.maximum(maximum,scores.amax(dim=-1,keepdim=True))
            # A tile may be entirely masked for a row: avoid -inf - -inf NaNs.
            finite=torch.isfinite(updated)
            safe=torch.where(finite,updated,torch.zeros_like(updated))
            scale=torch.where(torch.isfinite(maximum),torch.exp(maximum-safe),torch.zeros_like(maximum))
            probabilities=torch.exp(scores-safe)
            accumulator=accumulator*scale+probabilities@v[...,ks:ks+key_tile,:]
            normalizer=normalizer*scale+probabilities.sum(dim=-1,keepdim=True)
            maximum=updated
        result[...,qs:qs+query_tile,:]=accumulator/normalizer
    # Preserve the input storage dtype while accumulating softmax in FP32/FP64.
    return result.to(original_dtype)
