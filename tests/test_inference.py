import copy
from pathlib import Path
import sys
import unittest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trialmatch.decoding import greedy,speculative
from trialmatch.attention import reference_attention,tiled_attention


class AttentionTests(unittest.TestCase):
    def test_shapes_mask_offsets_and_online_stability(self):
        torch.manual_seed(42)
        for nq,nk,offset,causal in [(1,1,0,True),(7,7,0,True),(5,13,8,True),(9,11,0,False)]:
            for factor in [1,50]:
                q=torch.randn(2,3,nq,8,dtype=torch.float64)*factor
                k=torch.randn(2,3,nk,8,dtype=torch.float64)*factor
                v=torch.randn(2,3,nk,6,dtype=torch.float64)
                expected=reference_attention(q,k,v,causal,offset)
                for qt,kt in [(1,1),(3,4),(32,64)]:
                    actual=tiled_attention(q,k,v,causal,offset,qt,kt)
                    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
                mask=torch.arange(nk)[None,:] <= (torch.arange(nq)[:,None]+offset) if causal else None
                independent=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                torch.testing.assert_close(expected,independent,atol=1e-10,rtol=1e-10)

    def test_future_values_do_not_change_earlier_causal_outputs(self):
        q,k,v=[torch.randn(1,2,9,8) for _ in range(3)]
        before=tiled_attention(q,k,v,query_tile=3,key_tile=2)
        v[:,:,5:,:]=1000
        after=tiled_attention(q,k,v,query_tile=3,key_tile=2)
        torch.testing.assert_close(before[:,:,:5],after[:,:,:5])


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.target=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,
            intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,max_position_embeddings=256,attention_dropout=0)).eval()
        torch.manual_seed(17)
        cls.draft=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=16,
            intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
            num_key_value_heads=1,max_position_embeddings=256,attention_dropout=0)).eval()

    def test_cached_and_uncached_match_library_and_each_other(self):
        for length,count in [(1,1),(7,9),(19,17)]:
            ids=(torch.arange(length)[None,:]%60)+1
            cached=greedy(self.target,ids,count,True)
            uncached=greedy(self.target,ids,count,False)
            with torch.inference_mode():
                library=self.target.generate(ids,attention_mask=torch.ones_like(ids),
                    max_new_tokens=count,do_sample=False,eos_token_id=None,pad_token_id=0)
            self.assertEqual(cached['token_ids'],uncached['token_ids'])
            self.assertEqual(cached['token_ids'],library[0,length:].tolist())
            self.assertEqual(cached['cache_tokens'],length+count-1)

    def test_speculation_rejection_and_all_accepted_rounds(self):
        for draft in [self.draft,copy.deepcopy(self.target)]:
            for block in [1,4,8]:
                for length,count in [(1,1),(7,9),(19,17)]:
                    ids=(torch.arange(length)[None,:]%60)+1
                    expected=greedy(self.target,ids,count)['token_ids']
                    actual=speculative(self.target,draft,ids,count,block)
                    self.assertEqual(actual['token_ids'],expected)
                    self.assertEqual(actual['target_cache_tokens'],length+count)
                    self.assertEqual(actual['target_cache_tokens'],actual['draft_cache_tokens'])
                    if draft.config.hidden_size==self.target.config.hidden_size:
                        self.assertEqual(actual['acceptance_rate'],1)
                    elif count>1:
                        self.assertTrue(any(r['rejected'] for r in actual['rounds']))

    def test_scripted_partial_acceptance_and_rejection_positions(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Cache:
            def __init__(self): self.length=0
            def get_seq_length(self): return self.length
            def crop(self,length): self.length=length
        class ScheduledModel:
            config=SimpleNamespace(max_position_embeddings=256)
            def __init__(self,is_draft): self.is_draft=is_draft
            def eval(self): return self
            def __call__(self,input_ids,attention_mask,position_ids,cache_position,past_key_values,use_cache):
                past=0 if past_key_values is None else past_key_values.length
                self_test.assertEqual(position_ids[0].tolist(),list(range(past,past+input_ids.shape[1])))
                self_test.assertEqual(attention_mask.shape[1],past+input_ids.shape[1])
                logits=torch.full((1,input_ids.shape[1],64),-100.0)
                for i in range(input_ids.shape[1]):
                    next_position=past+i+1
                    token=(next_position*3)%60+1
                    if self.is_draft and next_position%3==2: token=(token+1)%64
                    logits[0,i,token]=100
                if past_key_values is not None: past_key_values.length+=input_ids.shape[1]
                return SimpleNamespace(logits=logits,past_key_values=past_key_values)
        self_test=self
        ids=torch.tensor([[1]])
        with patch('trialmatch.decoding.DynamicCache',Cache):
            target,draft=ScheduledModel(False),ScheduledModel(True)
            expected=greedy(target,ids,13)['token_ids']
            actual=speculative(target,draft,ids,13,4)
            self.assertEqual(actual['token_ids'],expected)
            self.assertTrue(any(0<r['accepted']<r['drafted'] for r in actual['rounds']))
            rejected_first=speculative(target,draft,torch.tensor([[1,2]]),7,4)
            self.assertEqual(rejected_first['rounds'][0]['accepted'],0)
            self.assertEqual(rejected_first['token_ids'],greedy(target,torch.tensor([[1,2]]),7)['token_ids'])
            self.assertEqual(actual['target_cache_tokens'],14)

    def test_eos_and_zero_output_limit(self):
        ids=torch.tensor([[1,3,4]])
        token=greedy(self.target,ids,1)['token_ids'][0]
        for draft in [self.draft,copy.deepcopy(self.target)]:
            self.assertEqual(speculative(self.target,draft,ids,8,4,[token])['token_ids'],[token])
        self.assertEqual(greedy(self.target,ids,0)['token_ids'],[])
        self.assertEqual(speculative(self.target,self.draft,ids,0)['token_ids'],[])
        with self.assertRaises(ValueError):
            greedy(self.target,ids,300)


if __name__=='__main__':
    unittest.main()
