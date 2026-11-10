import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trialmatch.training import epoch_batches, update_groups, checkpoint_score


class BatchTests(unittest.TestCase):
    def test_incomplete_update_group_covers_every_example(self):
        batches=epoch_batches(11,2,42,0)
        groups=update_groups(batches,4)
        self.assertEqual([sum(len(x) for x in g) for g in groups],[8,3])
        self.assertEqual(sorted(i for group in groups for indices in group for i in indices),list(range(11)))
        self.assertEqual(batches,epoch_batches(11,2,42,0))
        self.assertNotEqual(batches,epoch_batches(11,2,42,1))

    def test_classification_is_primary_selection_metric(self):
        def m(c,e):
            return {'overall':{'classification':{'macro_f1':c},'evidence':{'micro':{'f1':e}}}}
        self.assertGreater(checkpoint_score(m(.6,.1)),checkpoint_score(m(.5,.9)))
        self.assertGreater(checkpoint_score(m(.5,.2)),checkpoint_score(m(.5,.1)))


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ['torch','peft','transformers','matplotlib']),
                     'Optional training stack not installed')
class WorkflowTests(unittest.TestCase):
    def test_token_weighted_accumulation_matches_full_batch(self):
        import copy
        from types import SimpleNamespace
        import torch
        from trialmatch.training import train_epoch
        class LossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor(.2))
                self.config=SimpleNamespace(use_cache=False)
            def forward(self,input_ids,attention_mask,labels,use_cache):
                valid=labels[:,1:] != -100
                losses=(self.weight-input_ids[:,1:].float()/10)**2
                return SimpleNamespace(loss=losses[valid].mean())
        first=LossModel()
        second=copy.deepcopy(first)
        examples=[
            {'input_ids':[1,1],'attention_mask':[1,1],'labels':[-100,1]},
            {'input_ids':[1,5,5],'attention_mask':[1,1,1],'labels':[-100,5,5]},
            {'input_ids':[1,9,9,9],'attention_mask':[1,1,1,1],'labels':[-100,9,9,9]},
        ]
        for model,batch,accumulation in [(first,1,4),(second,3,1)]:
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1)
            train_epoch(model,examples,optimizer,scheduler,torch.device('cpu'),0,
                        batch,accumulation,42,0)
        torch.testing.assert_close(first.weight,second.weight,rtol=0,atol=1e-7)

    def test_real_tiny_model_resume_matches_uninterrupted_training(self):
        import json
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
        from safetensors.torch import load_file
        import trialmatch.workflow as workflow

        torch.set_num_threads(1)
        torch.manual_seed(17)
        vocabulary={'<pad>':0,'<unk>':1,'<eos>':2,'<|system|>':3,'<|user|>':4,
                    '<|assistant|>':5,'0.':6,'Patient':7,'is':8,'50.':9,'included':10}
        backend=Tokenizer(WordLevel(vocabulary,unk_token='<unk>'))
        backend.pre_tokenizer=WhitespaceSplit()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',pad_token='<pad>',eos_token='<eos>')
        tokenizer.chat_template=("{% for m in messages %}{{ '<|' + m['role'] + '|>\\n' + m['content'] + '\\n<eos>\\n' }}{% endfor %}"
                                 "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}")
        def example(annotation,patient):
            return {'annotation_id':annotation,'patient_id':patient,'trial_id':'synthetic-trial',
                    'input':{'note':'0. Patient is 50.','criterion_type':'inclusion','criterion_text':'Age 18 or older'},
                    'target':{'label':'included','evidence_sentence_ids':[0]}}
        training=[example(0,'synthetic-train'),example(1,'synthetic-train'),
                  example(4,'synthetic-train'),example(5,'synthetic-train')]
        training[-1]['target']['label']='not included'
        validation=[example(2,'synthetic-validation'),example(3,'synthetic-validation')]
        config={'prompt_version':'v2','class_weighting':'inverse_sqrt_capped','class_weight_cap':3.0,
                'seed':42,'epochs':2,'batch_size':1,'accumulation':2,'learning_rate':2e-4,
                'lora_rank':2,'lora_alpha':4,'lora_dropout':.05,'warmup_ratio':.1,'weight_decay':.01,
                'max_length':512,'eval_batch_size':2,'max_new_tokens':4}
        provenance={'model':{'repository':'local-random-fixture','revision':'synthetic-only'}}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            base=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer),hidden_size=16,
                intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                max_position_embeddings=1024,pad_token_id=0,eos_token_id=2))
            base.save_pretrained(root/'base')
            def load_base():
                return Qwen2ForCausalLM.from_pretrained(root/'base')
            complete=workflow.run_workflow(root/'complete',tokenizer,load_base,training,validation,
                                           config,provenance,torch.device('cpu'))
            original=workflow.train_epoch
            def interrupt_after_first(*args,**kwargs):
                # Epoch index is positional argument 9 in this explicit loop API.
                if args[9]==1:
                    raise RuntimeError('Synthetic disconnect before epoch two')
                return original(*args,**kwargs)
            with patch.object(workflow,'train_epoch',side_effect=interrupt_after_first):
                with self.assertRaisesRegex(RuntimeError,'Synthetic disconnect'):
                    workflow.run_workflow(root/'resumed',tokenizer,load_base,training,validation,
                                           config,provenance,torch.device('cpu'))
            resumed=workflow.run_workflow(root/'resumed',tokenizer,load_base,training,validation,
                                          config,provenance,torch.device('cpu'),resume=True)
            self.assertEqual(complete['adapted'],resumed['adapted'])
            for epoch in [1,2]:
                left=load_file(root/'complete'/f'epoch-{epoch:02d}'/'adapter/adapter_model.safetensors')
                right=load_file(root/'resumed'/f'epoch-{epoch:02d}'/'adapter/adapter_model.safetensors')
                for name in left:
                    torch.testing.assert_close(left[name],right[name],rtol=0,atol=0)
            report=json.loads((root/'resumed/checkpoint_reload.json').read_text())
            self.assertTrue(report['validation_outputs_equal'])
            self.assertTrue((root/'resumed/training-validation-curves.png').is_file())
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                changed={**config,'epochs':3}
                workflow.run_workflow(root/'resumed',tokenizer,load_base,training,validation,
                                       changed,provenance,torch.device('cpu'),resume=True)


if __name__=='__main__':
    unittest.main()
