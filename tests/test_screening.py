from pathlib import Path
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trialmatch.screening import (prepare_note,prepare_criteria,result_from_output,
                                  aggregate,assessment,report_html,input_fingerprint)


class ScreeningTests(unittest.TestCase):
    def test_note_numbering_preserves_original_ids_and_rejects_mixed_or_duplicates(self):
        note,mapping,origin=prepare_note('Patient is 50.\n\nNo diabetes.')
        self.assertEqual(note,'0. Patient is 50.\n1. No diabetes.')
        self.assertEqual(mapping,{0:'Patient is 50.',1:'No diabetes.'})
        self.assertEqual(prepare_note('4. Patient is 50.\n9. No diabetes.')[1],{4:'Patient is 50.',9:'No diabetes.'})
        for bad in ['', '0. Patient\nUnnumbered', '0. One\n0. Two']:
            with self.assertRaises(ValueError): prepare_note(bad)

    def test_criterion_types_limits_and_blank_lines(self):
        rows=prepare_criteria('Age\n\nConsent','Diabetes')
        self.assertEqual([r['criterion_id'] for r in rows],[1,2,3])
        self.assertEqual(rows[-1]['criterion_type'],'exclusion')
        for a,b in [('', ''),('x\n'*21,''),('x'*2001,'')]:
            with self.assertRaises(ValueError): prepare_criteria(a,b)

    def result(self,label,kind='inclusion',evidence=True,valid=True):
        criterion={'criterion_id':1,'criterion_type':kind,'criterion_text':'Synthetic criterion'}
        raw=json.dumps({'label':label,'evidence_sentence_ids':[0] if evidence else []}) if valid else 'not json'
        return result_from_output(criterion,'0. Original text.',raw,{})

    def test_aggregate_never_promotes_unknown_invalid_uncited_or_all_na(self):
        self.assertEqual(aggregate([self.result('included')])['outcome'],'No conflict identified in supplied criteria')
        for r in [self.result('included',evidence=False),self.result('not enough information'),self.result('included',valid=False)]:
            self.assertEqual(aggregate([r])['outcome'],'Needs review / more information')
        self.assertEqual(aggregate([self.result('not applicable',evidence=False)])['outcome'],'Unassessed')
        for r in [self.result('not included'),self.result('excluded','exclusion')]:
            self.assertEqual(aggregate([r])['outcome'],'Potential criterion conflict')

    def test_source_citations_and_html_escape(self):
        note='0. <script>alert("patient")</script>'
        criterion={'criterion_id':1,'criterion_type':'inclusion','criterion_text':'<img src=x onerror=alert(1)>'}
        result=result_from_output(criterion,note,'{"label":"included","evidence_sentence_ids":[0]}',{})
        self.assertEqual(result['evidence'][0]['text'],'<script>alert("patient")</script>')
        value=assessment(note,[criterion],[result],{'model':'fixture'},'Cached greedy','Original IDs')
        exported=report_html(value)
        self.assertNotIn('<script>',exported)
        self.assertNotIn('<img src=',exported)
        self.assertIn('&lt;script&gt;',exported)
        self.assertEqual(value['input_sha256'],input_fingerprint(note,[criterion]))
        bad=result_from_output(criterion,note,'{"label":"included","evidence_sentence_ids":[99]}',{})
        self.assertFalse(bad['valid'])
        self.assertFalse(bad['evidence'])


class BrowserTests(unittest.TestCase):
    def test_form_validation_screen_exports_comparison_and_stale_input(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        from unittest.mock import patch
        import trialmatch.app_runtime as runtime_module
        class FixtureRuntime:
            # Test-only fixture. Production has no simulated prediction path.
            provenance={'target':'test fixture','device':'CPU'}
            def screen(self,note,criteria,mode,max_new_tokens,on_result=None):
                rows=[]
                for criterion in criteria:
                    label='included' if criterion['criterion_type']=='inclusion' else 'not excluded'
                    rows.append(result_from_output(criterion,note,json.dumps({'label':label,'evidence_sentence_ids':[0]}),{'total_seconds':.1}))
                if on_result: on_result(len(rows),len(rows))
                return rows
            def compare(self,note,criterion,max_new_tokens,repeats):
                return {'all_tokens_equal':True,'summaries':[{'mode':mode,'median_seconds':.1,
                    'median_tokens_per_second':10,'median_prefill_seconds':.01,'median_decode_seconds':.09,
                    'output_tokens':1,'all_tokens_equal':True,'acceptance_rate':None,'peak_allocated_mib':None}
                    for mode in runtime_module.MODES], 'note':note,'criterion':criterion,
                    'max_new_tokens':max_new_tokens,'repeats':repeats,'decoded_text':'{}',
                    'provenance':self.provenance,'measurements':[]}
        st.cache_resource.clear()
        import tempfile,os
        with tempfile.TemporaryDirectory() as adapter_dir, \
             patch.dict(os.environ,{'TRIALMATCH_ADAPTER_DIR':adapter_dir}), \
             patch.object(runtime_module,'ModelRuntime',lambda _:FixtureRuntime()):
            (Path(adapter_dir)/'adapter_model.safetensors').write_bytes(b'test-only placeholder')
            app=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
            self.assertFalse(app.exception)
            app.button(key='screen').click().run()
            self.assertTrue(app.error)
            app.text_area(key='patient').set_value('Patient is 50.').run()
            app.text_area(key='inclusion').set_value('Age above 18.').run()
            app.button(key='screen').click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['assessment']['aggregate']['outcome'],'No conflict identified in supplied criteria')
            self.assertEqual(len(app.get('download_button')),2)
            app.button(key='compare').click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['comparison']['all_tokens_equal'])
            app.text_area(key='patient').set_value('A changed patient record.').run()
            self.assertFalse(app.exception)
            self.assertNotIn('assessment',app.session_state)
            self.assertNotIn('comparison',app.session_state)
        st.cache_resource.clear()


class RuntimeIntegrationTests(unittest.TestCase):
    def test_real_tiny_models_use_all_custom_modes_and_repeated_live_comparison(self):
        import threading
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
        from trialmatch.app_runtime import ModelRuntime,MODES
        torch.set_num_threads(1)
        torch.manual_seed(42)
        backend=Tokenizer(WordLevel({'<pad>':0,'<unk>':1,'<eos>':2,'patient':3},unk_token='<unk>'))
        backend.pre_tokenizer=WhitespaceSplit()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',unk_token='<unk>',eos_token='<eos>')
        tokenizer.chat_template=("{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' <eos> ' }}{% endfor %}"
                                 "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}")
        def tiny():
            return Qwen2ForCausalLM(Qwen2Config(vocab_size=4,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                max_position_embeddings=2048,pad_token_id=0,eos_token_id=2)).eval()
        engine=ModelRuntime.__new__(ModelRuntime)
        engine.lock=threading.RLock()
        engine.tokenizer=tokenizer
        engine.device=torch.device('cpu')
        engine.target,engine.draft=tiny(),tiny()
        engine.provenance={'model':'tiny-random-test-only'}
        note,_,_=prepare_note('Synthetic patient.')
        criteria=prepare_criteria('Synthetic criterion.','')
        results=[engine.screen(note,criteria,mode,8) for mode in MODES]
        self.assertEqual([r[0]['decoding']['token_ids'] for r in results],
                         [results[0][0]['decoding']['token_ids']]*3)
        compared=engine.compare(note,criteria[0],8,3)
        self.assertTrue(compared['all_tokens_equal'])
        self.assertEqual(len(compared['measurements']),9)
        self.assertTrue(all(row['peak_allocated_mib'] is None for row in compared['summaries']))
        # Random weights are not valid clinical predictions; parser must retain failure.
        self.assertTrue(all(not r[0]['valid'] for r in results))


if __name__=='__main__': unittest.main()
