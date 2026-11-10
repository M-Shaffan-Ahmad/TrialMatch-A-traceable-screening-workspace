"""Target masking and real tiny-model adapter round trip; no benchmark examples."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trialmatch.sft import OverlengthExample, encode_supervised, one_optimizer_step, tokenizer_compatibility


class FakeTokenizer:
    """Character template fixture makes the supervision boundary inspectable."""
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = 1
    unk_token_id = None
    special_tokens_map = {'eos_token': '<end>'}
    chat_template = 'fixture'

    def get_vocab(self):
        return {'a': 2, 'b': 3}

    def get_added_vocab(self):
        return {'<end>': 1}

    def __len__(self):
        return 4

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ''.join(f'<{m["role"]}>' + m['content'] + '<end>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return self.encode(text)


class SupervisionTests(unittest.TestCase):
    def example(self):
        return {'annotation_id': 'synthetic',
                'input': {'note': '0. A 50-year-old patient.', 'criterion_type': 'inclusion',
                          'criterion_text': 'Age 18 or older'},
                'target': {'label': 'included', 'evidence_sentence_ids': [0]}}

    def test_prompt_masked_and_full_assistant_json_supervised(self):
        tokenizer = FakeTokenizer()
        encoded = encode_supervised(self.example(), tokenizer, 4096)
        boundary = next(i for i, label in enumerate(encoded['labels']) if label != -100)
        self.assertTrue(all(label == -100 for label in encoded['labels'][:boundary]))
        self.assertEqual(encoded['labels'][boundary:], encoded['input_ids'][boundary:])
        supervised = ''.join(chr(x) for x in encoded['labels'][boundary:])
        self.assertIn('"label": "included"', supervised)
        self.assertIn('"evidence_sentence_ids": [0]', supervised)
        self.assertNotIn('A 50-year-old patient', supervised)
        self.assertTrue(supervised.endswith('<end>'))

    def test_overlength_rejected_instead_of_truncated(self):
        with self.assertRaises(OverlengthExample):
            encode_supervised(self.example(), FakeTokenizer(), 10)

    def test_changed_token_ids_fail_compatibility(self):
        target, draft = FakeTokenizer(), FakeTokenizer()
        self.assertTrue(tokenizer_compatibility(target, draft)['compatible'])
        draft.get_vocab = lambda: {'a': 3, 'b': 2}
        self.assertFalse(tokenizer_compatibility(target, draft)['compatible'])

    def test_changed_special_token_fails_compatibility(self):
        draft = FakeTokenizer()
        draft.eos_token_id = 9
        self.assertFalse(tokenizer_compatibility(FakeTokenizer(), draft)['compatible'])


@unittest.skipUnless(all(importlib.util.find_spec(m) is not None for m in ['torch', 'peft', 'transformers']),
                     'Optional model libraries not installed')
class TinyModelTests(unittest.TestCase):
    def test_optimizer_update_and_adapter_reload(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=128)
        config._attn_implementation = 'eager'
        base = Qwen2ForCausalLM(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            base.save_pretrained(path / 'base')
            model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2,
                                                   lora_alpha=4, lora_dropout=0,
                                                   target_modules=['q_proj', 'v_proj']))
            ids = [3, 8, 5, 12, 7, 9, 4, 2]
            encoded = {'input_ids': ids, 'attention_mask': [1]*len(ids),
                       'labels': [-100]*4 + ids[4:]}
            result = one_optimizer_step(model, encoded, torch.device('cpu'))
            self.assertEqual(result['optimizer_steps'], 1)
            self.assertGreater(result['max_parameter_change'], 0)
            self.assertEqual(result['supervised_tokens'], 4)
            model.eval()
            inputs = torch.tensor([ids])
            with torch.inference_mode():
                expected = model(inputs).logits
            model.save_pretrained(path / 'adapter')
            loaded = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(path / 'base'),
                                               path / 'adapter').eval()
            with torch.inference_mode():
                actual = loaded(inputs).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
