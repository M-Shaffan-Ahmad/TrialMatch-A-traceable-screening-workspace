"""Meaningful integrity tests using synthetic records, never the held-out examples."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from trialmatch.data import build_messages, normalize, sentence_map
from prepare_data import split_patients


class DataContractTests(unittest.TestCase):
    def row(self):
        return dict(annotation_id=1, patient_id="synthetic", trial_id="synthetic-trial",
                    note="0. Patient is 50 years old.\n1. No diabetes is documented.",
                    criterion_type="inclusion", criterion_text="Age at least 18 years",
                    expert_eligibility="included", expert_sentences="[0]",
                    gpt4_explanation="LEAK_SENTINEL", gpt4_eligibility="LEAK_SENTINEL")

    def test_allowlist_blocks_generated_content(self):
        import json
        example = normalize(self.row())
        self.assertNotIn("LEAK_SENTINEL", json.dumps(example))
        prompt = json.dumps(build_messages(example))
        self.assertNotIn("expert_eligibility", prompt)
        self.assertNotIn("target", prompt)
        self.assertNotIn("synthetic-trial", prompt)

    def test_preserve_sentence_ids(self):
        self.assertEqual(sentence_map("4. First sentence.\n9. Second sentence."),
                         {4: "First sentence.", 9: "Second sentence."})

    def test_reject_dangling_evidence(self):
        row = self.row()
        row["expert_sentences"] = "[99]"
        with self.assertRaises(ValueError):
            normalize(row)

    def test_reject_wrong_label_for_type(self):
        row = self.row()
        row["expert_eligibility"] = "excluded"
        with self.assertRaises(ValueError):
            normalize(row)

    def test_empty_evidence_is_preserved(self):
        row = self.row()
        row.update(expert_sentences="[]", expert_eligibility="not enough information")
        self.assertEqual(normalize(row)["target"]["evidence_sentence_ids"], [])

    def test_fixed_split_is_disjoint_and_input_order_independent(self):
        patients = [f"patient-{i:02d}" for i in range(53)]
        actual = split_patients(patients + patients)
        self.assertEqual(actual, split_patients(reversed(patients)))
        self.assertEqual([len(v) for v in actual.values()], [37, 8, 8])
        import numpy as np
        reference = np.array(sorted(patients))
        np.random.default_rng(42).shuffle(reference)
        self.assertEqual(actual["train"], reference[:37].tolist())
        self.assertEqual(actual["validation"], reference[37:45].tolist())
        self.assertEqual(actual["test"], reference[45:].tolist())
        flattened = sum(actual.values(), [])
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_unexpected_patient_count_fails(self):
        with self.assertRaises(ValueError):
            split_patients(["only-one"])


if __name__ == "__main__":
    unittest.main()
