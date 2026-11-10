import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trialmatch.evaluation import (ALL_LABELS, INVALID, classification_metrics,
                                   evidence_metrics, evaluate_records, parse_prediction)


def record(gold='included',predicted='included',gold_ids=None,predicted_ids=None,valid=True,annotation=0,kind='inclusion'):
    return {'annotation_id':annotation,'criterion_type':kind,'gold_label':gold,
            'label':predicted,'gold_evidence_sentence_ids':gold_ids or [],
            'evidence_sentence_ids':predicted_ids or [],'valid':valid}


class PredictionTests(unittest.TestCase):
    def parse(self,text):
        return parse_prediction(text,'inclusion','0. Patient is 50.\n1. Patient has diabetes.')

    def test_plain_and_fenced_json(self):
        value={'label':'included','evidence_sentence_ids':[1,0]}
        plain=json.dumps(value)
        self.assertTrue(self.parse(plain)['valid'])
        parsed=self.parse('```json\n'+plain+'\n```')
        self.assertEqual(parsed['evidence_sentence_ids'],[0,1])
        self.assertEqual(parsed['normalization'],'stripped_markdown_fence')

    def test_reject_bad_categories_ids_duplicates_and_trailing_prose(self):
        cases=[
            '{"label":"excluded","evidence_sentence_ids":[]}',
            '{"label":"included","evidence_sentence_ids":[99]}',
            '{"label":"included","evidence_sentence_ids":[true]}',
            '{"label":"included","evidence_sentence_ids":[0,0]}',
            '{"label":"included","evidence_sentence_ids":[-1]}',
            '{"label":"included","evidence_sentence_ids":["0"]}',
            '{"label":"included","label":"not included","evidence_sentence_ids":[]}',
            '{"label":"included","evidence_sentence_ids":[],"explanation":"guess"}',
            '{"label":"included","evidence_sentence_ids":[]} I think so.',
            '{"label": "included"', 'null', '[]',
        ]
        for case in cases:
            with self.subTest(case=case):
                parsed=self.parse(case)
                self.assertFalse(parsed['valid'])
                self.assertEqual(parsed['label'],INVALID)


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_classification_and_invalid_column(self):
        records=[record(annotation=0),record(predicted='not included',annotation=1),
                 record(gold='not included',predicted='included',annotation=2),
                 record(gold='not included',valid=False,annotation=3)]
        metric=classification_metrics(records,['included','not included'])
        self.assertEqual(metric['accuracy'],.25)
        self.assertEqual(metric['per_class']['included']['precision'],.5)
        self.assertEqual(metric['per_class']['included']['recall'],.5)
        self.assertEqual(metric['macro_f1'],.25)
        self.assertEqual(metric['confusion_matrix']['counts'],[[1,1,0],[1,0,1]])

    def test_evidence_micro_and_macro_empty_conventions(self):
        records=[record(gold_ids=[0,1],predicted_ids=[1,2]),record(annotation=1),
                 record(gold_ids=[3],annotation=2)]
        metric=evidence_metrics(records)
        self.assertEqual((metric['tp'],metric['fp'],metric['fn']),(1,1,2))
        self.assertEqual(metric['micro']['precision'],.5)
        self.assertAlmostEqual(metric['micro']['recall'],1/3)
        self.assertAlmostEqual(metric['micro']['f1'],.4)
        self.assertAlmostEqual(metric['macro']['f1'],.5)

    def test_invalid_empty_response_is_not_rewarded(self):
        metric=evidence_metrics([record(valid=False)])
        self.assertEqual(metric['macro']['f1'],0)
        self.assertIsNone(metric['micro']['f1'])
        self.assertEqual(metric['fp'],0)
        self.assertEqual(metric['invalid_examples'],1)

    def test_all_empty_micro_undefined_and_macro_one(self):
        metric=evidence_metrics([record()])
        self.assertIsNone(metric['micro']['f1'])
        self.assertEqual(metric['macro']['f1'],1)
        self.assertTrue(metric['micro_no_evidence_condition'])

    def test_fixed_label_sets_and_type_breakdown(self):
        result=evaluate_records([record(),record(gold='not excluded',predicted='not excluded',
                                                 annotation=1,kind='exclusion')])
        self.assertEqual(result['overall']['classification']['accuracy'],1)
        self.assertEqual(result['overall']['classification']['macro_f1'],2/6)
        self.assertEqual(len(result['overall']['classification']['per_class']),6)
        for value in result['by_criterion_type'].values():
            self.assertEqual(len(value['classification']['per_class']),4)
        with self.assertRaises(ValueError):
            evaluate_records([record(),record()])


if __name__=='__main__':
    unittest.main()
