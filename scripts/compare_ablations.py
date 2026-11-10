"""Compare two completed validation ablations, enforcing a one-change A/B protocol."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare_runs(unweighted, balanced, output):
    unweighted,balanced,output = map(Path,(unweighted,balanced,output))
    runs = [json.loads((p/'run.json').read_text()) for p in (unweighted,balanced)]
    statuses = [json.loads((p/'status.json').read_text()) for p in (unweighted,balanced)]
    if any(s.get('status')!='passed' or s.get('test_set_used') is not False for s in statuses):
        raise ValueError('Both validation runs must have passed without test-set use')
    first,second = [r['config'] for r in runs]
    if first.get('prompt_version')!='v2' or second.get('prompt_version')!='v2':
        raise ValueError('Both ablations must use prompt v2')
    if (first.get('class_weighting'),second.get('class_weighting'))!=('none','inverse_sqrt_capped'):
        raise ValueError('Expected unweighted A and class-weighted B')
    exclude = {'class_weighting','experiment_id'}
    if {k:v for k,v in first.items() if k not in exclude}!={k:v for k,v in second.items() if k not in exclude}:
        raise ValueError('A/B training or generation settings differ beyond the balancing change')
    if runs[0]['provenance'] != runs[1]['provenance']:
        raise ValueError('A/B model, data, code or runtime provenance differs')
    comparison = [json.loads((p/'comparison.json').read_text()) for p in (unweighted,balanced)]
    if comparison[0]['baseline'] != comparison[1]['baseline']:
        raise ValueError('Untouched-base baseline metrics differ between A and B')
    histories = [[json.loads(line) for line in (p/'training_history.jsonl').read_text().splitlines()]
                 for p in (unweighted,balanced)]
    budgets = [[(r['global_step'],r['examples'],r['supervised_tokens']) for r in history] for history in histories]
    if budgets[0] != budgets[1]:
        raise ValueError('A/B exposure or optimizer-update budgets differ')
    output.mkdir(parents=True,exist_ok=True)
    result = {'split':'validation','test_set_used':False,'protocol_check':'passed',
              'only_training_change':'capped inverse-square-root class weights',
              'A':comparison[0],'B':comparison[1],
              'runtime':runs[0]['provenance']['environment']}
    (output/'ablation-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    a,b = comparison[0]['adapted']['overall'],comparison[1]['adapted']['overall']
    rows = [('Accuracy',a['classification']['accuracy'],b['classification']['accuracy']),
            ('Classification macro-F1',a['classification']['macro_f1'],b['classification']['macro_f1']),
            ('Evidence micro-F1',a['evidence']['micro']['f1'],b['evidence']['micro']['f1']),
            ('Evidence macro-F1',a['evidence']['macro']['f1'],b['evidence']['macro']['f1'])]
    text = '# Prompt v2 and class-balancing ablation\n\n'
    text += 'Validation only. Same model, prompt, dataset, runtime, example exposure and update budget.\n\n'
    text += '| Metric | A: prompt v2 | B: v2 + weighting | B minus A |\n|---|---:|---:|---:|\n'
    for name,x,y in rows:
        values = ['undefined' if n is None else f'{n:.4f}' for n in (x,y,None if x is None or y is None else y-x)]
        text += f'| {name} | '+ ' | '.join(values)+' |\n'
    text += f'| Invalid outputs | {a["invalid_outputs"]} | {b["invalid_outputs"]} | {b["invalid_outputs"]-a["invalid_outputs"]} |\n'
    text += '\n| Label | Support | A recall | B recall | A F1 | B F1 |\n|---|---:|---:|---:|---:|---:|\n'
    for label,values in a['classification']['per_class'].items():
        other=b['classification']['per_class'][label]
        text += f'| {label} | {values["support"]} | {values["recall"]:.4f} | {other["recall"]:.4f} | {values["f1"]:.4f} | {other["f1"]:.4f} |\n'
    text += '\nEach run selects its own epoch using the same validation rule. This compares checkpoint-selected training recipes; validation selection can be optimistic. No test results are claimed.\n'
    text += '\nTraining loss values optimize different objectives and should not be compared as identical quantities; each log also records unweighted token loss.\n'
    reference_path=ROOT/'docs/reference-experiment-01.json'
    if reference_path.exists():
        reference=json.loads(reference_path.read_text())
        origin=reference['selected_metrics']['overall']
        text += ('\nHistorical original prompt-v1 run: accuracy '
                 f'{origin["classification"]["accuracy"]:.4f}, macro-F1 {origin["classification"]["macro_f1"]:.4f}, '
                 f'hardware {reference["hardware"]}. New A/B hardware: {result["runtime"].get("gpu","unknown")}. '
                 'The historical comparison is contextual; hardware/source differences must be disclosed.\n')
    (output/'ABLATION-COMPARISON.md').write_text(text)
    print(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unweighted',type=Path,required=True)
    parser.add_argument('--balanced',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    compare_runs(args.unweighted,args.balanced,args.output_dir)


if __name__=='__main__':
    main()
