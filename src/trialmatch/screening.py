"""Patient-input validation, conservative aggregation and escaped readable export."""
from datetime import datetime, timezone
import hashlib
import html
import json
import re

from trialmatch.data import sentence_map
from trialmatch.evaluation import parse_prediction

NOTICE='Educational screening prototype, not a clinical enrollment decision. All results require human review.'
LIMITATION='Validation missed both rare categories excluded and not included; predictions may overlook a disqualifying condition.'


def prepare_note(text):
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Enter a patient record. Put one sentence per line, or use original numbered sentences.')
    if len(text)>40000:
        raise ValueError('Patient record is too long (maximum 40,000 characters).')
    text=text.strip()
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    numbered=[bool(re.match(r'^\d+\.\s+',line)) for line in lines]
    if any(numbered) and not all(numbered):
        raise ValueError('Use either all numbered sentences or one unnumbered sentence per line; do not mix formats.')
    if all(numbered):
        note='\n'.join(lines)
        origin='Original supplied sentence IDs preserved.'
    else:
        note='\n'.join(f'{i}. {line}' for i,line in enumerate(lines))
        origin='Assigned zero-based IDs to nonempty input lines; sentence boundaries are user supplied.'
    sentences=sentence_map(note)
    if len(sentences)>200:
        raise ValueError('Use at most 200 patient sentences.')
    return note,sentences,origin


def prepare_criteria(inclusion,exclusion):
    rows=[]
    for kind,text in [('inclusion',inclusion),('exclusion',exclusion)]:
        if not isinstance(text,str):
            raise ValueError('Criteria must be text, one criterion per line.')
        for line in text.splitlines():
            criterion=line.strip()
            if criterion:
                if len(criterion)>2000:
                    raise ValueError('Each criterion must be at most 2,000 characters.')
                rows.append({'criterion_id':len(rows)+1,'criterion_type':kind,'criterion_text':criterion})
    if not rows:
        raise ValueError('Enter at least one inclusion or exclusion criterion.')
    if len(rows)>20:
        raise ValueError('Screen at most 20 criteria per assessment.')
    return rows


def input_fingerprint(note,criteria):
    return hashlib.sha256(json.dumps({'note':note,'criteria':criteria},sort_keys=True).encode()).hexdigest()


def result_from_output(criterion,note,text,decoding,error=None):
    parsed=parse_prediction(text,criterion['criterion_type'],note)
    if error:
        parsed.update(valid=False,label='__invalid__',evidence_sentence_ids=[],error=error)
    sentences=sentence_map(note)
    evidence=[{'sentence_id':sid,'text':sentences[sid]} for sid in parsed['evidence_sentence_ids']]
    if not parsed['valid']:
        explanation='No validated decision is available. Review the input and generated output, then retry.'
    elif parsed['label']=='not enough information':
        explanation='The model reports insufficient information to resolve this criterion. Review the criterion and cited text; no missing clinical facts have been inferred.'
    elif not evidence:
        explanation='The model supplied no evidence sentence IDs. This does not prove eligibility or ineligibility.'
    else:
        explanation='The model cites the source sentences shown below for this category. This is an evidence summary, not an independently verified medical rationale.'
    return {**criterion,**parsed,'evidence':evidence,'explanation':explanation,
            'raw_output':text,'decoding':decoding}


def aggregate(results):
    if not results:
        return {'outcome':'Unassessed','reason':'No criteria have been assessed.'}
    blockers=[r['criterion_id'] for r in results if r['valid'] and
              ((r['criterion_type']=='inclusion' and r['label']=='not included') or
               (r['criterion_type']=='exclusion' and r['label']=='excluded'))]
    if blockers:
        return {'outcome':'Potential criterion conflict','reason':f'Model flagged disqualifying criteria: {blockers}. Human verification required.'}
    unresolved=[r['criterion_id'] for r in results if not r['valid'] or r['label']=='not enough information' or not r['evidence']]
    # All-not-applicable is unassessed even if the model cites context for it.
    if all(r['valid'] and r['label']=='not applicable' for r in results):
        return {'outcome':'Unassessed','reason':'All criteria were labeled not applicable.'}
    if unresolved:
        return {'outcome':'Needs review / more information','reason':f'Invalid, uncertain or uncited decisions at criteria: {unresolved}.'}
    return {'outcome':'No conflict identified in supplied criteria',
            'reason':'A rule-based summary of model outputs only; it does not confirm enrollment eligibility.'}


def assessment(note,criteria,results,provenance,mode,sentence_origin):
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'notice':NOTICE,'known_limitation':LIMITATION,
            'patient_note':note,'sentence_origin':sentence_origin,'criteria':criteria,'results':results,
            'aggregate':aggregate(results),'model':provenance,'mode':mode,
            'input_sha256':input_fingerprint(note,criteria),
            'aggregation_rule':'Conflict on explicit model not-included/excluded; unresolved on invalid/unknown/uncited; all-not-applicable unassessed; otherwise no identified conflict. Not ground-truth trial eligibility.'}


def report_html(value):
    escape=lambda x:html.escape(str(x),quote=True)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>TrialMatch screening report</title>',
           '<style>body{font:16px/1.6 system-ui;max-width:920px;margin:40px auto;padding:0 24px;color:#172c32}article{border-top:1px solid #ccd8d8;padding:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}aside{background:#edf5f4;padding:16px}small{color:#42565c}@media print{body{margin:0}}</style>',
           '<h1>TrialMatch screening report</h1>',f'<aside>{escape(value["notice"])}<br>{escape(value["known_limitation"])}</aside>',
           f'<p>Created: {escape(value["created_utc"])} · Mode: {escape(value["mode"])}</p>',
           f'<h2>{escape(value["aggregate"]["outcome"])}</h2><p>{escape(value["aggregate"]["reason"])}</p>',
           '<h2>Patient source sentences</h2>',f'<pre>{escape(value["patient_note"])}</pre>']
    for row in value['results']:
        parts.extend(['<article>',f'<h2>Criterion {row["criterion_id"]} · {escape(row["criterion_type"])}</h2>',
                      f'<p>{escape(row["criterion_text"])}</p>',f'<strong>{escape(row["label"] if row["valid"] else "Invalid prediction / review required")}</strong>',
                      f'<p>{escape(row["explanation"])}</p>'])
        for cited in row['evidence']:
            parts.append(f'<blockquote>[{cited["sentence_id"]}] {escape(cited["text"])}</blockquote>')
        if row['error']:
            parts.append(f'<p>Validation error: {escape(row["error"])}</p>')
        parts.extend([f'<details><summary>Original model response</summary><pre>{escape(row["raw_output"])}</pre></details>','</article>'])
    parts.extend(['<h2>Provenance and aggregation</h2>',f'<p>{escape(value["aggregation_rule"])}</p>',
                  f'<p>{escape(value["sentence_origin"])}</p>',f'<pre>{escape(json.dumps(value["model"],indent=2))}</pre>',
                  f'<small>Input SHA256: {escape(value["input_sha256"])}</small></html>'])
    return '\n'.join(parts)
