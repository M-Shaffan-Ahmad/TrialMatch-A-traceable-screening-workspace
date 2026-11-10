"""Run: python -m streamlit run app/streamlit_app.py"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import streamlit as st
from trialmatch.screening import (NOTICE,LIMITATION,prepare_note,prepare_criteria,
                                  input_fingerprint,assessment,report_html)
from trialmatch.app_runtime import ModelRuntime,MODES

st.set_page_config(page_title='TrialMatch | Evidence-based screening',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp{background:#f6f8f7}h1,h2,h3{color:#163d3b}div[data-testid="stMetric"]{background:white;border:1px solid #dce8e4;border-radius:12px;padding:16px}
.block-container{max-width:1240px;padding-top:2.2rem}div.stButton>button[kind="primary"]{background:#16685b;border-color:#16685b}section[data-testid="stSidebar"]{background:#eaf1ee}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_runtime(adapter_directory):
    return ModelRuntime(adapter_directory)


def runtime():
    directory=os.environ.get('TRIALMATCH_ADAPTER_DIR',str(ROOT/'checkpoints/candidates/prompt-v2-unweighted-epoch-03'))
    if not (Path(directory)/'adapter_model.safetensors').exists():
        raise ValueError('The selected model adapter is not installed. Follow the app setup instructions and relaunch.')
    return get_runtime(directory)


def load_synthetic():
    st.session_state['patient']='A synthetic patient is 50 years old.\nDiabetes is explicitly denied.\nFever has been present for two days.'
    st.session_state['inclusion']='Age at least 18 years.\nFever lasting more than one week.'
    st.session_state['exclusion']='Diabetes mellitus.'
    for key in ['assessment','comparison']:
        st.session_state.pop(key,None)

with st.sidebar:
    st.markdown('### TrialMatch')
    st.caption('A traceable screening workspace')
    mode=st.selectbox('Screening mode',MODES,key='mode')
    output_limit=st.select_slider('Response token budget',options=[64,128,192],value=192,key='output_limit')
    st.button('Load synthetic example',on_click=load_synthetic,width="stretch")
    st.caption('The example is invented and is not a benchmark patient.')
    st.divider()
    st.markdown('**Selected model**\n\nQwen2.5-1.5B-Instruct + unweighted LoRA, epoch 3, prompt v2.')
    st.caption('Draft: Qwen2.5-0.5B-Instruct. Both use pinned model revisions.')
    st.warning('Rare-category limitation: the validation model missed all “excluded” and “not included” cases. Human review is essential.')
    if st.button('Clear this assessment',width="stretch"):
        for key in ['assessment','comparison','patient','inclusion','exclusion']:
            st.session_state.pop(key,None)
        st.rerun()

st.caption('CLINICAL TRIAL RESEARCH · EDUCATIONAL PROTOTYPE')
st.title('Every criterion. Its evidence.')
st.write('Screen a patient record against supplied trial criteria, then inspect exactly what the model cited.')
st.info(NOTICE)
left,right=st.columns([1.05,1],gap='large')
with left:
    st.subheader('1 · Patient record')
    st.caption('Enter one sentence per line. Existing numbered sentence IDs are preserved.')
    patient=st.text_area('Patient source text',height=260,key='patient',placeholder='A patient is 50 years old.\nDiabetes is explicitly denied.')
with right:
    st.subheader('2 · Trial criteria')
    st.caption('Enter one individual criterion per line. Assess up to 20 criteria.')
    inclusion=st.text_area('Inclusion criteria',height=115,key='inclusion',placeholder='Age at least 18 years.')
    exclusion=st.text_area('Exclusion criteria',height=115,key='exclusion',placeholder='Diabetes mellitus.')

prepared=None
try:
    note,sentences,origin=prepare_note(patient)
    criteria=prepare_criteria(inclusion,exclusion)
    fingerprint=input_fingerprint(note,criteria)
    prepared=(note,sentences,origin,criteria,fingerprint)
except ValueError as exc:
    validation_error=str(exc)

for key in ['assessment','comparison']:
    saved=st.session_state.get(key)
    if saved is not None:
        old=saved.get('input_sha256')
        if prepared is None or old!=prepared[-1]:
            st.session_state.pop(key,None)
            st.caption('Inputs changed; previous results were cleared. Run the updated assessment.')

if prepared:
    with st.expander(f'Preview source sentence IDs · {len(sentences)} sentences / {len(criteria)} criteria'):
        st.text(note)
        st.caption(origin)

if st.button('Screen supplied criteria',type='primary',key='screen'):
    if prepared is None:
        st.error(validation_error)
    else:
        st.session_state.pop('assessment',None)
        progress=st.progress(0,text='Loading selected model…')
        try:
            with st.spinner('Loading model and screening criteria…'):
                engine=runtime()
                rows=engine.screen(note,criteria,mode,output_limit,
                    on_result=lambda done,total:progress.progress(done/total,text=f'Assessed {done} of {total} criteria'))
                value=assessment(note,criteria,rows,dict(engine.provenance),mode,origin)
                st.session_state['assessment']=value
        except Exception as exc:
            st.error('Assessment could not finish. Your inputs are retained; review the message and retry.')
            with st.expander('Failure details'):
                st.text(str(exc))
        finally:
            progress.empty()

value=st.session_state.get('assessment')
if value:
    st.divider()
    st.subheader(value['aggregate']['outcome'])
    st.write(value['aggregate']['reason'])
    st.caption('This is an explicit rule-based summary of criterion predictions, not a separately validated patient–trial verdict.')
    cols=st.columns(3)
    cols[0].metric('Criteria assessed',len(value['results']))
    cols[1].metric('Need information / invalid',sum(not r['valid'] or r['label']=='not enough information' for r in value['results']))
    cols[2].metric('Without cited evidence',sum(not r['evidence'] for r in value['results']))
    if value['mode']!=mode:
        st.caption(f'Displayed assessment used {value["mode"]}. Run again to use the newly selected mode.')
    for row in value['results']:
        label=row['label'] if row['valid'] else 'Invalid output / needs review'
        with st.expander(f'{row["criterion_id"]} · {row["criterion_type"].title()} · {label}',expanded=True):
            st.write(row['criterion_text'])
            st.write(row['explanation'])
            if row['error']:
                st.error(row['error'])
            for evidence in row['evidence']:
                st.text(f'[{evidence["sentence_id"]}] {evidence["text"]}')
            if not row['evidence']:
                st.caption('No supporting sentence was cited. Do not infer absence of a condition from this alone.')
            with st.expander('Model output and timing'):
                st.code(row['raw_output'],language='json')
                st.json(row['decoding'])
    download_cols=st.columns(2)
    download_cols[0].download_button('Download readable report',report_html(value),
        file_name='trialmatch-screening-report.html',mime='text/html',key='html_export')
    download_cols[1].download_button('Download assessment data',json.dumps(value,indent=2),
        file_name='trialmatch-assessment.json',mime='application/json',key='json_export')
    st.caption('The HTML report opens in a browser and can be printed to PDF. Exports include the patient record you entered.')

st.divider()
st.subheader('Compare inference on your input')
st.write('Run one supplied criterion through uncached, cached and speculative greedy decoding. Compare actual outputs and repeated timings.')
if prepared:
    chosen=st.selectbox('Criterion to compare',list(range(len(criteria))),
        format_func=lambda i:f'{i+1} · {criteria[i]["criterion_type"]} · {criteria[i]["criterion_text"][:90]}')
else:
    chosen=0
    st.caption('Enter a patient record and at least one criterion to compare modes.')
bench_cols=st.columns(2)
bench_limit=bench_cols[0].select_slider('Comparison output token budget',options=[32,64,128,192],value=64)
repeats=bench_cols[1].selectbox('Timed repetitions per mode',[3,5])
if st.button('Run live inference comparison',key='compare'):
    st.session_state.pop('comparison',None)
    if prepared is None:
        st.error(validation_error)
    else:
        try:
            with st.spinner('Warming up all modes and measuring the supplied criterion…'):
                engine=runtime()
                compared=engine.compare(note,criteria[chosen],bench_limit,repeats)
                compared['input_sha256']=fingerprint
                st.session_state['comparison']=compared
        except Exception as exc:
            st.error('Comparison could not finish. Inputs are retained; retry after addressing the failure.')
            with st.expander('Comparison failure details'):
                st.text(str(exc))
compared=st.session_state.get('comparison')
if compared:
    if compared['all_tokens_equal']:
        st.success('All measured output token sequences match uncached greedy.')
    else:
        st.error('Output token mismatch detected. Treat speed comparisons as diagnostic; inspect exported records.')
    st.caption(f'Measured criterion: {compared["criterion"]["criterion_text"]} · budget {compared["max_new_tokens"]} tokens · {compared["repeats"]} repetitions.')
    st.dataframe(compared['summaries'],width="stretch",hide_index=True)
    st.bar_chart(compared['summaries'],x='mode',y='median_seconds',horizontal=True)
    st.caption('Seconds, lower is faster. One warmup per mode; execution order rotates. Both models remain resident. EOS may stop before the token budget. Model loading is excluded; prefill and Python overhead are included. No speedup is assumed.')
    with st.expander('Generated output and run provenance'):
        st.code(compared['decoded_text'],language='json')
        st.json(compared['provenance'])
    st.download_button('Download comparison measurements',json.dumps(compared,indent=2),
        file_name='trialmatch-live-comparison.json',mime='application/json')

with st.expander('How the preliminary summary works'):
    st.write('An explicit model “not included” inclusion result or “excluded” exclusion result flags a potential conflict. Invalid, insufficient-information or uncited decisions require review. If every criterion is not applicable, the case remains unassessed. Otherwise the app reports no identified conflict in the supplied criteria. These rules do not verify enrollment eligibility.')
    st.write(LIMITATION)
    st.caption('Patient inputs and results are held in this browser session’s server state and are not written by this app to a database. Only model weights are shared across sessions; generation is serialized. Downloaded reports contain your entered information.')
