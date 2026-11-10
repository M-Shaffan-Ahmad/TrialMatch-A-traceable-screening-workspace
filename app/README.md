# TrialMatch browser application

A standalone Streamlit application using the selected prompt-v2, unweighted epoch-3
adapter and real custom cached/uncached/speculative decoders. No fake/demo prediction
backend is available in production. The built-in example supplies synthetic inputs only.
Educational screening prototype, not a clinical enrollment decision.

## Colab GPU launch

Open `notebooks/05_colab_browser_app.ipynb`, select a GPU, and run its startup cells in
order. Upload the newest `trialmatch-colab-starter.zip` and the separate selected-adapter
ZIP. Installation streams pip errors and saves an installation log. After startup, use the
browser UI for input, screening, evidence, downloads and repeated live comparisons.
No training, notebook editing or patient benchmark download is needed to use these flows.

The notebook displays the UI via Colab's supported iframe port proxy. This authenticated
proxy is session-dependent and must be verified in the user's actual Colab browser.
Colab's old new-window helper is deprecated; a public tunnel is not automatically created.
If cookies/WebSockets prevent the iframe connecting, preserve the server log and use the
standalone local route below. Colab startup and full selected-model app inference are not
claimed verified merely because the local server and tests pass.

The first screen/comparison loads pinned model weights and can take time. The draft is
loaded only when speculation/comparison is requested. No base weights are in the source
or adapter ZIP; Hugging Face downloads supply the pinned public weights. Keep runtime
active. Download reports to retain them; they are not automatically saved to Drive.

## Standalone local launch (no notebook or IDE required)

One-time operator setup at the project root, using Python 3.10+:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-app.txt -e .
```

Install a compatible PyTorch build for the machine if one is not already present. CUDA
GPU is recommended; CPU falls back to FP32 and can be slow and memory-intensive.
The local development environment has CPU-only PyTorch; do not copy that environment
over a Colab CUDA installation.

The selected adapter is already present in this workspace under
`checkpoints/candidates/prompt-v2-unweighted-epoch-03`. On a new machine unpack
`trialmatch-selected-adapter.zip` into that folder, or point the environment variable
`TRIALMATCH_ADAPTER_DIR` to its extracted directory. The app verifies every candidate
file hash before loading. The earlier smoke/weighted adapters will be rejected.

Start the application:

```bash
./start-app.sh
```

Open http://127.0.0.1:8501 in a browser. The operator starts it once; the evaluator uses
all screening and comparison features through the browser, without executing model code.
The launcher binds loopback only. Stop with Ctrl+C in its launch terminal. For another
port use `.venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8502`.
This is a research/demo server, not a production deployment or clinical system.

## Browser flow

1. Paste a patient record: one sentence per line, or all lines already numbered. The
   displayed source preview makes exact sentence IDs inspectable. Mixed/duplicate IDs
   are rejected; supplied numeric IDs are preserved. Unnumbered lines get IDs from zero.
2. Enter one inclusion/exclusion criterion per line (up to 20 total). No benchmark
   criteria or patient examples are required; arbitrary unseen inputs are supported.
3. Choose cached, uncached or speculative greedy and screen. Results show each type,
   criterion text, original category, source sentence IDs/text and uncertainty/failure.
4. Download an HTML report (browser-readable and printable to PDF) or complete JSON.
   Exports include input snapshot, model/version/hash, timing, raw output and aggregate rule.
5. Choose a supplied criterion and run live comparison: 3 or 5 timed repeats per mode
   after one warmup, rotating mode order. Both models are resident. Compare median latency,
   prefill/decode duration, throughput, allocator peak, draft acceptance and token equality.
6. Download raw comparison records. Input changes clear stale screening/comparison results.
   Changing a mode leaves the saved assessment labeled with its actual original mode.

Input character limits and a 2048-token per-criterion prompt cap prevent silent truncation.
Every criterion is preflighted before any decision is generated. The model's output is
strictly parsed by the existing evaluator; invalid JSON/categories/citations are flagged
without repair or guessed decisions. Session inputs survive a recoverable model failure.

## Meaning and limitations

The displayed explanation is an exact-source evidence summary and uncertainty notice,
not an independently verified clinical rationale. The app never invents missing facts.
It flags uncertainty for model insufficient-information outcomes, invalid responses and
uncited decisions; users inspect the supplied criterion to identify what information is
needed. It does not claim to infer a complete list of missing medical tests or diagnoses.

Rule-based aggregate: any explicit model not-included/excluded -> potential conflict;
invalid/unknown/uncited -> needs review; all-not-applicable -> unassessed; otherwise no
identified conflict in supplied criteria. This is more conservative about uncited outcomes
than the initial proposal. It is not a separately trained or ground-truth patient-trial verdict.
The model has zero rare-category recall on the development validation set, shown clearly
in the interface and report. No-conflict output must not be treated as enrollment clearance.

Results and patient inputs live only in per-session server state. Model weights are a shared
cached resource; an RLock serializes screening/comparison and each decoder creates fresh
request-local caches. Results are never cached globally. No app database or external
patient API is used. Exported HTML escapes all patient/criterion/model text to prevent
script injection. Clearing assessment state does not clear model weights or claim secure
erasure of Python memory. Do not treat this prototype as a clinical data-storage service.

## Verification status

- Local Streamlit server launched; actual browser displayed the form and synthetic-input flow.
- Screenshot: `results/app/browser-preview.png` (input view, no claimed model prediction).
- Streamlit AppTest verifies missing-input recovery, screening/display/download controls,
  comparison/chart rendering and stale-input clearing with a test-only runtime fixture.
- Real tiny random Qwen models exercise the production runtime's three decoders, strict
  failure handling and repeated nine-measurement comparison. No fabricated model-quality
  results are inferred from these random weights.
- Unit tests cover sentence IDs, limits, conservative aggregation and safe HTML exports.
- Full selected-model browser screening, actual report downloading via browser and Colab
  proxy/GPU integration still need user execution. Earlier selected-checkpoint inference
  benchmark results establish decoder feasibility but not this app's integration by themselves.

Primary implementation references:
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py
