# AI-use log / report appendix seed

2026-11-06: OpenAI Codex assisted with reading the assignment, breaking down requirements,
checking official dataset/model documentation, proposing the initial architecture, and
creating the environment, data-preparation scripts, contract tests and report outline.
The student must review, understand, test and acknowledge this assistance. No training,
clinical validation or inference benchmark results were generated during this setup.

Append future AI assistance and reused libraries/code here with actual verification done.
Do not claim this is the complete final appendix until the project is finished.

2026-11-06 (Colab preparation): Codex created a Colab notebook, portable source ZIP,
pinned model-library requirements, tokenizer preflight, assistant-only supervision and
one-step LoRA feasibility scripts. Verification: 12 passing local unit tests including a
real tiny-random-Qwen optimizer step and adapter round trip; both selected models' actual
tokenizers matched, and immutable model commits were recorded. No selected full model
weights were downloaded or trained, and no Colab/A100 execution is claimed.

2026-11-07 (training workflow): Codex reviewed the uploaded successful A100 feasibility
logs, implemented strict structured-output parsing, fixed-label classification and evidence
metrics, baseline validation, an explicit LoRA training loop, token-weighted accumulation,
completed-epoch resume, validation checkpoint selection, actual adapter-reload verification,
curves, export and notebook 02. Tests use synthetic records and tiny random Qwen weights;
full-size pretrained-model results are not claimed before the Colab run.

2026-11-07 (prompt/balancing ablations): Codex diagnosed the uploaded first full run,
preserved original source/results, versioned a clearer prompt, added capped type-specific
training-only example loss weights with the same data/update budget, guarded shared-base
evaluation reuse, and prepared A/B notebook 03 and a comparison/export workflow. Tests
use synthetic data and tiny random models; no new full-model A/B results are claimed.

2026-11-10 (inference mechanisms): Codex implemented explicit cached/uncached forward
loops, greedy speculative draft/verify/crop/correction cache handling, standalone dense
and tiled online-softmax attention, synthetic correctness tests, and a selected-checkpoint
Colab benchmark with repeated synchronized timing and memory records. CPU tests do not
establish full pretrained-model GPU equivalence or speed; notebook 04 must measure that.

2026-11-10 (browser application): Codex implemented the standalone Streamlit UI, input
validation and source numbering, real custom-decoder runtime with serialized shared models,
conservative aggregation, exact evidence display, escaped HTML/JSON exports, repeated
live input comparison, local launcher and Colab launch notebook 05. Local browser and
synthetic/tiny-model tests were used; full selected-model app and Colab proxy use remain
to be verified by the user. Production contains no simulated inference mode.
