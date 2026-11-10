# TrialMatch - Generative AI Assignment 2

[IEEE report](TrialMatch-IEEE-Report.pdf) · [Demo video](https://youtu.be/7SLR7LBg0mo) · [Selected adapter ZIP](https://github.com/M-Shaffan-Ahmad/TrialMatch-A-traceable-screening-workspace/raw/f6d5ea50e408df3ccae7795f00dc44d8d764edfb/trialmatch-selected-adapter.zip) · [Browser setup](app/README.md) · [Submission details](SUBMISSION-LINKS.txt)

**Muhammad Shaffan Ahmad · 23i-0673 · Gen AI Section A**


Educational screening prototype; not a clinical enrollment decision.

## Current status

The controlled A/B runs completed on an A100 40 GB. Current validation candidate:
**prompt v2, unweighted, epoch 3** - accuracy 73.8%, macro-F1 44.8%, evidence micro-F1
63.0%, zero invalid outputs. The weighted variant gains one correct classification but
has lower macro-F1 and substantially worse evidence precision/F1. Both variants still miss
all rare `excluded` and `not included` validation cases. Test evaluation remains unused.

The selected A adapter is preserved at `checkpoints/candidates/prompt-v2-unweighted-epoch-03`;
`configs/model_candidate.json` records the pinned base, prompt version, weights hashes and
validation metrics. See [the verified ablation review](results/colab/ablation-20261107T065156632067Z/REVIEW.md).
This is a development candidate, not a final held-out result.

Custom cached/uncached greedy decoding, tiled online-softmax attention and greedy
speculative decoding are now implemented and verified locally with tiny random models.
Run [notebook 04](notebooks/04_colab_inference_benchmarks.ipynb) with the updated
`output/trialmatch-colab-starter.zip` and separate `output/trialmatch-selected-adapter.zip`
to measure the actual selected checkpoint on Colab. Read [the mechanism/benchmark
protocol](docs/inference-mechanisms.md). The uploaded full-model benchmark passed on NVIDIA L4: 45 repeated decoder
records and an EOS-aware synthetic screening case agree token-for-token. At 1024/128,
cache speedup was 2.69x; speculative speedup over cached was 1.16x on repetitive synthetic
text, but the single screening diagnostic was 27.9% slower. Tiled attention passed with
max error 5.96e-7 and lower allocator peaks, while being substantially slower than dense.
See [the verified inference review](results/colab/inference-20261110T135259207631Z/REVIEW.md).
The browser app is now implemented with real custom-mode screening, exact evidence,
HTML/JSON reports and repeated live comparison. The local UI and synthetic/tiny-model
tests passed; full selected-model browser use and Colab embedding still require a run.
Use [notebook 05](notebooks/05_colab_browser_app.ipynb) with the updated source/adapter ZIPs,
or [the standalone launch](app/README.md). Final held-out evaluation, paper and demo remain.

The original v1 run and both ablations were preserved. To reproduce the A/B pair use
[notebook 03](notebooks/03_colab_prompt_balance.ipynb) and [the experiment protocol](docs/prompt-balancing-experiments.md).
Earlier notebooks 01/02 remain available. Use the exact original source/config for resumes;
never mix a new implementation into an existing checkpoint history.

Rebuild the source ZIP after edits with `python scripts/package_colab.py`.
Read [the requirements checklist](docs/assignment-checklist.md),
[architecture](docs/architecture.md), and [data protocol](docs/data-protocol.md).

## Setup

Python 3.10+ (created locally with Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/check_environment.py
python scripts/prepare_data.py
python -m unittest discover -s tests -v
```

`prepare_data.py` downloads the official public dataset to the project-local cache,
projects an explicit column allowlist, and creates the required 37/8/8 split.
It pins the resolved dataset commit in `data/manifests/split_manifest.json`.
Recreate that dataset version with `python scripts/prepare_data.py --revision COMMIT`.
The prepared dataset contains 1,015 annotations: train 603, validation 206, test 206.
The manifest is intended to be published; raw data and prepared examples are ignored by Git.
Do not open the test examples or use them for development.

The selected pair is Qwen2.5-1.5B-Instruct (main target) and Qwen2.5-0.5B-Instruct
(speculative draft). Training and benchmarks will use Google Colab; the user reports
A100 access. Verify the assigned GPU and VRAM each session. Conservative settings
should also support the assignment's free-compute requirement.

Before training in Colab, install the proposed model dependencies:

```bash
python -m pip install -e '.[model]'
```

The Colab notebook instead uses `requirements-colab.txt`, preserving Colab's CUDA PyTorch,
and records the actual resolved environment. Model revisions are already pinned and
tokenizer compatibility passed. Run the notebook's one-batch GPU feasibility check
before full training.
Full-size training/evaluation execution, custom inference algorithms, benchmarks and the
browser app are the next work. See notebook 02 for the implemented training/evaluation workflow.

## Repository layout

- `configs/`: initial, editable experiment proposal.
- `src/trialmatch/`: data normalization and prompt boundary; future model/inference modules.
- `scripts/`: environment check and reproducible dataset preparation.
- `data/manifests/`: publishable split IDs, counts, and provenance.
- `data/processed/`: local train/validation/test files; keep test sealed until final evaluation.
- `tests/`: synthetic data-contract and split-integrity checks.
- `results/`: environment evidence, future metrics and benchmark logs.
- `app/`: future standalone browser interface.
- `report/`: IEEE LaTeX outline; no invented results.
- `checkpoints/`: local adapters; host large weights outside GitHub.

The `.git` directory is managed read-only in this workspace. Git initialization and GitHub
publication have not been performed. See `docs/next-steps.md`.
