# Next work, in priority order

1. Use prompt-v2 unweighted epoch 3 as the current validation candidate. Both A/B
   runs are verified; class weighting did not improve macro-F1 or evidence F1. Preserve
   those negative results. The adapter and provenance are in checkpoints/candidates and
   configs/model_candidate.json. Test results remain unmeasured.
2. Inference GPU verification has passed on the selected checkpoint (NVIDIA L4).
   Preserve the repeated fixed-work gains and single-screening speculative slowdown;
   do not generalize 100% acceptance on repetitive text to clinical screening.
3. Run notebook 05 to verify full selected-model browser screening, exported reports and
   repeated live comparison on the actual GPU. Local UI, domain tests and tiny-model
   runtime checks passed; Colab proxy integration still needs verification. A standalone
   local launcher is also provided. See app/README.md.
4. Finish at least five primary-source readings and document technical choices critically.
5. Analyze at least eight distinct errors, including failed rare categories and unsupported
   citations. Do not claim the small validation A/B difference is statistically significant.
6. Freeze choices and run the final base/adapted comparison on the untouched test split.
7. Complete IEEE paper, GitHub publication, adapter download and demo video; rehearse defense.

GitHub publication is still required. This environment exposes a read-only managed `.git`
directory, so no Git repository or remote has been initialized here. Use a writable clone
when publishing, excluding `.venv`, raw/prepared data, caches, secrets and large model files.
Exact package versions for the data setup are recorded in `requirements-data.lock.txt`;
the optional training stack must be tested and locked on the chosen compute environment.
