"""Prepare the assignment split. Never print examples or held-out label statistics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from trialmatch.data import ALLOWED_COLUMNS, normalize

REPOSITORY = "ncbi/TrialGPT-Criterion-Annotations"


def split_patients(patient_ids):
    patients = np.array(sorted(set(patient_ids)))
    if len(patients) != 53:
        raise ValueError(f"Expected 53 unique patients; got {len(patients)}. Inspect source drift.")
    np.random.default_rng(42).shuffle(patients)
    return {"train": patients[:37].tolist(), "validation": patients[37:45].tolist(),
            "test": patients[45:].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Dataset commit; resolved SHA is recorded")
    args = parser.parse_args()
    raw_dir = ROOT / "data/raw"
    processed_dir = ROOT / "data/processed"
    cache = ROOT / ".cache/huggingface"
    for folder in (raw_dir, processed_dir, cache, ROOT / "data/manifests"):
        folder.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPOSITORY, revision=args.revision)
    paths = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    if not paths:
        raise ValueError("Dataset no longer exposes Parquet files; inspect upstream format")
    rows, sources = [], []
    for filename in paths:
        path = Path(hf_hub_download(repo_id=REPOSITORY, filename=filename,
                    repo_type="dataset", revision=info.sha,
                    local_dir=raw_dir, cache_dir=cache))
        # Column projection excludes previous GPT outputs before conversion to Python rows.
        rows.extend(pq.read_table(path, columns=list(ALLOWED_COLUMNS)).to_pylist())
        sources.append({"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    ids = [row["annotation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation IDs; refusing ambiguous benchmark")
    groups = split_patients(row["patient_id"] for row in rows)
    if sum(len(set(v)) for v in groups.values()) != len(set().union(*map(set, groups.values()))):
        raise ValueError("Patient leakage across splits")
    counts = Counter(row["patient_id"] for row in rows)
    manifest = {
        "dataset": REPOSITORY, "dataset_revision": info.sha, "source_files": sources,
        "seed": 42, "algorithm": "sort IDs; numpy.default_rng(42).shuffle; slices 37/8/8",
        "numpy_version": np.__version__, "total_rows": len(rows),
        "splits": {name: {"patient_ids": patients, "patient_count": len(patients),
                          "annotation_count": sum(counts[p] for p in patients),
                          "annotations_per_patient": {p: counts[p] for p in patients}}
                   for name, patients in groups.items()},
    }
    manifest_path = ROOT / "data/manifests/split_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        # A different NumPy version is acceptable only if all actual split IDs,
        # counts, source hashes and the source revision are identical. Preserve
        # the original provenance rather than rewriting it on each environment.
        comparable = lambda value: {k: v for k, v in value.items() if k != "numpy_version"}
        if comparable(previous) != comparable(manifest):
            raise ValueError("Existing manifest differs. Preserve it and investigate source/version drift.")
        manifest = previous
    # Validate every record before writing any split; no test-specific diagnostics or scoring.
    normalized = [normalize(row) for row in rows]
    for name, patients in groups.items():
        members = set(patients)
        examples = sorted((r for r in normalized if r["patient_id"] in members),
                          key=lambda r: r["annotation_id"])
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples)
        (processed_dir / f"{name}.jsonl").write_text(text)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(rows)} annotations from dataset commit {info.sha}.")
    for name, entry in manifest["splits"].items():
        print(f"{name}: {entry['patient_count']} patients / {entry['annotation_count']} annotations")
    print("No predictions generated. Test examples were not displayed or evaluated.")


if __name__ == "__main__":
    main()
