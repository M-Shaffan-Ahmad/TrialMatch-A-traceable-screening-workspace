"""Build a separate hash-verified adapter download for notebook 04; no base weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    candidate=json.loads((ROOT/'configs/model_candidate.json').read_text())
    directory=ROOT/candidate['adapter_directory']
    target=ROOT/'output/trialmatch-selected-adapter.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,expected in candidate['adapter_file_sha256'].items():
        if Path(name).name!=name:
            raise ValueError('Adapter manifest must contain filenames only')
        path=directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Adapter hash mismatch: {name}')
        paths.append(path)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path,path.name)
    print(f'Packaged selected adapter: {target} ({target.stat().st_size:,} bytes)')


if __name__=='__main__':
    main()
