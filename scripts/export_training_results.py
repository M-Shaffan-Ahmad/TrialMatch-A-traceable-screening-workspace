"""Export logs, plots and the selected adapter; leave resume optimizer states on Drive."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    selected_file = root/'selected_adapter.json'
    selected = None
    if selected_file.exists():
        selected = (root/json.loads(selected_file.read_text())['relative_path']).resolve()
        if not selected.is_relative_to(root):
            raise ValueError('Selected adapter is outside the run directory')
    paths = []
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part.startswith('.') for part in relative.parts) or path.suffix in {'.pt','.tmp'}:
            continue
        if 'adapter' in relative.parts and (selected is None or not path.is_relative_to(selected)):
            continue
        if path.suffix not in {'.json','.jsonl','.txt','.log','.md','.png','.svg','.safetensors','.jinja'}:
            continue
        paths.append(path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path,path.relative_to(root).as_posix())
    print(f'Exported {len(paths)} artifacts to {args.output}. Optimizer resume states remain in the run folder.')


if __name__=='__main__':
    main()
