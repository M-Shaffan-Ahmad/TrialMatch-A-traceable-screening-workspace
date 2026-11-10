"""Package an explicit allowlist; exclude data examples, environments, weights and secrets."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'pyproject.toml', 'requirements-colab.txt',
         'requirements-data.lock.txt', 'requirements-app.txt', 'start-app.sh', '.gitignore', 'data/manifests/split_manifest.json']
FOLDERS = ['src', 'scripts', 'configs', 'docs', 'tests', 'notebooks', 'report', 'app']


def main():
    files = {ROOT / name for name in FILES}
    for folder in FOLDERS:
        files.update(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in {'.pyc', '.pyo'}
                     and '.ipynb_checkpoints' not in p.parts)
    target = ROOT / 'output/trialmatch-colab-starter.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'Packaged {len(files)} source/config files: {target}')


if __name__ == '__main__':
    main()
