"""Check local dependencies and frozen assets without downloading or forecasting."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems = []
    if sys.version_info < (3, 14):
        problems.append('Use the tested Python 3.14 environment.')
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if '==' not in line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = importlib.metadata.version(name)
            if actual != expected:
                problems.append(f'{name}: installed {actual}, expected {expected}')
        except importlib.metadata.PackageNotFoundError:
            problems.append(f'{name}: not installed')
    path = ROOT/'artifacts/runtime-files.json'
    if not path.exists():
        problems.append('Runtime manifest is missing from the checkout.')
    else:
        manifest = json.loads(path.read_text())
        for entry in manifest['files']:
            path = ROOT/entry['path']
            if not path.is_file():
                problems.append('Missing runtime file: '+entry['path'])
                continue
            with path.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != entry['sha256']:
                problems.append('Changed runtime file: '+entry['path'])
    if problems:
        print('\n'.join(problems))
        print('\nInstall requirements.txt and the matching runtime bundle before a full refresh.')
        return 1
    print('Dependencies and frozen runtime assets verified. Ready to start Snowline.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
