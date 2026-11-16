"""Install a locally supplied runtime ZIP after verifying the source manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def safe_target(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in name
            or not relative.parts or relative.parts[0] not in {'data', 'outputs'}
            or str(relative) != name):
        raise ValueError(f'Unexpected runtime path: {name}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Runtime path leaves project: {name}')
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root.parent):
        raise ValueError(f'Symbolic links are not allowed in runtime targets: {name}')
    return target


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def install(archive, root=ROOT, replace=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'artifacts/runtime-files.json').read_text())
    entries = manifest['files']
    expected = {entry['path']: entry for entry in entries}
    if len(expected) != len(entries):
        raise ValueError('Duplicate paths in runtime manifest')
    targets = {name: safe_target(root, name) for name in expected}
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) != len(expected) or {m.filename for m in members} != set(expected):
            raise ValueError('Archive contents differ from the runtime manifest')
        for member in members:
            entry = expected[member.filename]
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(mode) or member.file_size != entry['bytes']:
                raise ValueError(f'Invalid file type or size: {member.filename}')
            target = targets[member.filename]
            if target.exists() and not replace and digest(target) != entry['sha256']:
                raise ValueError(f'Existing file differs: {member.filename}; use --replace only to restore released artifacts')
        # Validate every byte before changing any installed artifact.
        with tempfile.TemporaryDirectory(prefix='.snowline-install-', dir=root) as scratch:
            staged = {}
            for member in members:
                path = Path(scratch)/member.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target, 1024*1024)
                if digest(path) != expected[member.filename]['sha256']:
                    raise ValueError(f'Checksum mismatch: {member.filename}')
                staged[member.filename] = path
            for name, path in staged.items():
                target = safe_target(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Companion snowline-runtime ZIP downloaded from the same release')
    parser.add_argument('--replace', action='store_true', help='Restore differing installed files from the verified bundle')
    args = parser.parse_args()
    try:
        count = install(args.archive, replace=args.replace)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Runtime installation stopped: {exc}\n')
    print(f'Installed {count} verified runtime files. Run .venv/bin/python scripts/doctor.py next.')


if __name__ == '__main__':
    main()
