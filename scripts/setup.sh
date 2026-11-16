#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
python_command="${PYTHON:-python3}"
"$python_command" -c 'import sys; assert sys.version_info >= (3, 14), "Use Python 3.14 or later; the released artifacts were tested with 3.14."'
"$python_command" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
printf '\nDependencies installed. Install the companion runtime bundle, then run scripts/doctor.py.\n'
