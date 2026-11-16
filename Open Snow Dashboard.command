#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ -x ".venv/bin/python" ]]; then
    exec .venv/bin/python src/serve_snow_dashboard.py --open
fi
exec python3 src/serve_snow_dashboard.py --open
