#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_OVERRIDE="${ADVICE_AUDIT_PYTHON:-}"
if [[ -n "$PYTHON_OVERRIDE" ]]; then
    PYTHON_BIN="$PYTHON_OVERRIDE"
else
    PYTHON_BIN=""
    for candidate in python3.11 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)]' 2>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    printf '%s\n' 'Python 3.11 or 3.12 is required. Install it plus the matching venv package, then retry.' >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)], "Use Python 3.11 or 3.12"'
if [[ -d .venv && ! -x .venv/bin/python ]]; then
    printf '%s\n' '.venv exists but is not a usable Linux venv. Move it aside manually; this script does not delete it.' >&2
    exit 1
fi
if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)], "Existing venv must use Python 3.11/3.12"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[sim,dev,plot]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
printf '%s\n' 'Ready. Read docs/SERVER_AGENT_GUIDE.md, then run scripts/server_run.py with a NEW --run-dir.'
