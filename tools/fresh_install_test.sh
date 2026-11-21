#!/bin/bash
# Reproduce the graders' steps from a clean copy of the committed repo:
# new venv -> pip install -r requirements.txt -> prepare.py -> run.py.
# Usage: tools/fresh_install_test.sh <repo tarball> <work dir> <input folder>
set -euo pipefail
TARBALL=$1
WORK=$2
INPUT=$3
rm -rf "$WORK" && mkdir -p "$WORK/repo"
tar -xf "$TARBALL" -C "$WORK/repo"
cd "$WORK/repo"
python3.11 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
t=$(date +%s); pip install -q -r requirements.txt; echo "pip install: $(( $(date +%s) - t ))s"
pip check
t=$(date +%s); python prepare.py; echo "prepare: $(( $(date +%s) - t ))s"
t=$(date +%s.%N)
python run.py --input "$INPUT" --output "$WORK/results.jsonl"
python -c "import time; print('run: %.1fs' % (time.time() - $t))"
wc -l "$WORK/results.jsonl"
