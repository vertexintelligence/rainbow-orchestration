#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== OPERATOR ONE-SHOT =="

# 1) Sync main to origin/main
echo "== SYNC MAIN =="
git fetch origin
git switch main
git reset --hard origin/main

# 2) Ensure venv exists
echo "== VENV =="
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# 3) Install dev deps (inside venv)
echo "== INSTALL DEV DEPS =="
python -m pip install -U pip
python -m pip install -r requirements-dev.txt

# 4) Run doctor + dashboard
echo "== DOCTOR =="
./tools/ci_doctor.sh

echo "== DASHBOARD =="
./tools/dashboard.sh

echo "OK: Operator one-shot complete."
