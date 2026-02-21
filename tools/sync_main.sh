#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch origin
git switch main
git reset --hard origin/main
echo "OK: main aligned to origin/main"
