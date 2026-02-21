#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== LINT ==="
make lint-all
echo "=== TESTS ==="
make test
echo "=== VALIDATE EXAMPLES ==="
make validate-examples
echo "OK: doctor complete"
