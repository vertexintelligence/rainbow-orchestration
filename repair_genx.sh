#!/usr/bin/env bash
set -euo pipefail

cd ~/genx

echo "==> Writing firewall/main.py"
cat > genx_kernel/firewall/app/main.py <<'PY'
# (PASTE CLEAN FIREWALL CODE HERE - I will provide it again if you want)
PY

echo "==> Writing broker/main.py"
cat > genx_kernel/broker/app/main.py <<'PY'
# (PASTE CLEAN BROKER CODE HERE - I will provide it again if you want)
PY

echo "==> Rebuilding containers"
docker compose down
docker compose up --build -d

echo "==> Compile checks inside containers"
docker exec -it genx-firewall-1 python -m py_compile /app/app/main.py
docker exec -it genx-broker-1   python -m py_compile /app/app/main.py

echo "==> Done."
