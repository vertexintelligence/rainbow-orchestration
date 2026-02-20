.PHONY: all lint-all lint-broker lint-policy lint-firewall lint-sandbox up down status logs health boot doctor protect-main

all: lint-all

lint-all: lint-broker lint-policy lint-firewall lint-sandbox

lint-broker:
	python3 -m py_compile genx_kernel/broker/app/main.py

lint-policy:
	python3 -m py_compile genx_kernel/policy/app/main.py

lint-firewall:
	python3 -m py_compile genx_kernel/firewall/app/main.py

lint-sandbox:
	python3 -m py_compile genx_kernel/sandbox/app/main.py

up:
	docker compose up -d --force-recreate

down:
	docker compose down

status:
	docker compose ps

logs:
	docker compose logs --tail 120

health:
	@curl -fsS http://localhost:8787/docs >/dev/null && echo "broker docs ok" || echo "broker docs FAIL"
	@curl -fsS http://localhost:8786/docs >/dev/null && echo "firewall docs ok" || echo "firewall docs FAIL"
	@curl -fsS http://localhost:8788/docs >/dev/null && echo "policy docs ok" || echo "policy docs FAIL"
	@curl -fsS http://localhost:8789/docs >/dev/null && echo "sandbox docs ok" || echo "sandbox docs FAIL"

boot: lint-all up status logs health

doctor:
	@echo "=== LINT ==="; $(MAKE) lint-all
	@echo "=== STATUS ==="; $(MAKE) status
	@echo "=== LOGS (broker) ==="; docker compose logs --tail 80 broker || true
	@echo "=== HEALTH ==="; $(MAKE) health

protect-main:
	./bin/branch_protect_main
