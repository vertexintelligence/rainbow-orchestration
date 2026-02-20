# GATES DOCTRINE v1.0 (LOCKED)

## Compile Gates (Hard Requirement)
- Local: make lint-all must pass before push
- CI: GitHub Actions runs make lint-all on push + PR

## Gate Targets
- lint-broker, lint-policy, lint-firewall, lint-sandbox
- lint-all aggregates all compile checks

## Operational Gates
- make boot = lint-all + up + status + logs + health
- make doctor = fastest triage loop

## Purpose
Prevent syntax/indent regressions and enforce always-green deployability.
