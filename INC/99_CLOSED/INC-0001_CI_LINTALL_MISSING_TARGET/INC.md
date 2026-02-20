# INC-0001 — CI Failure: missing make target lint-all

## 00_INTAKE
- Symptom: GitHub Actions run failed on main push
- Error: `make: *** No rule to make target 'lint-all'. Stop.`
- Run ID: 22240589551
- Commit checked out by runner: d98e54b...

## 01_EVIDENCE
- GH run log shows workflow executed `make lint-all` but Makefile lacked that target at d98e54b...

## 02_ROOTCAUSE
- Workflow referenced `lint-all` before Makefile defined it (ordering gap).

## 03_PATCH
- Added Makefile targets: lint-all + subtargets (broker/policy/firewall/sandbox).
- Added tab-safe recipes.

## 04_PROOF
- Later runs succeeded (✓) in `gh run list` after Makefile update.
- Local `make lint-all` succeeds.

## 05_PREVENTION
- Rule: introduce CI workflow only after Makefile target exists (or merge both in same commit/PR).
- Branch protection requires CI passing before merge to main.
