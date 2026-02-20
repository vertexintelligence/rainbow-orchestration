# BRANCH PROTECTION DOCTRINE v1.0 (LOCKED)

## What is protected
- Repository: vertexintelligence/rainbow-orchestration
- Branch: main
- Required status check: lint
- Require branch to be up to date before merge (strict: true)
- Require at least 1 approving review
- Enforce branch protection for admins (enforce_admins: true)

## Why JSON input is required
Using gh api -f form fields can coerce booleans into strings.
The canonical safe approach is to send a strict JSON body with --input - so booleans remain true booleans and nested structures apply exactly.

## Canonical apply command
./bin/branch_protect_main

## Verification command
gh api repos/vertexintelligence/rainbow-orchestration/branches/main/protection
