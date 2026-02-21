#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: tools/two_key_merge.sh <pr-number>"
  exit 2
fi
PR="$1"

echo "Current identity:"
gh api user --jq .login

echo "PR status:"
gh pr view "$PR" --json mergeStateStatus,reviewDecision --jq '{state:.mergeStateStatus, review:.reviewDecision}'

echo
echo "NOTE: Approvals are dismissed on force-push."
echo "Use vertex-control-node to approve, then vertexintelligence to merge."
