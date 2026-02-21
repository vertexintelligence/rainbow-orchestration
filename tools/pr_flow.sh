#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: tools/pr_flow.sh <branch> <commit-message>"
  exit 2
fi
BRANCH="$1"
MSG="$2"
cd "$(dirname "$0")/.."
git switch -c "$BRANCH"
git status
echo "Now make edits, then run: git add -A"
echo "Press Enter to commit+push+PR..."
read -r
git commit -m "$MSG"
git push -u origin "$BRANCH"
gh pr create --fill
gh pr checks --watch
