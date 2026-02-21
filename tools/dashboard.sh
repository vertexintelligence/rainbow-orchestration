#!/usr/bin/env bash

echo "==============================="
echo " RAINBOW ORCHESTRATION DASHBOARD"
echo "==============================="
echo

echo "IDENTITY:"
gh api user --jq .login 2>/dev/null || echo "Not authenticated"
echo

echo "GIT STATE:"
git branch --show-current
git status -sb
echo

echo "LAST 5 COMMITS:"
git log --oneline -n 5
echo

echo "BRANCH PROTECTION REQUIRED CHECK:"
gh api repos/vertexintelligence/rainbow-orchestration/branches/main/protection \
  --jq '.required_status_checks.contexts'
echo

echo "OPEN PRS:"
gh pr list
echo

echo "DOCKER SERVICES:"
docker compose ps
echo

echo "==============================="
