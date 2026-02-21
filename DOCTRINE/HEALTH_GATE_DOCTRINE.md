# HEALTH GATE DOCTRINE v1.0 (LOCKED)

## Problem
Immediately after `docker compose up -d --force-recreate`, `/docs` endpoints may return:
- curl: (56) Recv failure: Connection reset by peer

This is a startup race (services are Up but app not ready yet).

## Canonical fix
Always run:
- `make wait-health` before `make health`
- `make doctor` includes the wait step

## Commands
make wait-health
make health
make doctor
