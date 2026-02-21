# rainbow-orchestration

Contract-first orchestration for `DUNBAR_OS::DAILY_LOG_EVENT` payloads.

## Daily Log Contract

- Canon schema: `SCHEMAS/daily_log.schema.json`
- Draft: JSON Schema 2020-12
- Canon examples:
  - `EXAMPLES/daily_log_event.min.json`
  - `EXAMPLES/daily_log_event.full.json`

## Validate one event locally

Install dependencies first:

```bash
pip install -r requirements-dev.txt
```

Then run:

```bash
./bin/validate_daily_log <path-to-json>
```

Example:

```bash
./bin/validate_daily_log EXAMPLES/daily_log_event.min.json
```

Behavior:
- Exit code `0` on valid input.
- Non-zero on invalid input or runtime/load failure.
- Prints deterministic, human-readable error lines: `path`, violated `rule`, and message.

## Test and validate examples

```bash
make test
make validate-examples
```

## Publish Safety Guard

The contract enforces a fail-closed publish rule:
- When `dispatch.ready_to_publish == true`:
  - `dispatch.requires_human_confirm` must be `true`, and
  - at least one publish target in `dispatch.publish_targets` must be `true`.

## Adding new event schemas

1. Add schema under `SCHEMAS/` with strict JSON Schema draft 2020-12.
2. Add at least one minimal valid and one rich valid example under `EXAMPLES/`.
3. Add validation tests under `tests/` (valid + invalid cases).
4. Add/extend validator entrypoint in `bin/`.
5. Ensure CI runs tests and example validation.
