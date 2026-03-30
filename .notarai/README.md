# NotarAI -- 0.5.1

This directory contains intent specs for this project. Specs are the canonical
source of truth -- when code and spec disagree, the spec wins.

## What's here

- `*.spec.yaml` -- intent specs (validated against `notarai.spec.json`)
- `notarai.spec.json` -- JSON Schema for spec files
- `.cache/` -- hash cache for reconciliation (gitignored)

## Workflow

1. Edit specs to capture intent
2. Write code and docs to match
3. Run `/notarai-reconcile` to detect drift
4. Approve proposed changes -- auto-sync never happens without review

## Validation

Specs are validated on every write via the PostToolUse hook.
Run `notarai validate .notarai/` to validate manually.

## Slash commands

- `/notarai-reconcile` -- detect drift between specs and code
- `/notarai-bootstrap` -- generate initial specs for a new codebase
