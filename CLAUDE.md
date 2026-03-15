## NotarAI

Specs live in `.notarai/*.spec.yaml` and are the canonical source of truth.
Run `/notarai-reconcile` to detect drift between specs, code, and docs.
Run `notarai validate .notarai/` to validate specs manually.
The PostToolUse hook auto-validates any spec file you write or edit.
