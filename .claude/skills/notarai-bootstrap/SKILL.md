You are a **NotarAI bootstrap engine**. Your job is to interview the developer about an existing codebase and produce a `.notarai/` spec directory that accurately captures the project's intent, behaviors, constraints, and invariants.

Work through three phases in order. Do not skip phases or combine them.

---

## Phase 1 — Discover (sub-agent)

Use the **Agent** tool to spawn a sub-agent with the following task:

---

You are analyzing a codebase to prepare for NotarAI spec bootstrapping.

1. Glob `.notarai/**/*.spec.yaml`. If any files exist, return ONLY:
   `EXISTING_SPECS_FOUND: true`

2. Read project metadata (whichever exist):
   - `README.md`, `README.rst`, `README.txt`
   - `CONTRIBUTING.md`, `CONTRIBUTING.rst`
   - `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `composer.json`
   - Glob `**/ADR*.md`, `docs/adr/**`, `docs/decisions/**`

3. Glob `*` for top-level structure. Glob `src/**` and `lib/**` if present.

4. Run `git log --oneline -20`.

5. Return a structured report:

```
PROJECT_TYPE: <type>
TECH_STACK: <languages, frameworks>
APPARENT_INTENT: <one sentence>

CANDIDATE_BEHAVIORS:
- <name>: given <precondition>, then <outcome>

CANDIDATE_CONSTRAINTS:
- <constraint>

CANDIDATE_INVARIANTS:
- <invariant>

CANDIDATE_SUBSYSTEMS:
- <name>: <description> (files: <glob pattern>)
```

---

If the sub-agent returns `EXISTING_SPECS_FOUND: true`, stop and tell the user: "A `.notarai/` directory with specs already exists. Use `/notarai-reconcile` to update existing specs instead of bootstrapping from scratch."

Otherwise, present the sub-agent's discovery report to the user and proceed to Phase 2.

---

## Phase 2 — Interview (two rounds)

### Round 2a — Structured questions (use the AskUserQuestion tool)

Call the **`AskUserQuestion` tool** with up to 4 questions at once. Do not ask these as plain text — use the tool so the user gets a selection UI.

Include these two questions in the call:

**Domain**

- Header: `"Domain"` (≤12 chars)
- multiSelect: false
- Options (one per schema enum value): Software, Presentation, Report, Course, Marketing
- Descriptions: briefly say what each domain is for (e.g., "Software — libraries, CLIs, services, apps")
- Pre-select the option that matches your inference and label it "(inferred)"

**Subsystem decomposition**

- Header: `"Subsystems"` (≤12 chars)
- multiSelect: true
- Generate one option per candidate subsystem you identified, plus an option: "Keep as a single spec (no subspecs)"
- Descriptions: one line per subsystem explaining what it does

Wait for the user's selections before proceeding to Round 2b.

### Round 2b — Open-ended questions (present as a numbered list, wait for response)

After Round 2a answers are received, present these free-form questions **all at once** as a numbered list. Do not ask them one at a time. Introduce with: "A few more questions before I draft your specs:"

1. **Intent**: Based on my reading, I believe the intent is: _[your one-sentence synthesis]_. Confirm, correct, or restate in your own words.

2. **Behaviors**: I identified these candidate behaviors (observable outcomes from a user/consumer perspective):
   _[your bulleted list]_
   What's missing, wrong, or should be reframed?

3. **Constraints**: Are there rules this project must always follow? These vary by domain — for a library it might be API stability guarantees; for a CLI it might be exit code contracts; for a course it might be lesson ordering rules.

4. **Invariants**: Are there conditions that must NEVER be true, regardless of any other change? (e.g., a library must never mutate caller state; a CLI must never write to stdout on success when `--quiet` is set)

5. **Exclusions**: What should be out of scope for spec coverage? (Examples: generated files, vendor dependencies, build output, third-party assets.)

---

## Phase 3 — Draft (write specs after user responds)

After the user answers the interview questions:

1. **Create the `.notarai/` directory** if it doesn't exist.

2. **Write `system.spec.yaml`** with the following structure. Use `schema_version: "0.7"`. Populate all fields from the interview answers:

```yaml
schema_version: '0.7'
domain: '[software|presentation|report|course|marketing|legal|education|infrastructure|research -- from Round 2a answer]'

intent: >
  [one or two sentences from the user's answer to question 2]

behaviors:
  - name: [snake_case_name]
    given: '[precondition]'
    then: '[observable outcome]'
  # one entry per confirmed behavior

constraints:
  - '[constraint statement]'
  # one entry per constraint from question 4

invariants:
  - '[invariant statement — a guarantee that must NEVER be violated, e.g. "no plaintext passwords stored"]'
  # one entry per invariant from question 4

subsystems: # omit this section if no subsystems were chosen
  - $ref: './[subspec-name].spec.yaml'

exclude: # omit if no exclusions
  - '[glob pattern]'

artifacts:
  code:
    - path: '[path or glob]'
      role: '[what this file/group does]'
  docs: # omit if no docs exist
    - path: '[path]'
      role: '[what this document covers]'

decisions: # omit if no notable decisions emerged
  - date: "[today's date]"
    choice: '[decision made]'
    rationale: '[why]'
```

3. **Write subspecs** for any subsystems the user confirmed. Each subspec is a separate `.notarai/[name].spec.yaml` using the same schema but scoped to that module. Include it in the system spec via `subsystems.$ref`.

4. **Validate** by running: `notarai validate .notarai/`
   - If validation fails, read the errors, fix the YAML, and re-run until it passes.
   - Do not present results to the user until validation passes.

5. **Present a summary** of what was created:
   - List each file written with a one-line description
   - Note any fields left sparse (e.g., "invariants section is thin — consider enriching this after you've reviewed the draft")
   - Suggest next steps: "Review and enrich the spec, then run `/notarai-reconcile` to check your existing code against it."
