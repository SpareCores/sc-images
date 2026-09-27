# Phase 01 — Extract methodology from code (OpenAI GPT)

Read this prompt together with the repo root [`AGENTS.md`](../../AGENTS.md).
Use an **OpenAI GPT** model for this phase.

You are **read-only** except for `facts.md`. Do not edit `README.md`,
`docs/*.md`, `CHANGELOG.md`, Dockerfiles, application code, or
`.agents/context/<image>.md`. Humans own the context file and the
changelog; a re-extract must not touch either.

## Input

The human names a target image folder, e.g. `images/benchmark-pgbench-postgres/`.

Read at least:

- Everything under that folder (Dockerfile, `benchmark.py` / scripts, SQL,
  config, existing `README.md`, `docs/`, and `CHANGELOG.md` if present)
- Sibling images referenced by `DEPENDS_ON` or README links
- Build metadata: `BUILD_ARGS`, `DEPENDS_ON`, `PLATFORMS`, `CONTEXT`, `ZRAM`,
  `SCCACHE`
- Relevant CI pieces under `.github/` if the image has special build
  behavior
- `.agents/context/<image-folder-name>.md` when it exists (maintainer
  answers from earlier runs). Read it; do not modify it.

## Output

Write:

```text
.agents/work/<image-folder-name>/facts.md
```

Example: `.agents/work/benchmark-pgbench-postgres/facts.md`.

Create the directory if needed. Overwriting `facts.md` is expected on a
fresh extract. Never copy maintainer answers into `facts.md`.

## `facts.md` structure (required)

Use these exact top-level headings:

```markdown
# Facts: <image-folder-name>

## What is measured
## Workload
## Parameters and defaults
## Outputs and schema
## How to run
## Dependencies and platforms
## Conflicts between code and existing docs
## Open questions for maintainers
```

### Rules for every factual claim

- Cite `path:line` (or `path:start-end`) for **every** code-derived
  claim under the first six sections. A claim taken from
  `.agents/context/<image>.md` is cited as `source: context` plus the
  heading, not a line number.
- Prefer code and metadata files over existing prose. `CHANGELOG.md` is
  a history log, not a source of runtime facts. Do not copy it into the
  factual sections. If it contradicts the code, list that under
  **Conflicts**.
- If the README asserts something the code does not support, list it under
  **Conflicts**, not under the factual sections.
- Write **no publishing-ready prose**. Bullets, tables, and short neutral
  statements only. No rewritten README draft in this file.

### Section guidance

| Section | Include |
|---------|---------|
| What is measured | Headline metric(s), what varies across runs, what is held constant |
| Workload | Scripts, SQL, concurrency model, warmup/settle/duration, topology assumptions that appear in code |
| Parameters and defaults | Env vars, CLI flags, constants — name, meaning, default, citation |
| Outputs and schema | Printed metrics, JSON/files, field names and units |
| How to run | Docker image tag, required env, minimal command reconstructed from code/entrypoint |
| Dependencies and platforms | `DEPENDS_ON`, `FROM` bases, `PLATFORMS`, sibling server images |
| Conflicts | Doc claim vs code reality, each with citations on both sides |
| Open questions | Anything a maintainer must confirm (production topology, historical experiments, intentional omissions). Phrase as questions. If `.agents/context/<image>.md` already answers one, do not repeat it here — cite the context heading instead under the matching factual section, marked `source: context` (no `file:line`). If context contradicts the code, list that under Conflicts. |

## Done criteria

- [ ] `facts.md` exists at the path above
- [ ] Every claim in the first six sections has a citation
- [ ] Conflicts and open questions are non-empty whenever uncertainty exists
  (empty sections must say `None found.` explicitly)
- [ ] No edits to published docs, `CHANGELOG.md`, source files, or `.agents/context/`
