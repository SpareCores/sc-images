# Phase 01 — Extract facts from code (OpenAI GPT)

Read with the repo root [`AGENTS.md`](../../AGENTS.md).

Your only write is `facts.md`. Do not edit docs, `CHANGELOG.md`, code,
or anything under `.agents/context/`.

## Read

For the image folder the human names:

- Every file in the folder: Dockerfile, harness code, SQL, config,
  `README.md`, `docs/`, `CHANGELOG.md`
- Build metadata (`BUILD_ARGS`, `DEPENDS_ON`, `PLATFORMS`, `CONTEXT`,
  `ZRAM`, `SCCACHE`) and any sibling image it depends on or links to
- `.github/` only if the image has special build behavior
- `.agents/context/shared.md` and `.agents/context/<image>.md` if it
  exists
- The family manual, for a variant image (see `AGENTS.md`)

## Write

`.agents/work/<image>/facts.md`, overwriting any previous version:

```markdown
# Facts: <image>

## What is measured
## Workload
## Parameters and defaults
## Outputs and schema
## How to run
## Dependencies and platforms
## Conflicts between code and existing docs
## Open questions for maintainers
```

| Section | Include |
|---------|---------|
| What is measured | Headline metric, what varies between runs, what is held constant |
| Workload | Scripts, queries, concurrency, warmup and duration, topology the code implies |
| Parameters and defaults | Env vars, flags, constants: name, default, meaning |
| Outputs and schema | Printed output or files, field names, units |
| How to run | Image tag, required env, minimal command from the entrypoint |
| Dependencies and platforms | `DEPENDS_ON`, base images, `PLATFORMS`, sibling images |
| Conflicts | A doc claim versus what the code does, with citations for both |
| Open questions | What only a maintainer can answer, phrased as questions |

## Rules

- Cite `path:line` or `path:start-end` for every claim in the first six
  sections. Cite a context answer as `source: context`, with the file
  and heading.
- Do not repeat a question the context files already answer; use the
  answer instead. If context contradicts the code, list it under
  Conflicts.
- Treat README and `CHANGELOG.md` text as claims to verify, not as
  facts.
- Write bullets and tables, not publishable prose. Never put maintainer
  answers in `facts.md`.
- Write `None found.` under an empty section.
