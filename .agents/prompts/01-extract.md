# Phase 01 — Extract facts from code (recent GPT reasoning model)

Read with the repo root [`AGENTS.md`](../../AGENTS.md).

Your only write is the facts file. Do not edit docs, `CHANGELOG.md`,
code, or anything under `.agents/context/`.

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

`.agents/facts/<image>.md`, overwriting any previous version. The file
is tracked in Git, so later runs can skip extraction while it is still
current.

```markdown
# Facts: <image>

Source commit: <output of `git rev-parse HEAD`>
Inputs: images/<image>, .agents/context/shared.md, .agents/context/<image>.md, <other paths read>

## What is measured
## Workload
## Parameters and defaults
## Outputs and schema
## How to run
## Dependencies and platforms
## Open questions for maintainers
```

| Section | Include |
| --------- | --------- |
| What is measured | Headline metric, what varies between runs, what is held constant |
| Workload | Scripts, queries, concurrency, warmup and duration, topology the code implies |
| Parameters and defaults | Env vars, flags, constants: name, default, meaning |
| Outputs and schema | Printed output or files, field names, units |
| How to run | Image tag, required env, minimal command from the entrypoint |
| Dependencies and platforms | `DEPENDS_ON`, base images, `PLATFORMS`, sibling images |
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
- Write bullets and tables, not publishable prose. Never copy
  maintainer answers into the facts file; cite them.
- Write `None found.` under an empty section.
- List every path you relied on in `Inputs`: the image folder, both
  context files, and any sibling image, family manual, or `.github/`
  file. The staleness check in `AGENTS.md` watches exactly those paths.
- Extract from committed files. If any input path has uncommitted
  changes, append `(uncommitted changes)` to the `Source commit` line.
