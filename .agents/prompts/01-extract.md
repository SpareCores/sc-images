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
- The previous `.agents/facts/<image>.md`, if it exists, only to collect
  bullets marked `(added in review)` and the existing Previous Review
  Findings (see below). Never list the facts file in `Inputs`.

## Recheck Previous Review Findings

The review step (`03-review.md`) appends findings to the facts file,
marked `(added in review)`. Before overwriting the file, recheck each of
them, together with any bullet already under Previous Review Findings:

- Read the source files the finding cites, and any others it needs,
  together with the context files.
- If the code confirms it, keep it, with citations updated to the
  current lines.
- If the code or a context answer shows it is wrong or incomplete,
  rewrite it to match, and cite the new evidence.
- If the code no longer supports it at all, drop it, and add a bullet
  saying what was removed and why.
- If context contradicts the code, list it under Conflicts.
- Move every finding you keep under Previous Review Findings, without
  the `(added in review)` marker. Add the code paths you read to
  `Inputs` if they are not already covered, following the `Inputs` rule
  below.

## Write

`.agents/facts/<image>.md`, overwriting any previous version once the
previous review findings are rechecked. The file is tracked in Git, so
later runs can skip extraction while it is still current.

```markdown
# Facts: <image>

Source commit: <output of `git rev-parse HEAD`>
Inputs: images/<image> ':(exclude)images/<image>/README.md' ':(exclude)images/<image>/docs' ':(exclude)images/<image>/CHANGELOG.md' .agents/context/shared.md .agents/context/<image>.md <other code paths read>

## What is measured
## Workload
## Parameters and defaults
## Outputs and schema
## How to run
## Dependencies and platforms
## Previous Review Findings
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
| Previous Review Findings | Review findings rechecked against the code and context: kept, corrected, or removed, each with citations |
| Open questions | What only a maintainer can answer, phrased as questions |

## Rules

- Cite `path:line` or `path:start-end` for every claim in the first seven
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
- List every code and context path you relied on in `Inputs`: the image
  folder, both context files, and any sibling image, family harness
  (such as `vllm-common/`), or `.github/` file. Write them as
  space-separated Git pathspecs, so the line can be pasted into the
  staleness check in `AGENTS.md`, which watches exactly those paths.
- Keep docs out of `Inputs`, so a docs-only commit never makes the facts
  stale. For every folder you list, add `':(exclude)<folder>/README.md'`,
  `':(exclude)<folder>/docs'`, and `':(exclude)<folder>/CHANGELOG.md'`.
  Do not list a family manual, `README.md`, `docs/`, `CHANGELOG.md`, or
  the facts file itself, even though you read them.
- Extract from committed files. If any `Inputs` path has uncommitted
  changes, append `(uncommitted changes)` to the `Source commit` line.
  Uncommitted docs do not count, because they are not in `Inputs`.
