# Phase 02 — Write the docs (Claude Opus)

Read with the repo root [`AGENTS.md`](../../AGENTS.md) and follow its
documentation architecture. Follow the
[images style manual](../../../sc-docs/tw-materials/images-style-manual.md)
for all editorial rules.

## Read

- The images style manual. If it cannot be read, stop and ask.
- `.agents/facts/<image>.md` — facts from the code. If its `Source
  commit` is stale (see `AGENTS.md`), stop and ask for a re-extract.
- `.agents/context/shared.md` and `.agents/context/<image>.md` —
  approved maintainer answers
- The current `README.md`, `docs/`, and `CHANGELOG.md` for the image
- The family manual, for a variant image
- The image source, only to check a citation

## Write

- `images/<image>/README.md` — the brief manual
- `images/<image>/docs/*.md` — only if one README would fail a first
  pass; say why in section (b)
- `images/<image>/docs/references.md` — the acronym glossary, whenever
  the style manual's Glossary section requires one. Create it from
  [`glossary-template.md`](../../../sc-docs/tw-materials/glossary-template.md);
  `images/benchmark-pgbench-postgres/docs/references.md` is a filled-in
  example.
- `images/<image>/CHANGELOG.md` — prepend a dated entry holding any
  experiment log, calibration note, or other detail cut from the manual,
  kept as written. Create the file if needed.

Reply with sections (a) and (b) from `AGENTS.md`. Do not paste the docs
into chat.

## Rules

- Publish only claims found in the facts file or a context file. Put
  anything else in section (a) as an open question.
- If context and code disagree, stop and report it rather than picking
  one.
- Never edit `.agents/context/`. If an answer exists only in chat, ask
  the human to record it there first.
- Never delete an experiment log or calibration note; move it to
  `CHANGELOG.md`.
- For a variant image, write only what differs from the family manual.
- Follow the style manual for wording, terminology, acronyms, headings,
  formatting, punctuation, numbers, and links.

## Editorial self-check

Before replying, check every file you wrote for the following, and fix
what you find:

- em dashes, en dashes, curly quotes, or trailing ellipses
- `~` in Markdown prose that is not escaped as `\~`
- phrases from the style manual's avoid list, "simply" or "just" in
  instructions, "I", exclamation marks, or emoji
- acronyms outside the exempt list that are not expanded on first use in
  each file, or are missing from `references.md` (`CHANGELOG.md` is exempt)
- headings not in APA title case (the README H1 stays the folder name)
- numbers, ranges, and units that do not follow the style manual
