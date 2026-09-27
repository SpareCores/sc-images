# Phase 02 — Write the docs (Claude)

Read with the repo root [`AGENTS.md`](../../AGENTS.md) and follow its
documentation architecture and house style.

## Read

- `.agents/work/<image>/facts.md` — facts from the code
- `.agents/context/shared.md` and `.agents/context/<image>.md` —
  approved maintainer answers
- The current `README.md`, `docs/`, and `CHANGELOG.md` for the image
- The family manual, for a variant image
- The image source, only to check a citation

## Write

- `images/<image>/README.md` — the manual
- `images/<image>/docs/*.md` — only if one README would fail a first
  pass; say why in section (b)
- `images/<image>/CHANGELOG.md` — prepend a dated entry holding any
  experiment log, calibration note, or other detail cut from the manual,
  kept as written. Create the file if needed.

Reply with sections (a) and (b) from `AGENTS.md`. Do not paste the docs
into chat.

## Rules

- Publish only claims found in `facts.md` or a context file. Put
  anything else in section (a) as an open question.
- If context and code disagree, stop and report it rather than picking
  one.
- Never edit `.agents/context/`. If an answer exists only in chat, ask
  the human to record it there first.
- Never delete an experiment log or calibration note; move it to
  `CHANGELOG.md`.
- For a variant image, write only what differs from the family manual.
