# Phase 03 — Review the docs (Gemini)

Read with the repo root [`AGENTS.md`](../../AGENTS.md).

Review only. Your only write is `review.md`; do not fix the docs.

## Read

- `.agents/work/<image>/facts.md`
- `.agents/context/shared.md` and `.agents/context/<image>.md` if it
  exists. A question in `facts.md` that neither answers is still open.
- The image's `README.md`, `docs/`, and `CHANGELOG.md`
- The family manual, for a variant image
- The image source, to spot-check citations and catch changes since the
  extract

## Write

`.agents/work/<image>/review.md`:

```markdown
# Review: <image>

## Summary
Ship, needs work, or blocked, and why, in 2–5 sentences.

## Blockers
## Should-fix
## Nits
## Checklist
```

Write each finding as:

```markdown
- **`path:line`** — short title.
  What is wrong, why it matters, and what correct looks like, citing
  facts.md, a context file, or an AGENTS.md rule.
```

| Severity | Use for |
|----------|---------|
| Blocker | A claim that contradicts or is missing from `facts.md`, the context files, and the code; a README reduced to a table of contents; a missing prerequisite; a misleading link |
| Should-fix | Style-guide violations that hurt clarity; a `docs/` split that is not needed; a variant README repeating the family manual; experiment logs or LLM notes left in the manual; detail cut with no `CHANGELOG.md` entry |
| Nit | Wrapping, punctuation, minor wording |

## Checklist

Copy into `review.md` and mark each item `[x]` or `[ ]`:

```markdown
### Accuracy
- [ ] Every claim traces to facts.md, a context file, or the code
- [ ] Parameters, defaults, and outputs match the implementation
- [ ] Conflicts from facts.md are resolved or explicitly deferred

### Structure
- [ ] README reads as a complete manual in one pass
- [ ] Opening paragraph has no env var names
- [ ] Sections follow the order in AGENTS.md
- [ ] Every docs/ page is justified; simple images stay one README
- [ ] FAQ, if present, is short and does not repeat Limitations
- [ ] History is a short retrospective; the full log is in CHANGELOG.md
- [ ] A variant README states only the difference from its family manual

### Style
- [ ] Product names spelled as upstream; program names in backticks
- [ ] "run via Docker", not "bash script"; headings match the body
- [ ] ~80-column wrap; no one-sentence-per-line paragraphs
- [ ] Lists have an intro sentence and consistent punctuation
- [ ] Niche terms glossed; no TODO/FIXME; no marketing phrasing
- [ ] Every link and anchor resolves and delivers what its text promises
```
