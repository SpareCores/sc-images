# Phase 03 — Review documentation (Gemini)

Read this prompt together with the repo root [`AGENTS.md`](../../AGENTS.md).
Use a **Gemini** model for this phase (Gemini Pro when available).

You are a **reviewer only**. Do not rewrite `README.md` or `docs/*.md`.
Do not "fix while reviewing". Your only write is the review report.

## Input

- `.agents/work/<image>/facts.md` — code-derived claims
- `.agents/context/<image>.md` — durable maintainer answers (may be
  absent on a first pass; then treat every open question in `facts.md`
  as unanswered)
- Current draft: `images/<image>/README.md`, `images/<image>/docs/**`,
  and `images/<image>/CHANGELOG.md` if present
- Image source tree (to spot-check citations and catch drift since extract)

Do not edit `.agents/context/`. If the draft states something that is
neither in `facts.md` nor in context, that is a blocker.

## Output

Write:

```text
.agents/work/<image>/review.md
```

## `review.md` structure (required)

```markdown
# Review: <image-folder-name>

## Summary
(2–5 sentences: ship / needs work / blocked — and why)

## Blockers
## Should-fix
## Nits

## Checklist
```

### Finding format

Each item under Blockers / Should-fix / Nits:

```markdown
- **`path:line`** — short title.
  Detail: what is wrong, why it matters, and what "correct" looks like
  (reference `facts.md`, `.agents/context/<image>.md`, or `AGENTS.md`
  style rules). Do not supply a full
  rewritten section unless a one-line suggestion is enough.
```

Severity:

| Level | Use when |
|-------|----------|
| Blocker | Factual error vs `facts.md`, context, or code; claim with no source in either file; README reduced to TOC; missing critical prerequisite; broken link/anchor that misleads |
| Should-fix | Style-guide violations that hurt clarity; imprecise scope; terminology drift; weak structure; raw experiment log or LLM calibration notes left in the README or `docs/` instead of `CHANGELOG.md`; detail removed from the manual with no changelog entry |
| Nit | Wrapping, punctuation consistency, minor wording |

### Checklist (tick or fail each)

Copy into `review.md` and mark `[x]` / `[ ]`:

```markdown
## Checklist

### Accuracy
- [ ] Claims match `facts.md`, `.agents/context/<image>.md`, or code; no invented topology or history
- [ ] Parameters, defaults, and outputs match the implementation
- [ ] Conflicts listed in `facts.md` are resolved in context or explicitly deferred
- [ ] Published text does not contradict `.agents/context/<image>.md`

### Architecture
- [ ] README is a self-contained overview (not a bare TOC)
- [ ] Opening description stays high-level (no premature env vars)
- [ ] Detail lives in `docs/` with in-context links from the README
- [ ] Content order: measure/why → methodology → run → outputs → limits
- [ ] Design history in the README and `docs/` is a retrospective summary, not a notes dump
- [ ] Raw experiment logs and LLM calibration notes live in `CHANGELOG.md`, and the manual links to them when that file exists

### Style (AGENTS.md house guide)
- [ ] PostgreSQL / `postgres` / `pgbench` terminology correct
- [ ] DBaaS / IaaS / vCPU casing correct
- [ ] "shell command" / "run via Docker" — not "bash script"; headings match
- [ ] ~80-character wrap; no one-sentence-per-line paragraphs
- [ ] Lists: intro sentence, consistent punctuation; shallow nesting
- [ ] No TODO/FIXME in published docs
- [ ] Niche terms glossed on first use
- [ ] Relative links and anchors verified after any moves
- [ ] No marketing fluff ("fleet", "delve", filler)

### Links
- [ ] Every internal link target exists and matches the link text's promise
```

## Done criteria

- [ ] `review.md` written at the path above
- [ ] Every finding has a file/line reference
- [ ] Checklist fully marked
- [ ] No edits to published documentation
