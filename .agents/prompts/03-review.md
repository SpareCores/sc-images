# Phase 03 — Review the docs (Gemini Pro)

Read with the repo root [`AGENTS.md`](../../AGENTS.md).

Review only. Your only write is `review.md`; do not fix the docs.

## Read

- `.agents/facts/<image>.md`. If its `Source commit` is stale (see
  `AGENTS.md`), report that as a blocker.
- `.agents/context/shared.md` and `.agents/context/<image>.md` if it
  exists. A question in the facts file that neither answers is still
  open.
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

Check if existing, but not mandatory documentation files are justified. If not, mark as a should-fix. If there are acronyms anywhere in the documentation, treat `references.md` as a mandatory glossary.

## Conflicts
## Editorial

Check for the following, and suggest changes if necessary:

### Grammar
### Phrasing

- incorrect use of singular vs. plural phrasing
- inconsistent pronoun usage such as "the benchmark" vs "this benchmark"
- run-on sentences, excessive asides, and unrelated tangents

### Formatting

- incorrect title-casing in headings. Use the [APA Style](https://apastyle.apa.org/style-grammar-guidelines/capitalization/title-case) for headings and subheadings
- Inconsistent usage of subheadings, bolded title phrases, and lists within a file or across all documentation

### Acronyms

- incorrect acronym format. Correct by suggesting changing it to "Acronym (extension)" format
- incorrect first-use of acronyms. Correct by suggesting expanding the acronym on first use in each file, and adding it to `images\<image>\docs\references.md` if it is a project glossary term. Exclude `images\<image>\CHANGELOG.md` from this check.

## Nits
## Checklist
```

Write each finding using the following format:

```markdown
- [ ] **`path:line`** — short line using a clickable relative link to the exact lines within the relevant files. Make sure to include the affected line number in the visible part of the link as well.
  What is wrong, why it matters, and what correct looks like, citing
  the facts file, a context file, or an AGENTS.md rule.
```

| Severity | Use for |
| ---------- | --------- |
| Blocker | A claim that contradicts or is missing from the facts file, the context files, and the code; a README reduced to a table of contents; a missing prerequisite; a misleading link |
| Should-fix | Style-guide violations that hurt clarity; a `docs/` split that is not needed; a variant README repeating the family manual; experiment logs or LLM notes left in the manual; detail cut with no `CHANGELOG.md` entry |
| Conflicts | A doc claim versus what the code does, with citations for both |
| Editorial | Style-guide violations that hurt clarity; a `docs/` split that is not needed; a variant README repeating the family manual; experiment logs or LLM notes left in the manual; detail cut with no `CHANGELOG.md` entry |
| Nit | Wrapping, punctuation, minor wording |

## Checklist

Copy into `review.md` and mark each item `[x]` or `[ ]`:

```markdown

### Accuracy

- [ ] Every claim traces to the facts file, a context file, or the code
- [ ] Parameters, defaults, and outputs match the implementation
- [ ] Conflicts from the facts file are resolved or explicitly deferred

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
