# Phase 03 — Review the docs (Gemini Pro)

Read with the repo root [`AGENTS.md`](../../AGENTS.md) and the
[images style manual](../../../sc-docs/tw-materials/images-style-manual.md),
which holds every editorial rule.

Review only. Your writes are `review.md` and new findings appended to
the facts file (see [Update the facts file](#update-the-facts-file)); do
not fix the docs.

## Read

- The images style manual. If it cannot be read, report that as a
  blocker and skip the Editorial section.
- `.agents/facts/<image>.md`. If its `Source commit` is stale (see
  `AGENTS.md`), report that as a blocker.
- `.agents/context/shared.md` and `.agents/context/<image>.md` if it
  exists. A question in the facts file that neither answers is still
  open.
- The image's `README.md`, `docs/`, and `CHANGELOG.md`
- The family manual, for a variant image

## Check Claims Against the Facts File

Check every doc claim against the facts file and the context files, not
against the image source. Do not reread the source code to re-verify
facts the facts file already covers.

Open a specific source file only when the facts file says nothing about
the issue a doc claim raises. Read only the files that issue needs, for
example the one script that sets a default the facts file does not
list. Cite every such file as `path:line` in the finding.

## Update the Facts File

When a source-file check turns up something the facts file does not
cover, append it to `.agents/facts/<image>.md` after writing `review.md`:

- Add each finding as a bullet under the matching facts section (for
  example, Parameters and defaults), ending with `(added in review)`.
- Cite `path:line` or `path:start-end` for every added bullet, as step 1
  does.
- Add a question that only a maintainer can answer to Open questions for
  maintainers, not to the other sections.
- Before appending a finding, run
  `git log --oneline <source-commit>..HEAD -- <path>` for each source
  file it cites. If there is output, the file changed after the extract:
  do not append the finding; report the stale facts as a blocker.
- Add a newly read code path to the `Inputs` line only if no existing
  `Inputs` pathspec covers it, using the same format as step 1. Never add
  `README.md`, `docs/`, `CHANGELOG.md`, or the facts file.
- Do not change the `Source commit` line or rewrite existing facts.
  Appending findings does not make the facts stale, because the facts
  file is not in `Inputs`.
- List what you added under "Additional Review Findings" in `review.md`, or
  write `None.`

## Write

`.agents/work/<image>/review.md`:

```markdown
# Review: <image>

## Summary

Ship, needs work, or blocked, and why, in 2–5 sentences.

## Blockers

### Conflicts

Every doc claim that contradicts the facts file, or a source file read for an issue the facts file does not cover. Cite both sides: the doc `path:line`, and the facts file section or the code `path:line`.

### Other Blockers

## Should-fix

Check if existing, but not mandatory documentation files are justified. If not, mark as a should-fix. If the docs use an acronym outside the style manual's exempt list, treat `images/<image>/docs/references.md` as a mandatory glossary.

## Editorial

Check the docs against the style manual, and cite the manual section in each finding:

### Grammar
### Voice and Phrasing

- incorrect use of singular vs. plural phrasing
- inconsistent references such as "the benchmark" vs. "this benchmark" (or "this image" for non-benchmark images)
- run-on sentences, excessive asides, and unrelated tangents
- phrases from the manual's avoid list, hype, exclamation marks, humor, or emoji
- "I" instead of "we"; "simply", "just", "easy", or "simple" in instructions
- terminology that breaks the manual's Terminology section, such as "infra" or "bash script"

### Headings and Formatting

- headings not in [APA title case](https://apastyle.apa.org/style-grammar-guidelines/capitalization/title-case); the README H1 stays the image folder name
- inconsistent use of subheadings, bold labels, and lists within a file or across the image's docs
- lists without an intro sentence, or with inconsistent case and punctuation

### Punctuation and Numbers

- em dashes, en dashes, curly quotes or apostrophes, trailing ellipses, or `&` in prose
- `~` in Markdown prose that is not escaped as `\~`
- thousands separators, ranges, and units that do not follow the manual

### Acronyms

- incorrect acronym format. Correct by suggesting the "ACRONYM (Expansion)" format, with the link on the expansion at first use only
- acronyms outside the manual's exempt list that are not expanded on first body use in each file, or that are expanded in a heading. Exclude `images/<image>/CHANGELOG.md` from this check
- acronyms or custom abbreviations missing from `images/<image>/docs/references.md`
- plurals with an apostrophe (vCPU's), or custom abbreviations that break the manual's Custom Abbreviations rules

## Nits
## Additional Review Findings

Each bullet appended to the facts file, with its section and citation, or `None.`

## Checklist
```

Write each finding using the following format:

```markdown
- [ ] **`path:line`** — short line using a clickable relative link to the exact lines within the relevant files. Make sure to include the affected line number in the visible part of the link as well. If a problem exists in multiple places, subdivide the finding into checklist sub-items, and include line links for each instance.
  What is wrong, why it matters, and what correct looks like, citing
  the facts file, a context file, an AGENTS.md rule, or a style manual
  section.
```

| Severity | Use for |
| ---------- | --------- |
| Blocker: Conflict | A doc claim that contradicts the facts file or a source file checked for an uncovered issue, with citations for both; list under Blockers > Conflicts |
| Blocker: Other | A claim missing from the facts file, the context files, and the code; a stale facts file; a README reduced to a table of contents; a missing prerequisite; a misleading link; list under Blockers > Other Blockers |
| Should-fix | Style manual violations that hurt clarity; a `docs/` split that is not needed; a variant README repeating the family manual; experiment logs or LLM notes left in the manual; detail cut with no `CHANGELOG.md` entry; a missing required `references.md` |
| Editorial | Style manual violations in grammar, voice, terminology, headings, formatting, punctuation, numbers, and acronyms that do not hurt clarity |
| Nit | Wrapping, whitespace, minor wording |

## Checklist

Copy into `review.md` and mark each item `[x]` or `[ ]`:

```markdown

### Accuracy

- [ ] Every claim traces to the facts file or a context file
- [ ] Parameters, defaults, and outputs match the facts file
- [ ] Every new finding from a source file is appended to the facts file
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
- [ ] Headings in APA title case; README H1 is the folder name
- [ ] ~80-column wrap; no one-sentence-per-line paragraphs
- [ ] Lists have an intro sentence and consistent punctuation
- [ ] Niche terms glossed; no TODO/FIXME; no avoid-list phrases
- [ ] Acronyms expanded on first use in each file and listed in references.md
- [ ] No em/en dashes, curly quotes, or trailing ellipses; `~` escaped in prose
- [ ] Numbers, ranges, and units follow the style manual
- [ ] Every link and anchor resolves and delivers what its text promises
```
