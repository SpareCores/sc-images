# Phase 02 — Write / restructure documentation (Claude)

Read this prompt together with the repo root [`AGENTS.md`](../../AGENTS.md).
Use a **Claude** model for this phase.

You are the writing agent. Follow the Role, Documentation architecture, and
House style guide in `AGENTS.md` strictly.

## Preconditions

1. Phase 01 has produced `.agents/work/<image>/facts.md`.
2. A human has recorded answers in `.agents/context/shared.md` and, when
   the image needs its own decisions, `.agents/context/<image>.md`
   (tracked; see [`.agents/context/README.md`](../context/README.md)).
   Chat answers do not count until they are in one of those files.
3. Do **not** invent answers for questions still open in `facts.md` and
   absent from both context files. Leave them as explicit questions in
   section (a) feedback, and omit unverified claims from the published
   text.
4. Do not edit anything under `.agents/context/`. If a needed answer is
   only in chat, ask the human to add it there first.
5. Do not delete experiment logs or calibration notes. Move them into
   `images/<image>/CHANGELOG.md`.

## Input

- `.agents/work/<image>/facts.md` — code-derived claims (regenerable)
- `.agents/context/shared.md` and `.agents/context/<image>.md` —
  maintainer-approved claims that are not in the code. Treat these as
  approved facts.
- Current `images/<image>/README.md`, `images/<image>/docs/**`, and
  `images/<image>/CHANGELOG.md` if any
- For a variant image, the family manual named in `AGENTS.md`
- The image source tree (only to double-check citations). When context
  and code disagree, stop and report the conflict; do not pick a side.

## Output

1. Update published docs in place:
   - `images/<image>/README.md` — the manual. Keep it one file when a
     reader can finish it in one pass.
   - `images/<image>/docs/*.md` — only when that single README would be
     too long for a first pass. Say why in section (b). Do not add
     `purpose.md`, `design-history.md`, or `references.md` by default.
   - `images/<image>/CHANGELOG.md` — when the rewrite cuts experiment
     logs, calibration notes, or other detail out of the manual. Create
     the file if needed. Prepend a dated entry. Keep the moved text. Do
     not rewrite it into a summary.
   A variant image's README states only how it differs from the family
   manual. Do not copy that manual.
2. In the chat reply, provide:
   - **a. DX & Architectural Feedback**
   - **b. Structural Rationale**
   - Do **not** paste the full Markdown into chat when files were edited;
     point at the paths instead.

## Writing rules (hard)

- State only facts that appear in `facts.md`, `.agents/context/shared.md`,
  or `.agents/context/<image>.md` in the README and `docs/`. Everything
  else → open question, not prose. Text moved into `CHANGELOG.md` is
  preserved as a log, not treated as a new claim.
- The README is the whole manual unless a first-pass reader would not
  get through it. Never reduce it to a table of contents.
- Opening paragraph: general overview only — no env var names. A Usage
  section in the same README may list them. Move that list to
  `docs/usage.md` only when it makes the README too long to read once.
- Content order (adapt section titles to the image, keep the logic):

  1. What it measures and why
  2. How it works (high-level methodology)
  3. Running it via Docker
  4. Outputs and how to read them
  5. Limitations / out of scope
  6. Optional FAQ at the end, only when a few questions would still
     interrupt a first pass
  7. Links to `docs/` or `CHANGELOG.md` only when those files exist

- Methodology history in the README and `docs/` → retrospective
  narrative; summarize limiting factors; link to `CHANGELOG.md`.
- Raw experiment tables and LLM calibration notes belong in
  `CHANGELOG.md`, not in the manual. The changelog is exempt from the
  "summarize and drop" rule.
- Apply the full House style guide (terminology, 80-col wrap, no
  one-sentence-per-line, no TODOs in docs, link/anchor hygiene, no
  marketing fluff).
- After moving or splitting files, fix every relative link and drop
  "in this folder" phrasing.

## Done criteria

- [ ] README edited on disk
- [ ] Any `docs/` page was added because one README fails a first pass,
      and section (b) says why
- [ ] Detail cut from the manual is in `CHANGELOG.md`, not discarded
- [ ] Chat includes (a) and (b)
- [ ] No claim in the published text lacks a `facts.md` or context basis
      (`shared.md` or the image context file)
- [ ] Style guide checklist mentally passed (wrap, terminology, headings
      match body wording)
