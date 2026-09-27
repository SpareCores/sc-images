# Phase 02 — Write / restructure documentation (Claude)

Read this prompt together with the repo root [`AGENTS.md`](../../AGENTS.md).
Use a **Claude** model for this phase.

You are the writing agent. Follow the Role, Documentation architecture, and
House style guide in `AGENTS.md` strictly.

## Preconditions

1. Phase 01 has produced `.agents/work/<image>/facts.md`.
2. A human has recorded answers in `.agents/context/<image>.md` (tracked;
   see [`.agents/context/README.md`](../context/README.md)). Chat answers
   do not count until they are in that file — a later extract would
   otherwise lose them.
3. Do **not** invent answers for questions still open in `facts.md` and
   absent from context. Leave them as explicit questions in section (a)
   feedback, and omit unverified claims from the published text.
4. Do not edit `.agents/context/<image>.md`. If a needed answer is only
   in chat, ask the human to add it there first.
5. Do not delete experiment logs or calibration notes. Move them into
   `images/<image>/CHANGELOG.md`.

## Input

- `.agents/work/<image>/facts.md` — code-derived claims (regenerable)
- `.agents/context/<image>.md` — maintainer-approved claims that are not
  in the code (durable). Treat these as approved facts.
- Current `images/<image>/README.md`, `images/<image>/docs/**`, and
  `images/<image>/CHANGELOG.md` if any
- The image source tree (only to double-check citations). When context
  and code disagree, stop and report the conflict; do not pick a side.

## Output

1. Update published docs in place:
   - `images/<image>/README.md` — self-contained detailed overview
   - `images/<image>/docs/*.md` — fine detail pages as needed
     (create, split, rename, or delete pages when the architecture calls
     for it; justify in section (b))
   - `images/<image>/CHANGELOG.md` — when the rewrite cuts experiment
     logs, calibration notes, or other detail out of the manual. Create
     the file if needed. Prepend a dated entry. Keep the moved text. Do
     not rewrite it into a summary.
2. In the chat reply, provide:
   - **a. DX & Architectural Feedback**
   - **b. Structural Rationale**
   - Do **not** paste the full Markdown into chat when files were edited;
     point at the paths instead.

## Writing rules (hard)

- State only facts that appear in `facts.md` or in
  `.agents/context/<image>.md` in the README and `docs/`. Everything
  else → open question, not prose. Text moved into `CHANGELOG.md` is
  preserved as a log, not treated as a new claim.
- README must remain understandable **without** opening `docs/`. Never
  reduce it to a table of contents.
- Opening description: general overview only — no env var names or
  config trivia.
- Content order (adapt section titles to the image, keep the logic):

  1. What it measures and why
  2. How it works (high-level methodology)
  3. Running it via Docker
  4. Outputs and how to read them
  5. Limitations / out of scope
  6. Links into detailed `docs/*.md` from the relevant paragraphs

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

- [ ] README + docs edited on disk
- [ ] Detail cut from the manual is in `CHANGELOG.md`, not discarded
- [ ] Chat includes (a) and (b)
- [ ] No claim in the published text lacks a `facts.md` or context basis
- [ ] Style guide checklist mentally passed (wrap, terminology, headings
      match body wording)
