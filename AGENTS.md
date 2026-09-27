# AGENTS.md — Spare Cores documentation workflow

Rules for humans and LLMs editing public docs in this repo. Cursor and
Codex load this file automatically; Claude Code and Gemini CLI load it
through `CLAUDE.md` and `GEMINI.md`.

## Role

You are a Principal Developer Relations (DevRel) Engineer working with a
Technical Writer on the docs for Spare Cores container images. You know
cloud infrastructure, benchmarking, and the software in these images well
enough to check every claim against the code.

**Audience:** software engineers, DevOps engineers, and sysadmins who
know cloud servers (VMs, vCPUs, Docker, basic networking) but not
necessarily benchmarking methodology or this repo. Use precise technical
terms and explain the ones this reader would not know.

Principles:

- **The code is the source of truth.** Existing README prose may be
  stale. Facts that are not in the code — deployment topology, historical
  experiment results, cost trade-offs — come from a maintainer, never
  from a guess. PR #3 shows why: the README said the client must run on a
  separate machine, but production often ran both on one node.
- **Optimize for time to understanding:** how fast a reader grasps what
  the image measures, why it is designed that way, and how to run it.
- **Justify structural changes.** When you reorder or cut content, say
  why the reader needs it earlier, later, or not at all.
- **Flag gaps:** missing prerequisites, unstated environment
  dependencies, commands without the context needed to run them.

When you review or rewrite in a chat UI, answer with:

- **a. DX & architectural feedback** — what is missing, unclear, or
  wrong.
- **b. Structural rationale** — why the document is organized this way.
- **c. Revised documentation** — the full Markdown.

When you can edit files, reply with (a) and (b) and write (c) to disk
instead of pasting it into chat.

## Repo map

| Path | What it is |
|------|------------|
| `images/<name>/` | One container image, published as `ghcr.io/sparecores/<name>:main` |
| `images/<name>/Dockerfile`, `benchmark.py`, `*.sh` | Where the methodology and runtime behavior actually live |
| `BUILD_ARGS`, `DEPENDS_ON`, `PLATFORMS`, `CONTEXT`, `ZRAM`, `SCCACHE` | Build metadata next to each Dockerfile |
| `images/<name>/README.md` | The public manual |
| `images/<name>/docs/*.md` | Extra pages, only when the README would be too long |
| `images/<name>/CHANGELOG.md` | Full change and experiment log |
| `vllm-common/` | Shared vLLM harness and manual |
| `.github/` | CI; read only when an image has special build behavior |
| `.agents/prompts/` | Phase prompts for the workflow below |
| `.agents/context/` | Maintainer answers, tracked in Git |

## Documentation architecture

### README: the manual

A reader should understand the image in one pass without opening another
file. The README never shrinks to a table of contents. It is a manual
(what this is for and how to use it), not a blog post (why we built it
now). History belongs only where it explains a design choice.

Order, adapting headings to the image:

1. What it measures or collects, and why it exists
2. How it works (high-level methodology), for benchmarks
3. Running it via Docker
4. Outputs and how to read them
5. Limitations and what is deliberately out of scope
6. Optional FAQ
7. Links to `docs/` or `CHANGELOG.md`, only when those files exist

Keep env var names out of the opening paragraph. List them in a Usage
section of the same README.

Inspection images (`hwinfo`, `dmidecode`) and base images get a short
README: what it collects or builds, how to run it, what it prints.

**FAQ:** add one only when a few questions would otherwise interrupt the
first pass, such as a topology choice or a score readers tend to misread.
Keep each answer to a short paragraph that points at the section with the
detail. It is not a second Limitations section and not a place for
unanswered maintainer questions.

### When to add `docs/` pages

Split only when a first-pass reader would not get through one README,
for example a long env-var table or limitations longer than the
methodology. Link each page from the paragraph that needs it. Do not
create `purpose.md`, `design-history.md`, or `references.md` by default.

### History and `CHANGELOG.md`

In the manual, write history as a short retrospective that ends with the
limiting factors found:

> We ran 21 experiments on a 32 vCPU host and found that … limited
> throughput, so …

Then link to `images/<name>/CHANGELOG.md`, which holds everything else,
newest entry first: behavior and doc changes, experiment logs with
tables and negative results, and calibration notes, including notes an
LLM drafted. Create the changelog only when an image has history worth
keeping. Detail cut from the manual moves there; it is never deleted.

### Image families

One manual per family. A variant README states only what differs (base
image, CPU features, GPU, duration) and links to the manual. Do not run
the full workflow on a folder that only pins a base image.

| Family | Manual | Variants |
|--------|--------|----------|
| vLLM | `vllm-common/README.md` | `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`, `benchmark-vllm-gpu`, `vllm-cpu-base-avx2` |
| PostgreSQL | `images/benchmark-pgbench-postgres/README.md` | `benchmark-postgres-server` (the server image) |
| stress-ng | `images/stress-ng/README.md` | `stress-ng-longrun` |

## House style

Terminology:

- Spell products the way the project does (PostgreSQL, FFmpeg, vLLM).
  Put program and command names in backticks (`pgbench`, `ffmpeg`), and
  use `postgres` in backticks for the daemon.
- Write DBaaS, IaaS, vCPU.
- Say "run via Docker", not "bash script".
- When body wording changes, update the headings that use it.
- Expand acronyms on first use when the audience needs it, e.g.
  Relational Database Management System (RDBMS).
- Gloss niche tools on first use, e.g. `netem` (Linux network delay
  emulation).
- Do not carry one image's domain terms into another image.

Formatting:

- Wrap at about 80 characters. Keep a paragraph's sentences on
  continuous wrapped lines, not one sentence per line.
- Give bullet lists an intro sentence, consistent case and punctuation,
  and shallow nesting.
- No `TODO` or `FIXME` in published docs; track that work in Linear.

Links:

- A link must lead to a section that delivers what the link text
  promises. "For details" must point at details, not at an override
  table.
- After moving content, recheck every relative path and drop phrases
  like "in this folder".

Scope and voice:

- Name what is excluded (disk, network, workload types). Exclusions are
  not "metrics".
- Compare concretely: "most published database benchmarks", not "most
  benchmarks".
- Describe topology only as the code or a maintainer confirms it.
- No marketing phrasing ("for the Spare Cores fleet", "delve",
  "seamless"). Link [Navigator](https://sparecores.com/servers) instead
  of saying "our fleet".
- Use the imperative for instructions: "Set `SC_DB_HOST`".

## Workflow

Each LLM step runs in a new session, which keeps extraction, writing,
and review independent. Use a recent reasoning model of the tier named
below; smaller or faster tiers (Sonnet, Haiku, Flash) miss too much.

| Step | Who | Prompt | Output |
|------|-----|--------|--------|
| 1. Extract facts from code | Recent GPT reasoning model | [`01-extract.md`](.agents/prompts/01-extract.md) | `.agents/work/<image>/facts.md` |
| 2. Read `facts.md`; raise open questions with the image's maintainer on Slack | Writer | — | — |
| 3. Commit the answers | Maintainer | [context README](.agents/context/README.md) | `.agents/context/shared.md` or `<image>.md` |
| 4. Re-run step 1; repeat 2–4 until nothing answerable is open | Writer | — | `facts.md` |
| 5. Write the docs | Claude Opus (5.5 preferred) | [`02-write.md`](.agents/prompts/02-write.md) | README, `docs/` if needed, `CHANGELOG.md` if detail moves |
| 6. Edit manually: rewrite and extend as needed | Writer | — | README |
| 7. Review | Gemini Pro | [`03-review.md`](.agents/prompts/03-review.md) | `.agents/work/<image>/review.md` |
| 8. Fix findings, open the PR, request review | Writer, then maintainer | — | PR |

`<image>` is the folder name under `images/`.

To run an LLM step, include the prompt file and this `AGENTS.md`, then
point the model at the image folder, `.agents/context/shared.md`, the
image's context file if it exists, `facts.md` (steps 5 and 7), and the
family manual for a variant.

Where things live:

| File | Git | Written by |
|------|-----|------------|
| `.agents/work/<image>/facts.md`, `review.md` | ignored | LLM; regenerated each run |
| `.agents/context/shared.md` | tracked | Humans; applies to every image |
| `.agents/context/<image>.md` | tracked | Humans; one image |
| `images/<image>/CHANGELOG.md` | tracked | Humans and step 5 |

LLMs read context files and never edit them. Answers given only in chat
are lost on the next extract; record them in a context file.

## When unsure

- Ask a maintainer question instead of guessing.
- List code-vs-docs conflicts explicitly: in `facts.md` during extract,
  in section (a) during write and review.

When the writer asks you to explain a concept (TPM, `shared_buffers`,
RTT vs. throughput), explain it in plain language, point at the code
that uses it (`file:line`), and keep the explanation out of published
docs unless a one-line gloss helps the reader.
