# Maintainer context

Durable answers that are **not** in the image code: production topology,
historical experiment outcomes, cost trade-offs, intentional omissions.

One file per image, named after the folder under `images/`:

```text
.agents/context/benchmark-pgbench-postgres.md
```

Phase 01 regenerates `.agents/work/<image>/facts.md` and must not edit
files here. Phase 02 and 03 read them. Humans write them.

## Shape

```markdown
# Context: <image-folder-name>

Maintainer-approved facts that the code does not state. Each heading is
the question; the paragraph under it is the answer.

## Can the client run on the database server?

Yes. The benchmark is agnostic to colocated or remote deployment.
```

Rules:

- One heading per decision. Re-extracts match on the heading text, so
  keep headings stable once written.
- Who decided, and when, lives in Git history. Do not add timestamps
  or an `Answer:` label.
- If a decision changes, edit the paragraph. Add a short `Superseded:`
  note only when the old answer still matters.
- Do not put publishing-ready prose here. Short answers only.
