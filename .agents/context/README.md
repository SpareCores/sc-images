# Maintainer context

Answers that the code cannot give: production topology, cost
trade-offs, intentional omissions. Humans write these files; LLMs read
them and never edit them.

- `shared.md` — decisions that apply to every image
- `<image>.md` — decisions for one image, named after its folder under
  `images/` (e.g. `benchmark-pgbench-postgres.md`)

Experiment logs and calibration notes go in `images/<image>/CHANGELOG.md`,
not here.

## Format

```markdown
# Context: <image>

## Can the client run on the database server?

Yes. The benchmark is agnostic to colocated or remote deployment.
```

- The heading is the question and the paragraph under it is the answer.
- Keep headings stable once written; later extracts match on them.
- Git history records who decided and when, so no dates or labels.
- When a decision changes, edit the answer. Add a `Superseded:` note only
  if the old answer still matters.
