# Changelog rules

Rules for writing the root `CHANGELOG.md`. Readers use the Spare Cores Navigator's
benchmark data or run the images; they want to know which images exist, what they
measure, and when results changed.

## What to include

- Add each product change in the same PR, under the section of the month the PR is
  merged (`## November 2026`); create the section if it is missing. If a PR is still
  open when a new month starts, move its entry to the new month before merging.
- Product changes: new, removed, renamed or moved images; new or changed workloads,
  scoring, metrics or output format; tool version bumps; added or dropped architectures
  or GPU support; fixes that change results or make a benchmark run where it failed.
- Skip docs, glossary, CI, build and caching changes, refactors without an effect on the
  image's output, and the `internal` images.
- Describe the net change of a PR, not each of its commits. A later PR that changes
  something already merged adds its own entry in its own month; earlier entries stay
  as they are.
- Months are listed newest first. Skip a month with no entries.

## Subsections

- Use these subsections in this order and omit empty ones. Keep the headings exactly
  as written, in the plural even for a single entry:
  - `New images:`
  - `Breaking changes:`
  - `Improvements:`
  - `Fixes:`
- A breaking change is a removed or renamed image; a removed or renamed output field, a
  field with a changed type, unit or meaning, or a changed output structure; a dropped
  architecture; or a change that makes results incomparable with earlier runs (e.g. a
  new workload version or scoring method). New output fields that leave the existing
  ones unchanged are improvements, not breaking changes.

## Entries

- Start each entry with the image name in backticks and a colon; for one change that
  affects several images, list them all: `` `a`, `b`: ``.
- Describe what the image does or measures now. Write the description as a noun phrase
  or in the present tense; do not start it with a verb in any tense (`Add`, `Added`,
  `Fix`, `Fixed`, `Bump`).
  - New image: `` `benchmark-storage`: raw block device IOPS, bandwidth and latency with
    fio, for single disks and groups of identical disks ``
  - Improvement: `` `stress-ng`: stress-ng 0.20.01 (was 0.17.08), with a suite of 85
    stressors in seven categories ``
  - Fix: `` `nvbandwidth`: runs on servers where NVML failed to load ``
- Be specific: name the tool and its version, the metric, the workload or the
  architecture.
- End each entry with the PR number(s) when there is a PR, e.g. `(#12)`; no trailing
  period.

## Formatting

- Wrap entries at about 88 characters and indent continuation lines by two spaces.

## References

- GitLab, [Changelog entries](https://docs.gitlab.com/development/changelog/): describe
  the change by its effect on the reader, not its implementation.
- Google developer documentation style guide,
  [Present tense](https://developers.google.com/style/tense) and
  [Timeless documentation](https://developers.google.com/style/timeless-documentation):
  prefer the present tense; time-bound wording is fine in release notes.
