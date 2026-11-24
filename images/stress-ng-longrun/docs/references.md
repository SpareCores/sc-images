# Glossary

This page lists terms specific to this image in alphabetical order. The
`stress-ng` manual defines terms it shares with this image, such as
stress-ng bogo ops/s, in its
[Glossary](../../stress-ng/docs/references.md). Common terms (e.g.
CPU-credit throttling, shared-core) are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary), which also
lists the terms each image uses.

- **stress-ng bogo ops/s (real time)**: The stress-ng bogo ops/s for a run,
  measured over wall-clock (real) time and summed across all workers. This
  differs from the rate measured over the workers' CPU time. It is the
  `bogo-ops-per-second-real-time` field of the stress-ng YAML report and the
  headline result of this benchmark.
- **stress-ng YAML report**: A file in which `stress-ng` writes a run's results
  in YAML. The `-Y <file>` or `--yaml <file>` option specifies the file. The
  report has two sections. The `system-info` section describes the run and the
  system, including the `stress-ng` version, the time, and the CPU count. The
  `metrics` section holds the bogo ops count, the bogo ops/s, and the times of
  each stressor.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
