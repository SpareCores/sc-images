# References

Terms specific to this image, alphabetized. The terms of `stress-ng` that the
stress-ng manual also uses, such as stress-ng bogo ops/s, are defined in its
[references](../../stress-ng/docs/references.md). Common terms (e.g.
CPU-credit throttling, shared-core) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs, which
also lists the terms each image uses.

- **stress-ng bogo ops/s (real time)**: The stress-ng bogo ops/s of a run
  over its wall-clock (real) time, summed across all workers, as opposed to
  the CPU time of the workers. It is the `bogo-ops-per-second-real-time` field
  of the stress-ng YAML report and the headline result of this benchmark.
- **stress-ng YAML report**: A file in which `stress-ng` writes the results of
  a run in YAML, selected with the `-Y <file>` or `--yaml <file>` option. It
  holds two sections. The `system-info` section describes the run and the
  system, such as the `stress-ng` version, the time, and the CPU count. The
  `metrics` section holds the bogo ops count, the bogo ops/s, and the times of
  each stressor.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
