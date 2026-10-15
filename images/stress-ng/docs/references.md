# References

Terms specific to this image, alphabetized. Common terms (e.g. SCore, SIMD,
vCPU) are defined in the [glossary](https://docs.sparecores.com/glossary) of
the Spare Cores docs, which also lists the terms each image uses.

- **`stress-ng`**: An open-source tool that loads a computer in selectable
  ways, with hundreds of stress tests (stressors) that exercise the CPU,
  memory, file systems, and kernel interfaces. With `--metrics`, it reports
  each stressor's throughput in bogo ops/s.
  [stress-ng on GitHub](https://github.com/ColinIanKing/stress-ng)
- **stress-ng bogo ops/s** (bogus operations per second): The throughput
  metric of `stress-ng`: how many times a stressor completes its own unit of
  work per second. Each stressor defines that unit differently, so the values
  compare only within one stressor and one `stress-ng` version.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **stress-ng div16 method**: A method of the `stress-ng` `cpu` stressor,
  selected with `--cpu-method div16`, in which each operation runs 50,000
  16-bit unsigned integer divisions. Integer division takes many CPU cycles
  and depends little on memory, so it loads the core's arithmetic units
  directly.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **stress-ng metrc line**: A line in which `stress-ng --metrics` prints a
  stressor's results when the run ends, tagged `metrc`: the bogo ops count,
  the real, user, and system time, and the bogo ops/s, among other columns.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **stress-ng stressor**: One stress test of `stress-ng`, such as `cpu`,
  `memcpy`, or `fork`, selected with a command-line option of the same name
  and run by one or more worker processes.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
