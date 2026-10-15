# References

Terms specific to this image, alphabetized. Common terms (e.g. I/O, SMT, vCPU)
are defined in the [glossary](https://docs.sparecores.com/glossary) of the
Spare Cores docs, which also lists the terms each image uses.

- **bogo ops/s** (bogus operations per second): The throughput metric of
  `stress-ng`: how many times a stressor completes its own unit of work per
  second. Each stressor defines that unit differently, so the values compare
  only within one stressor and one `stress-ng` version.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **`nice`**: A Unix command that runs a program with a changed scheduling
  priority, from -20 (highest) to 19 (lowest). Raising a process's priority
  above the default needs root privileges.
  [man page](https://man7.org/linux/man-pages/man1/nice.1.html)
- **`psutil`**: A cross-platform Python library for retrieving information on
  running processes and system utilization, such as CPU, memory, disks, and
  network. It can count both logical CPUs and physical cores.
  [psutil docs](https://psutil.readthedocs.io/)
- **SCore**: The relative performance score of Spare Cores Navigator for a
  server type, used to compare the speed of server types. It is derived from
  how the throughput of the `stress-ng` `cpu` stressor scales with the number
  of workers.
- **SIMD** (Single Instruction, Multiple Data): A form of parallel processing
  in which one CPU instruction applies the same operation to several data
  elements at once. CPUs implement it with vector instruction set extensions,
  which x86 and Arm CPUs both provide.
  [Wikipedia](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
- **`stress-ng`**: An open-source tool that loads a computer in selectable
  ways, with hundreds of stress tests (stressors) that exercise the CPU,
  memory, file systems, and kernel interfaces. With `--metrics`, it reports
  each stressor's throughput in bogo ops/s.
  [stress-ng on GitHub](https://github.com/ColinIanKing/stress-ng)
- **stressor**: One stress test of `stress-ng`, such as `cpu`, `memcpy`, or
  `fork`, selected with a command-line option of the same name and run by one
  or more worker processes.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **TLB** (Translation Lookaside Buffer): A CPU cache of recent translations
  from virtual to physical memory addresses. A TLB shootdown invalidates
  entries on other cores after a mapping changes, which needs an interrupt to
  each of them.
  [Wikipedia](https://en.wikipedia.org/wiki/Translation_lookaside_buffer)
- **VM exit**: A transition from a virtual machine's guest code to the
  hypervisor, triggered by an operation the hypervisor has to handle, such as
  some privileged instructions, I/O, or interrupts. Each exit and return costs
  CPU time, so workloads that cause many exits run slower in a VM.
  [Wikipedia](https://en.wikipedia.org/wiki/Hardware-assisted_virtualization)
