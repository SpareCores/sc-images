# References

Terms specific to this image, alphabetized. Common terms (e.g. `nice`, SMT,
vCPU) are defined in the [glossary](https://docs.sparecores.com/glossary) of
the Spare Cores docs, which also lists the terms each image uses.

- **bogo ops/s** (bogus operations per second): The throughput metric of
  `stress-ng`: how many times a stressor completes its own unit of work per
  second. Each stressor defines that unit differently, so the values compare
  only within one stressor and one `stress-ng` version.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **CPU-credit throttling**: A limit on burstable cloud server types, which
  earn CPU credits while they run below a baseline CPU usage and spend them to
  run above it. When the credits run out, the provider throttles the server to
  its baseline, so sustained load runs slower than a short burst.
  [AWS burstable instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances.html)
- **`div16`**: A method of the `stress-ng` `cpu` stressor, selected with
  `--cpu-method div16`, in which each operation runs 50,000 16-bit unsigned
  integer divisions. Integer division takes many CPU cycles and depends little
  on memory, so it loads the core's arithmetic units directly.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **`metrc`**: The tag of the lines in which `stress-ng --metrics` prints each
  stressor's results when the run ends: the bogo ops count, the real, user,
  and system time, and the bogo ops/s, among other columns.
  [stress-ng man page](https://manpages.ubuntu.com/manpages/noble/man1/stress-ng.1.html)
- **nested paging**: Hardware support for memory virtualization, in which the
  CPU translates a guest's virtual addresses through the guest's page tables
  and then through a second set of page tables kept by the hypervisor. It
  removes the need for the hypervisor to maintain shadow page tables, but a
  TLB miss then needs more memory lookups. Intel calls it Extended Page Tables
  and AMD Rapid Virtualization Indexing.
  [Wikipedia](https://en.wikipedia.org/wiki/Second_Level_Address_Translation)
- **`psutil`**: A cross-platform Python library for retrieving information on
  running processes and system utilization, such as CPU, memory, disks, and
  network. It can count both logical CPUs and physical cores.
  [psutil docs](https://psutil.readthedocs.io/)
- **SCore**: The raw multi-core CPU speed of a server type in Spare Cores
  Navigator, used to compare the speed of server types. It is measured with
  the `cpu` stressor of `stress-ng`.
- **shared-core**: Describes a cloud server type whose vCPUs share physical
  cores with other servers, often with a guaranteed baseline of only a
  fraction of a core. Such server types are cheaper, but their CPU performance
  can vary with the load of neighboring servers or be limited by CPU credits.
  [Google Cloud shared-core machine types](https://cloud.google.com/compute/docs/general-purpose-machines#sharedcore)
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
