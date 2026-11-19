# Glossary

Terms specific to this image, alphabetized. Common terms (e.g. `lscpu`,
`lsblk`) are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary), which also
lists the terms each image uses.

- **hwloc** (Portable Hardware Locality): An open-source library with
  command-line tools that detects a machine's hardware topology, including
  processor packages (sockets), NUMA (Non-Uniform Memory Access) nodes,
  caches, cores, and hardware threads, as well as attached devices. Parallel
  programs use it to place their threads and memory close to each other.
  [hwloc](https://www.open-mpi.org/projects/hwloc/)
- **LIKWID** (Like I Knew What I'm Doing): A set of command-line tools for
  performance-oriented programmers on Linux, developed at the Erlangen National
  High Performance Computing Center. Its tools show the topology of a machine,
  pin threads to cores, read hardware performance counters, and run
  micro-benchmarks. [LIKWID on GitHub](https://github.com/RRZE-HPC/likwid)
- **`lshw`** (Hardware Lister): A Linux command that reports the hardware
  configuration of a machine, such as its CPUs, memory, firmware, buses, disks,
  and network adapters, as a tree of nodes. It reads sources such as `/proc`,
  `/sys`, and the DMI (Desktop Management Interface) tables, and prints text,
  XML, or JSON. It reports more details when run as root than when run as a
  normal user.
  [Ubuntu man page](https://manpages.ubuntu.com/manpages/noble/man1/lshw.1.html)
- **`lstopo`**: The hwloc command that shows the topology of a machine, as a
  drawing or as text. It can also export the topology to a file, such as an
  XML file that other hwloc programs can load or a vector image.
  [Ubuntu man page](https://manpages.ubuntu.com/manpages/noble/man1/lstopo.1.html)
