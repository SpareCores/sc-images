# Workloads

## Stressor Suite

[`run_stressng_benchmarks.py`](../run_stressng_benchmarks.py) runs the
stressor suite as follows:

1. It gets the list of stressors from `stress-ng --stressors`, and keeps the
   curated stressors on that list, in alphabetical order.
2. It runs each stressor with one worker, then with one worker per logical CPU
   available to the container, if there is more than one. Stressors run one
   at a time.
3. Each run is `stress-ng --metrics --metrics-brief --<stressor> <workers>
   -t <seconds>`, for 5 s, or 15 s for `stream`.
4. It reads bogo ops/s (bogus operations per second) in real time from the
   `metrc` line of the run's output.

There is no warm-up and no repetition. A run that exits with an error, takes
30 s longer than its duration, or prints no `metrc` line for the stressor is
recorded as `null`.

### Stressor Categories

The 85 stressors fall into the following seven categories:

| Category | Stressors |
| -------- | --------- |
| `cpu_integer` (13) | `cpu`, `branch`, `atomic`, `funccall`, `funcret`, `hash`, `rotate`, `str`, `wcs`, `nop`, `bitops`, `intmath`, `prime` |
| `cpu_float_simd` (18) | `fma`, `fp`, `trig`, `matrix`, `matrix-3d`, `eigen`, `vecmath`, `vecfp`, `vecshuf`, `vecwide`, `monte-carlo`, `mpfr`, `expmath`, `logmath`, `powmath`, `hyperbolic`, `fractal`, `veccmp` |
| `crypto` (2) | `crypt`, `ipsec-mb` |
| `compression` (2) | `zlib`, `jpeg` |
| `memory` (11) | `cache`, `cacheline`, `l1cache`, `memcpy`, `memrate`, `memthrash`, `stream`, `vm`, `cachehammer`, `ptr-chase`, `spinmem` |
| `algorithms` (13) | `bsearch`, `hsearch`, `qsort`, `heapsort`, `mergesort`, `bitonicsort`, `skiplist`, `tree`, `malloc`, `mcontend`, `insertionsort`, `bubblesort`, `fibsearch` |
| `hypervisor` (26) | See [Hypervisor Stressors](#hypervisor-stressors) |

`cpu_float_simd` covers floating-point and SIMD (Single Instruction, Multiple
Data) vector work.

### Hypervisor Stressors

The `hypervisor` stressors are the ones whose throughput on a VM is dominated
by hypervisor overhead, such as the cost of each VM exit, the extra
page-table lookups of nested paging, and the emulation of timers and traps.
They fall into the following groups:

- context switching and process creation: `context`, `switch`, `clone`,
  `fork`, `vfork`
- page faults and TLB (Translation Lookaside Buffer) invalidation: `fault`,
  `tlb-shootdown`, `mprotect`, `munmap`
- minimal kernel I/O (Input/Output), measuring the system-call round trip:
  `null`, `zero`, `getrandom`
- inter-process communication through the kernel: `pipe`, `eventfd`
- timers: `hrtimers`, `clock`
- scheduling: `yield`, `resched`
- privileged and architecture-specific paths: `vdso`, `usersyscall`,
  `priv-instr`, `tsc`, `x86syscall`, `cpu-sched`, `timermix`, `mtx`

Some of the last group run only on x86 CPUs. See
[Missing Results](./limitations.md#missing-results).

## CPU Scaling Loop

The CPU scaling loop runs the `cpu` stressor with `--cpu-method div16` for
10 s at each worker count that [`count.awk`](../count.awk) prints, and prints
bogo ops/s for each. The loop itself is a shell command in the Navigator's
Inspector; [Production Setup](./usage.md#production-setup) shows it in full.

`count.awk` takes the CPU count N and prints the following worker counts:

- every count from 1 to N, if N is 32 or less
- otherwise, 1, N, and the even counts below N that are divisible by N / 32
  (rounded down)

The Inspector uses this loop to determine a server type's SCore, a relative
performance score for comparing the speed of server types. The `div16`
method was chosen because it scaled as expected across x86_64 and Arm servers,
with and without SMT (Simultaneous Multithreading). See
[Design History](../README.md#design-history).
