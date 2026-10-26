# Workloads

## Stressor Suite

[`run_stressng_benchmarks.py`](../run_stressng_benchmarks.py) runs the
stressor suite as follows:

1. It gets the list of stressors from `stress-ng --stressors`, and keeps the
   curated stressors on that list, in alphabetical order.
2. It runs each stress-ng stressor with one worker, then with one worker per
   logical CPU available to the container, if there is more than one.
   Stressors run one at a time.
3. Each run is
   `stress-ng --metrics --metrics-brief --<stressor> <workers> -t <seconds>`,
   for 5 s, or 15 s for `stream`.
4. It reads stress-ng bogo ops/s (bogus operations per second) in real time
   from the stress-ng metrc line of the run's output.

There is no warm-up and no repetition. A run that exits with an error, takes
30 s longer than its duration, or prints no `metrc` line for the stressor is
recorded as `null`.

### Stressor Categories

The 85 stressors fall into the following seven categories:

| Category | Stressors | Exercises |
| -------- | --------- | --------- |
| `cpu_integer` | 13 | [Integer arithmetic, branching, and string handling](#integer-stressors) |
| `cpu_float_simd` | 18 | [Floating-point math and vector operations](#floating-point-and-simd-stressors) |
| `crypto` | 2 | [Password hashing and encryption](#crypto-and-compression-stressors) |
| `compression` | 2 | [Data and image compression](#crypto-and-compression-stressors) |
| `memory` | 11 | [Caches and memory bandwidth](#memory-stressors) |
| `algorithms` | 13 | [Searching, sorting, and memory allocation](#algorithm-stressors) |
| `hypervisor` | 26 | [System calls, process creation, page tables, timers, and scheduling](#hypervisor-stressors) |

The following tables summarize what each stressor does. Each stressor name
links to its entry in the
[`stress-ng` manual](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html),
which describes a newer `stress-ng` version than the 0.20.01 this image pins.

### Integer Stressors

| Stressor | What it does |
| -------- | ------------ |
| [`cpu`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#c~2) | Works through all methods of the `cpu` stressor in turn |
| [`branch`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#branch) | Branches to random locations, loading the branch predictor |
| [`atomic`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#atomic) | Runs atomic operations on integers shared among workers |
| [`funccall`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#funccall) | Calls functions with 1 to 9 arguments of various types |
| [`funcret`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#funcret) | Passes and returns small to large data types by value |
| [`hash`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#hash) | Hashes random short strings with various hash functions |
| [`rotate`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#rotate) | Rotates unsigned integers of 8 to 128 bits |
| [`str`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#str) | Runs C library string functions on random strings |
| [`wcs`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#wcs) | Runs C library wide-character string functions on random strings |
| [`nop`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#nop) | Issues no-op instructions |
| [`bitops`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#bitops) | Runs 32-bit bit-manipulation tricks |
| [`intmath`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#intmath) | Runs the basic arithmetic operations on signed integers of 8 to 128 bits |
| [`prime`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#prime) | Finds ever larger primes, so its bogo ops/s drop over the run |

### Floating-Point and SIMD Stressors

These stressors cover floating-point math and SIMD (Single Instruction,
Multiple Data) vector operations:

| Stressor | What it does |
| -------- | ------------ |
| [`fma`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#fma) | Runs multiply-add operations on floating-point arrays, as fused instructions where available |
| [`fp`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#fp) | Adds, multiplies, and divides a range of floating-point types |
| [`trig`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#trig) | Computes trigonometric functions at three precisions |
| [`matrix`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#matrix) | Runs floating-point operations on a 2D matrix, mixing memory, cache, and math work |
| [`matrix-3d`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#matrix~6) | Runs the same kind of operations on a 3D matrix |
| [`eigen`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#eigen) | Runs matrix arithmetic with the Eigen C++ library |
| [`vecmath`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#vecmath) | Runs integer math on 128-bit vectors |
| [`vecfp`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#vecfp) | Runs floating-point arithmetic on vectors |
| [`vecshuf`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#vecshuf) | Shuffles and repacks the elements of vectors |
| [`vecwide`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#vecwide) | Runs 8-bit math on vectors of 4 to 2048 bytes, showing how well the compiler vectorizes it |
| [`monte-carlo`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#monte) | Estimates pi and e with Monte Carlo experiments |
| [`mpfr`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#mpfr) | Runs multi-precision floating-point math with a GNU library |
| [`expmath`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#expmath) | Computes exponential functions |
| [`logmath`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#logmath) | Computes logarithms |
| [`powmath`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#powmath) | Computes power functions |
| [`hyperbolic`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#hyperbolic) | Computes hyperbolic functions at three precisions |
| [`fractal`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#fractal) | Computes a Mandelbrot set in double precision |
| [`veccmp`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#veccmp) | Compares integers in 128-bit vectors |

### Crypto and Compression Stressors

| Stressor | What it does |
| -------- | ------------ |
| [`crypt`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#crypt) | Hashes a random password with many `crypt(3)` methods, such as bcrypt and yescrypt |
| [`ipsec-mb`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#ipsec) | Runs cryptographic routines with Intel's multi-buffer crypto library; supported only on modern Intel x86_64 CPUs |
| [`zlib`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#zlib) | Compresses random data in one process and decompresses it in another |
| [`jpeg`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#jpeg) | Compresses a generated image with lossy image compression |

### Memory Stressors

| Stressor | What it does |
| -------- | ------------ |
| [`cache`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#C) | Reads and writes memory at random, widespread addresses to thrash the CPU cache |
| [`cacheline`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#cacheline) | Has several processes read and write the bytes of one shared cache line |
| [`l1cache`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#l1cache) | Reads and writes a buffer twice the size of the L1 cache, forcing evictions |
| [`memcpy`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#memcpy) | Copies and moves memory at three alignments |
| [`memrate`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#memrate) | Reads and writes a buffer with 8- to 1024-bit accesses at full speed |
| [`memthrash`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#memthrash) | Thrashes a 16 MB buffer with at least one thread per CPU |
| [`stream`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#stream) | Measures memory bandwidth on buffers the size of the L3 cache, loosely based on the STREAM benchmark |
| [`vm`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#m) | Repeatedly maps, writes, and unmaps memory |
| [`cachehammer`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#cachehammer) | Mixes random reads, writes, prefetches, and cache flushes |
| [`ptr-chase`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#ptr) | Follows random pointers across thousands of memory pages |
| [`spinmem`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#spinmem) | Keeps two processes in sync through shared memory with busy spin loops |

### Algorithm Stressors

| Stressor | What it does |
| -------- | ------------ |
| [`bsearch`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#bsearch) | Binary-searches a sorted array of integers |
| [`hsearch`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#hsearch) | Searches a hash table that is 80% full |
| [`qsort`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#Q,) | Sorts integers with quicksort |
| [`heapsort`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#heapsort) | Sorts integers with heapsort |
| [`mergesort`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#mergesort) | Sorts integers with merge sort |
| [`bitonicsort`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#bitonicsort) | Sorts integers with bitonic sort |
| [`skiplist`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#skiplist) | Stores and searches integers in a skip list |
| [`tree`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#tree) | Adds, finds, and removes integers in several kinds of trees |
| [`malloc`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#malloc) | Allocates, resizes, and frees memory at random |
| [`mcontend`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#mcontend) | Has several threads contend for one physical memory page while moving between CPUs |
| [`insertionsort`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#insertionsort) | Sorts integers with insertion sort |
| [`bubblesort`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#bubblesort~2) | Sorts integers with bubble sort |
| [`fibsearch`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#fibsearch) | Searches a sorted array of integers with Fibonacci search |

### Hypervisor Stressors

The `hypervisor` stressors are the ones whose throughput on a VM is dominated
by hypervisor overhead, such as the cost of each VM exit, the extra
page-table lookups of nested paging, and the emulation of timers and traps:

| Stressor | What it does |
| -------- | ------------ |
| [`context`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#context) | Switches between threads in user space |
| [`switch`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#s) | Forces context switches between two processes that block each other |
| [`clone`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#clone) | Rapidly creates and reaps clones |
| [`fork`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#f) | Forks children that exit immediately |
| [`vfork`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#vfork~2) | Vforks children that exit immediately |
| [`fault`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#fault) | Generates minor and major page faults |
| [`tlb-shootdown`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#tlb~6) | Changes shared memory mappings, forcing TLB (Translation Lookaside Buffer) flushes on other CPUs |
| [`mprotect`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#mprotect) | Changes the page protection of shared memory, causing TLB flushes |
| [`munmap`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#munmap) | Unmaps the shared memory of child processes page by page |
| [`null`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#null) | Calls I/O (Input/Output) system calls on `/dev/null` |
| [`zero`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#zero) | Calls I/O system calls on `/dev/zero` |
| [`getrandom`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#getrandom) | Reads random bytes from the kernel |
| [`pipe`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#p) | Writes and reads large blocks through a pipe between two processes |
| [`eventfd`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#eventfd) | Passes short event messages between two processes through the kernel |
| [`hrtimers`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#hrtimers) | Runs many high-resolution timers with short, random intervals |
| [`clock`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#clock) | Reads clocks and waits on timers, causing frequent context switches |
| [`yield`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#y) | Has at least two processes per CPU yield the CPU to each other |
| [`resched`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#resched) | Moves processes through priority levels and scheduling policies |
| [`vdso`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#vdso) | Calls the kernel functions mapped into each process that avoid a full system call |
| [`usersyscall`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#usersyscall) | Handles system calls in user space with a signal handler |
| [`priv-instr`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#priv) | Executes privileged instructions that the kernel traps |
| [`tsc`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#tsc) | Reads the CPU's time-stamp counter |
| [`x86syscall`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#x86syscall) | Makes system calls with the x86_64 `syscall` instruction |
| [`cpu-sched`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#cpu~9) | Moves processes between CPUs and scheduling policies |
| [`timermix`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#timermix) | Generates timer signals at a high rate |
| [`mtx`](https://manpages.debian.org/testing/stress-ng/stress-ng.1.en.html#mtx) | Locks and unlocks mutexes |

`tsc`, `usersyscall`, and `x86syscall` return `null` on Arm servers. See
[Missing Results](./limitations.md#missing-results).

## CPU Scaling Loop

The CPU scaling loop runs the `cpu` stressor with the stress-ng div16 method
(`--cpu-method div16`) for 10 s at each worker count that
[`count.awk`](../count.awk) prints, and prints bogo ops/s for each. The loop itself is a shell command in the Navigator's
Inspector; [Production Setup](./usage.md#production-setup) shows it in full.

`count.awk` takes the CPU count N and prints the following worker counts:

- every count from 1 to N, if N is 32 or less
- otherwise, 1, N, and the even counts below N that are divisible by N / 32
  (rounded down)

The Inspector uses this loop to determine a server type's SCore (Spare Cores
score): its raw multi-core CPU speed, for comparing the speed of server types.
The `div16` method was chosen because it scaled as expected across x86_64 and
Arm servers, with and without SMT (Simultaneous Multithreading). See
[Design History](../README.md#design-history).
