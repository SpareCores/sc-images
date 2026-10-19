# Workloads

## OpenSSL Speed

[`openssl.sh`](../openssl.sh) runs `openssl speed` once for each of the
following 11 algorithms, one algorithm at a time:

- hashes: `blake2b512`, `sha256`, `sha512`, `sha3-256`, `sha3-512`,
  `shake128`, and `shake256`
- ciphers: `aes-256-cbc`, `aria-256-cbc`, `camellia-256-cbc`, and `sm4-cbc`

Each run uses the following options:

- `-multi $(nproc)`: one forked process per logical CPU, all in parallel
- `-mr`: machine-readable output
- `-mlock`: memory locking
- `-elapsed`: wall-clock timing

The script runs `openssl speed` under `nice -n -20` and sets no duration, so
`openssl speed` measures each block size for its default duration (3 seconds in
production output). `openssl speed` also picks the block sizes itself;
production output shows 16, 64, 256, 1024, 8192, and 16384 bytes.

The `openssl` binary is a custom OpenSSL 3.4.0 build in `/opt/openssl`,
configured with many algorithms and features disabled to shorten the build. The
script puts it first on `PATH`. Ubuntu's system OpenSSL stays on the
image-wide `PATH`; the Geekbench capture server uses it to create its
certificate.

## Memory Bandwidth

[`bw_mem.sh`](../bw_mem.sh) runs lmbench's `bw_mem`, from Ubuntu's `lmbench`
package, in three of its modes:

- `rd` reads data into the processor, computing the sum of an array.
- `wr` writes data to memory, assigning a constant value to each element.
- `rdwr` reads and then writes the same location, adding the current value to
  a running sum before assigning a new constant value.

Each mode accesses every fourth word of an array of integers. The script runs
each mode over the following 11 sizes, 33 runs in sequence:
`16k 256k 1M 2M 4M 8M 16M 32M 64M 256M 512M`.

Each run is `nice -n -20 bw_mem -P $(nproc) <size> <op>`. With `-P`, each of
the `nproc` processes allocates its own buffer of the given size, and the
reported bandwidth is the total across all processes.

The [`bw_mem` manual](https://github.com/intel/lmbench/blob/master/doc/bw_mem.8)
opens by describing a buffer twice the given size that is copied from one half
to the other. That description applies to the copy modes, which this benchmark
does not run; `rd`, `wr`, and `rdwr` allocate one buffer of the given size.

## Compression

[`compress.py`](../compress.py) measures how well and how fast Python
compression libraries handle one input: `dickens`, a text file of about 10 MB
from the Silesia corpus. The script reads it into memory once.

It runs every compressor at the following levels and extra arguments:

| Compressor | Python library | Levels | Extra arguments |
| --- | --- | --- | --- |
| `brotli` | `brotli` | 0, 4, 8, 11 | None |
| `gzip` | `gzip` | 1, 5, 9 | None |
| `bzip2` | `bz2` | 1, 5, 9 | None |
| `lzma` | `lzma` | 1, 5, 9 | None |
| `zstd` | `pyzstd` | 1, 7, 14, 22 | None |
| `bzip3` | `bz3` | No level | `block_size` 1 MiB (default) and 64 MiB |
| `zpaq` | `pyzpaq` | 1, 3, 5 | None |
| `lz4` | `lz4.frame` | 1, 6, 12, 16 | `block_size` 64 KiB (default), 1 MiB, and 4 MiB |

Each combination runs with one thread and with `NUMCPUS` threads, where
`NUMCPUS` is the smaller of the physical core count and the number of CPUs the
process may run on. On a server with one core, only the single-thread case
runs.

Each combination runs in its own worker process, so the kernel's
out-of-memory killer stops only that measurement. A measurement works as
follows:

1. Compresses the input once on one thread to compute the ratio, then
   decompresses it and checks that the result matches the input byte for byte.
2. Starts one process per thread, each compressing the full input
   independently. The processes wait for a shared start signal, sent after
   1 second.
3. Takes the mean of the per-process times as the elapsed time, and computes
   throughput as the input size times the thread count, divided by the elapsed
   time.
4. Repeats steps 2 and 3 once more and keeps the faster run.
5. Measures decompression the same way.

"Threads" here means parallel, independent processes that each compress the
same data, not one multithreaded compressor working on shared data.

## Geekbench

[`geekbench.sh`](../geekbench.sh) runs Geekbench 6.7.1 once with its default
settings. Geekbench runs its own Single-Core and Multi-Core suites, each made
up of several workloads; the image configures no workload, iteration count, or
duration. The image ships both the `x86_64` build and the ARM preview build for
`aarch64`, and the script picks one by the machine architecture.

Geekbench runs with `--upload`, but the result never reaches the Geekbench
Browser. See [Geekbench](./usage.md#geekbench) for how the script captures the
upload inside the container.
