# Compression

## Running the Script

```shell
docker run --rm --privileged ghcr.io/sparecores/benchmark:main python /usr/local/bin/compress.py
```

In production, the median run takes \~10.7 minutes. See
[Run Duration](./usage.md#run-duration) for how it was measured, and
[Production Setup](./usage.md#production-setup) for how production runs differ
from a plain `docker run`.

## Workload

[`compress.py`](../compress.py) measures how well and how fast Python
compression libraries handle one input: `dickens`, a text file of \~10 MB
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

## How a Measurement Works

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

## Output

`compress.py` prints one indented JSON object,
`{<compressor>: {<level>: [<result>, ...]}}`. Levels are string keys, and
`bzip3`, which has no level, uses the key `"null"`. Each result has the
following fields:

- `threads`: the number of parallel processes.
- `extra_args`: the extra arguments, such as `block_size` in bytes for `bzip3`
  and `lz4`.
- `ratio`: compressed size divided by original size, as a percentage; lower is
  better.
- `compress` and `decompress`: bytes of uncompressed data per second, summed
  over all processes.

A measurement that fails, for example because its worker process was killed,
has `ratio`, `compress`, and `decompress` set to `null`, and its traceback goes
to stderr. Production runs on small servers show such failures.

## Compression Input

The compression benchmark uses only `dickens`, a text file of the Silesia
corpus, although the image ships the whole corpus. We needed one large-enough,
common text file to compare the algorithms, and there are no plans to add
other corpus files. The results show how the compressors handle text, not
other data types.

## CPU Count

The multi-thread case starts one process per physical core, not per logical
CPU (see [Workload](#workload)). On a server with SMT (Simultaneous
Multithreading), its results therefore do not use every logical CPU.

## Library Versions

The Python compression libraries are installed unpinned from PyPI (Python
Package Index) at build time, except `pyzpaq`, which is pinned to one commit.
A rebuilt image can measure newer library versions.
