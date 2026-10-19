# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### Disk and Network Performance

The scripts measure CPU and memory, not disk or network performance. The
compression benchmark reads its input into memory before it starts timing.

### Application Performance

Each script runs a fixed, synthetic workload: one hash or cipher at a time,
one memory access pattern, one compressor on one file, or Geekbench's fixed
suite. The results compare server types on these workloads; they do not
predict the throughput of a specific application.

## Scope Decisions

### Compression Input

The compression benchmark uses only `dickens`, a text file of the Silesia
corpus, although the image ships the whole corpus. We needed one large-enough,
common text file to compare the algorithms, and there are no plans to add
other corpus files. The results show how the compressors handle text, not
other data types.

### Geekbench Server Range

Production runs Geekbench only on servers with at most 32 vCPUs, because (based
on our empirical experience after running benchmarks on thousands of server
SKUs) Geekbench does not scale beyond 32 cores, and with at least 2.1 GiB of
memory. The image enforces neither limit, so `geekbench.sh` also runs on larger
and smaller servers.

### Tool Versions

The image pins some tool versions but not all of them:

- **OpenSSL:** a custom 3.4.0 build with many algorithms and features
  disabled. Only the 11 algorithms in [OpenSSL Speed](./workloads.md#openssl-speed)
  are measured.
- **Geekbench:** pinned to 6.7.1.
- **`bw_mem`:** from Ubuntu 24.04's `lmbench` package, whose version may differ
  from the upstream LMbench source.
- **Python compression libraries:** installed unpinned from PyPI (Python
  Package Index) at build time, except `pyzpaq`, which is pinned to one commit.
  A rebuilt image can measure newer library versions.

## Design Constraints

### CPU Counts

The scripts count CPUs in two different ways:

- OpenSSL speed and `bw_mem` start one process per logical CPU (`nproc`).
- The compression benchmark starts at most one process per physical core, and
  no more than the CPUs the process may run on.

On a server with SMT (Simultaneous Multithreading), the compression benchmark
therefore runs fewer processes than the other two, and its multi-thread results
do not use every logical CPU.
