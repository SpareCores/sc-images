# benchmark

This image bundles four independent benchmarking tools that measure general
CPU and memory performance: OpenSSL's `openssl speed`, LMbench's `bw_mem`
(memory bandwidth tool), a set of Python compression libraries, and
Geekbench 6. Each tool has its own script, and each run of the image runs one
script.

The tools report the following:

- **OpenSSL speed:** Throughput in bytes per second per algorithm and block
  size.
- **`bw_mem`:** Memory bandwidth in MB/s per operation and buffer size.
- **Compression:** Ratio and throughput per compressor, level, and thread
  count.
- **Geekbench:** Single-Core and Multi-Core scores.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for more than 5,000 cloud server types. This
image provides several of the Navigator's general-purpose measurements:
OpenSSL speed, memory bandwidth, compression algorithms, and Geekbench scores.

Each script is a separate production task that the Navigator's Inspector runs
on its own, so a result from one script does not depend on the others.

## Limitations

These results compare CPU and memory performance on fixed, synthetic workloads,
not disk, network, or application throughput. The main limitations are the
following:

- **Compression input:** The compression benchmark uses one text file of the
  Silesia corpus, not the whole corpus, so its results describe text only. See
  [Compression Input](./docs/limitations.md#compression-input).
- **Parallelism:** OpenSSL speed and `bw_mem` run one process per logical CPU,
  while the compression benchmark runs at most one process per physical core.
  See [CPU Counts](./docs/limitations.md#cpu-counts).
- **Geekbench server range:** Production runs Geekbench only on servers with
  up to 32 vCPUs and at least 2.1 GiB of memory. The image enforces neither. See
  [Geekbench Server Range](./docs/limitations.md#geekbench-server-range).
- **Tool versions:** The image builds OpenSSL 3.4.0 with many algorithms
  disabled and installs the Python compression libraries unpinned, so library
  versions can change between builds. See
  [Tool Versions](./docs/limitations.md#tool-versions).

## Usage

Run the image via Docker with the script to run as the command; the image has
no default command:

```shell
docker run --rm --privileged ghcr.io/sparecores/benchmark:main openssl.sh
docker run --rm --privileged ghcr.io/sparecores/benchmark:main bw_mem.sh
docker run --rm --privileged ghcr.io/sparecores/benchmark:main python /usr/local/bin/compress.py
docker run --rm --privileged ghcr.io/sparecores/benchmark:main geekbench.sh
```

`openssl.sh` and `bw_mem.sh` run their tools under `nice -n -20`, which needs
`--privileged`. Production also runs `compress.py` and `geekbench.sh` under
`nice -n -20`, from its own command line. See
[Production Setup](./docs/usage.md#production-setup) before you compare your
results with the Navigator's.

`geekbench.sh` does not upload results to the Geekbench Browser: it captures
Geekbench's upload on a local HTTPS (Hypertext Transfer Protocol Secure) server
inside the container. See [Geekbench](./docs/usage.md#geekbench) for how the
capture works.

In production, the median run takes the following time:

- `openssl.sh`: \~4 minutes
- `bw_mem.sh`: \~4 minutes
- `geekbench.sh`: \~6 minutes
- `compress.py`: \~11 minutes

See [Run Duration](./docs/usage.md#run-duration) for the measured medians.

### Settings

The scripts take no arguments, and all benchmark settings are fixed in the
scripts. The only benchmark setting is `BENCHMARK_SECRETS_PASSPHRASE`, which
is optional: it decrypts a Geekbench license shipped in the image, and
Geekbench runs without a license if it is unset. See the
[environment-variable reference](./docs/usage.md#environment-variables) for
all variables.

### Results

Each script prints its results to stdout in its own format:

- `openssl.sh` prints raw `openssl speed -mr` output for each algorithm, after
  a delimiter line `ALGO: <args> ---...`.
- `bw_mem.sh` prints one line per successful run, `<op> <size> <bandwidth>`,
  with the buffer size in MB and the bandwidth in MB/s summed over all
  processes.
- `compress.py` prints one JSON object with `ratio`, `compress`, and
  `decompress` per compressor, level, thread count, and extra arguments.
- `geekbench.sh` prints Geekbench's console report to stdout, and the full
  result document as JSON to stderr.

See [Outputs](./docs/usage.md#outputs) for the fields and units of each
format.

## Workloads

The image runs the following benchmarks, one per script:

- [OpenSSL speed](./docs/workloads.md#openssl-speed) (`openssl.sh`) measures
  the throughput of 11 hash and cipher algorithms with one process per logical
  CPU.
- [Memory bandwidth](./docs/workloads.md#memory-bandwidth) (`bw_mem.sh`)
  measures read, write, and read-write bandwidth over 11 buffer sizes from
  `16k` to `512M`, with one process per logical CPU.
- [Compression](./docs/workloads.md#compression) (`compress.py`) measures
  eight compressors at several levels on one thread and with at most one
  process per physical core.
- [Geekbench](./docs/workloads.md#geekbench) (`geekbench.sh`) runs Geekbench
  6.7.1's own Single-Core and Multi-Core suites.

See [Workloads](./docs/workloads.md) for the full parameter matrix of each
benchmark.

## References

See the full list of [references](./docs/references.md) used in this
documentation.
