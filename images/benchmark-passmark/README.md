# benchmark-passmark

This benchmark measures CPU and memory performance with PassMark
PerformanceTest for Linux (`pt_linux`). It runs PassMark's CPU and memory test
suites unattended, with PassMark's default test duration and iteration count,
and passes PassMark's own results file through unchanged.

It reports the CPU Mark and Memory Mark composite scores, together with the
individual CPU and memory test results behind them, as a YAML (YAML Ain't
Markup Language) document.
[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes both
marks and 14 of the individual test results.

## Purpose

This benchmark compares server types on PassMark's fixed set of tests:

- **CPU tests:** integer and floating-point math, prime numbers, string
  sorting, encryption, compression, physics, extended instructions, and a
  single-threaded test.
- **Memory tests:** database operations, cached and uncached reads, writes,
  and latency.

It runs PassMark's tests without changing them, so PassMark, not Spare Cores,
computes the composite scores.

## Limitations

These results compare servers on PassMark's synthetic CPU and memory tests,
not on application throughput, and they exclude disk, network, and GPU
performance. The main limitations are the following:

- **[Scaling](#results):** Many PassMark tests do not scale well above 32
  vCPUs. Memory Mark does not scale well above 16 vCPUs: its scores plateau and
  may even degrade as the vCPU count increases. Production runs this benchmark
  only on servers with at most 32 vCPUs; the image does not enforce this limit
  and runs on any server.
- **[PassMark version](#passmark-version):** The image downloads the latest
  PassMark release when it is built, without a version pin, so a rebuild can
  change the PassMark build that produces the scores.
- **Default test settings:** This benchmark passes no test duration or
  iteration options, so PassMark's defaults apply: in production output, one
  iteration at the `Medium` test duration. It reads no environment variables
  or command-line options, and runs once, with no repetitions.
- **Number of test processes:** PassMark chooses how many test processes to
  run; this benchmark does not set it. On `aws/a1.2xlarge` (8 vCPUs), PassMark
  ran eight processes.
- **Exit status:** The exit status is that of printing the results file, not
  of PassMark. A run that writes no results file exits with an error, but a
  failed PassMark run that still writes the file exits with status 0.

## Usage

Run the image via Docker, and save the results file from stdout:

```bash
docker run --rm --privileged ghcr.io/sparecores/benchmark-passmark:main > results.yml
```

The harness runs PassMark at the highest scheduling priority
(`nice -n -20`), which needs the `CAP_SYS_NICE` capability that `--privileged`
grants. Production runs the container the same way and also sets host
networking (`--network host`).

The entrypoint runs this benchmark under `resource-tracker`, which samples
resource usage during the run. Its data is not published.

The image is built for `amd64` and `arm64`. A run took about 3 minutes on
`aws/a1.2xlarge`
([production run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/a1.2xlarge/0/passmark/meta.json)).
Production allows up to 1 hour, because PassMark can be slow on some servers.

### Settings

This benchmark reads no environment variables or command-line options. The
suite selection depends only on the available memory (see
[Workload](#workload)), and PassMark's own test settings are left at their
defaults.

### Results

This benchmark prints PassMark's YAML results file to stdout:
`results_all.yml` when it runs both suites, or `results_cpu.yml` when it runs
the CPU suite only. PassMark's interactive terminal interface, including its
terminal escape sequences, goes to stderr; redirect it to a file or
`/dev/null` if you do not need it.

The YAML document has the following top-level keys:

- `Version` holds the PassMark version, such as `Major: 11`, `Minor: 0`, and
  `Build: 1002`, and the architecture of the binary in `ptArchitecture`.
- `Results` holds the scores, along with `NumTestProcesses`, the number of
  test processes PassMark ran.
- `SystemInformation` holds the OS, kernel, processor, core counts, and memory
  that PassMark detected.
- `BaselineInfo` and `Checksum` are PassMark's own metadata for the results
  file.

The Navigator publishes the following `Results` fields, each as a
`passmark:<measurement>` benchmark. Lower is better for Memory Latency; for
the others, higher is better:

| Field | Navigator name | Unit |
| ----- | -------------- | ---- |
| `SUMM_CPU` | CPU Mark | composite score, no unit |
| `CPU_INTEGER_MATH` | CPU Integer Maths Test | Mops/s |
| `CPU_FLOATINGPOINT_MATH` | CPU Floating Point Maths Test | Mops/s |
| `CPU_PRIME` | CPU Prime Numbers Test | Mnums/s |
| `CPU_SORTING` | CPU String Sorting Test | Kstrings/s |
| `CPU_ENCRYPTION` | CPU Encryption Test | MB/s |
| `CPU_COMPRESSION` | CPU Compression Test | kB/s |
| `CPU_SINGLETHREAD` | CPU Single Threaded Test | Mops/s |
| `CPU_PHYSICS` | CPU Physics Test | fps |
| `CPU_MATRIX_MULT_SSE` | CPU Extended Instructions Test | Mmat/s |
| `SUMM_ME` | Memory Mark | composite score, no unit |
| `ME_ALLOC_S` | Database Operations | Kops/s |
| `ME_READ_S` | Memory Read Cached | MB/s |
| `ME_READ_L` | Memory Read Uncached | MB/s |
| `ME_WRITE` | Memory Write | MB/s |
| `ME_LATENCY` | Memory Latency | ns |

The other `Results` fields, such as `CPU_avx`, `m_CPU_enc_AES`, `ME_LARGE`,
and `ME_THREADED`, are in the output but not published.

For example, on `aws/a1.2xlarge` (8 vCPUs), `SUMM_CPU` is \~3170 and `SUMM_ME`
is \~1418
([production output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/a1.2xlarge/0/passmark/stdout)).

Read the following results with their scaling in mind:

- Database Operations, like Memory Mark, does not scale well above 32 vCPUs.
- CPU Single Threaded Test measures single-threaded performance only, so its
  score does not increase with the vCPU count.
- Memory Read Cached, Memory Read Uncached, Memory Write, and Memory Latency
  are largely independent of the vCPU count within an instance family; they
  mainly reflect the hardware generation and memory characteristics.

### PassMark Version

PassMark publishes no versioned download URLs, so the image downloads the
current release from `passmark.com` at build time, and the image is rebuilt
rarely. Production results come from two builds of PassMark PerformanceTest
for Linux 11.0, `1002` and `1004`; check `Version.Build` in the output to see
which build produced a result. Production does not record the PassMark
version separately.

We treat scores from these two builds as comparable. The
[PerformanceTest version history](https://www.passmark.com/products/performancetest/history.php)
lists no CPU or memory test changes in builds 1003 and 1004, but it covers
PerformanceTest for Windows; PassMark publishes no version history for the
Linux builds.

## Workload

This benchmark runs one workload: PassMark's CPU and memory suites, in
PassMark's unattended mode. [`passmark.sh`](./passmark.sh) runs the following
steps:

1. Read the available memory in MiB from `free -m`.
2. Start `pt_linux` under `nice -n -20`, with the suites chosen by available
   memory:
   - 512 MiB or more: `pt_linux -r 3`, both the CPU and memory suites.
   - Below 512 MiB: `pt_linux -r 1`, the CPU suite only. No cloud server has
     less than 512 MiB of memory, so in practice production runs both
     suites.
3. Wait for PassMark to finish, then print its results file to stdout.

The [`Dockerfile`](./Dockerfile) sets `TERM=xterm`, because PassMark expects a
terminal. PassMark writes the results file to the container's working
directory.

## References

See [References](./docs/references.md) for the terms specific to this image,
and the [contributor guide](./docs/contributors.md) for building the image.
