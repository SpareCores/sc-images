# benchmark-passmark

This benchmark measures CPU and memory performance with PassMark's
`pt_linux` (PerformanceTest for Linux). It runs PassMark's CPU and memory
test suites unattended, with PassMark's default test duration and iteration
count, and passes PassMark's own results file through unchanged.

It reports the CPU Mark and Memory Mark composite scores, together with the
individual CPU and memory test results behind them, as a YAML document.
[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
both marks and 14 of the individual test results; its
[Benchmark Workloads](https://sparecores.com/navigator/benchmark-workloads)
page describes each test and shows its score distribution.

## Purpose

This benchmark compares server types on the following PassMark tests, which
the Navigator publishes:

- **[CPU tests](https://sparecores.com/servers?benchmark=eyJpZCI6InBhc3NtYXJrOmNwdV9tYXJrIiwiY29uZmlnIjoie30ifQ%3D%3D&order_by=selected_benchmark_score&order_dir=desc):**
  integer and floating-point math, prime numbers, string sorting, encryption,
  compression, physics, extended instructions, and a single-threaded test.
- **[Memory tests](https://sparecores.com/servers?benchmark=eyJpZCI6InBhc3NtYXJrOm1lbW9yeV9tYXJrIiwiY29uZmlnIjoie30ifQ%3D%3D&order_by=selected_benchmark_score&order_dir=desc):**
  database operations, cached and uncached reads, writes, and latency.

The Navigator already publishes other CPU and memory benchmarks. It adds
PassMark because a run is short (2 to 3 minutes), the self-hosting community
uses PassMark widely, and some of its tests, such as Database Operations,
cover workloads that the other benchmarks do not.

It runs PassMark's tests without changing them, so PassMark, not Spare Cores,
computes the composite scores.

## Limitations

These results compare servers on PassMark's synthetic CPU and memory tests,
not on application throughput, and they exclude
[disk](../benchmark-storage/README.md), network, and GPU performance. The main
limitations are the following:

- **Scaling:** Many PassMark tests, including Database Operations, do not
  scale well above 32 vCPUs. Memory Mark does not scale well above 16 vCPUs:
  its scores plateau and may even degrade as the vCPU count increases.
  Production runs this benchmark only on servers with at most 32 vCPUs; the
  image does not enforce this limit and runs on any server.
- **[PassMark Version](#passmark-version):** The image downloads the latest
  PassMark release when it is built, without a version pin, so a rebuild can
  change the PassMark build that produces the scores.
- **[Default Test Settings](#settings):** This benchmark passes no test
  duration or iteration options, so PassMark's defaults apply: in production
  output, one iteration at the `Medium` test duration. This benchmark runs it
  once, with no repetitions.
- **[Number of Test Processes](./docs/usage.md#results):** PassMark chooses
  how many test processes to run; this benchmark does not set it. On
  `aws/a1.2xlarge` (8 vCPUs), PassMark ran eight processes.
- **Exit Status:** The exit status is that of printing the results file, not
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

This benchmark prints PassMark's YAML results file to stdout, with the CPU
Mark in `Results.SUMM_CPU` and the Memory Mark in `Results.SUMM_ME`.
See [Results](./docs/usage.md#results) for the file's top-level keys and the
individual test results the Navigator publishes, with their units.

### PassMark Version

PassMark publishes no versioned download URLs, so the image downloads the
current release from `passmark.com` at build time. Production results come
from two builds of PassMark PerformanceTest for Linux 11.0, `1002` and
`1004`; check `Version.Build` in the output to see which build produced a
result.

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

See [References](./docs/references.md) for the terms specific to this image.
See the [contributor guide](./docs/contributors.md) for building the image.
