# benchmark-web

This benchmark measures how fast one `binserve` process serves static files
over HTTP, with `wrk` as the load generator. Server and client run in the same
container and talk over loopback. It sweeps five file sizes, from 1 KiB to
512 KiB, and the number of `wrk` threads and connections, which scale with the
number of CPUs. Each of the 69 configurations runs once for 10 seconds.

It reports the requests per second and the average latency of each
configuration, together with the CPU time the server and the client used.
[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
requests per second, throughput in bytes per second, and latency for each file
size and number of connections per vCPU. It also publishes extrapolated
requests per second and throughput, which estimate how fast `binserve` would
serve if `wrk` did not share its CPUs; see [Results](#results).

## Purpose

This benchmark shows how fast static files can be served over HTTP on a server
type, and how throughput and latency change with the file size and the number
of concurrent connections.

We serve the files with `binserve` because it is a single binary that needs no
tuning. Widely used web servers, such as nginx, have many settings, some of
them best set per server type, which would make a like-for-like comparison
across server types hard.

The web server and the load generator run side by side in the same
container: `wrk` sends its requests to `binserve` at `localhost:8080`, over
the loopback interface. Loopback is a virtual network interface that the
operating system provides for traffic between programs on the same machine,
so the requests never reach a network card or leave the server. This keeps
provisioning simple and benchmarking costs lower, because a run needs only
the server under test and no separate client machine, and it keeps network
latency out of the results. The trade-off is that `wrk` uses CPU time that
`binserve` could otherwise use. This benchmark records the CPU time of both,
so the Navigator can estimate how fast `binserve` would serve with the CPUs
to itself; see [Results](#results).

## Limitations

These results compare static file serving by one `binserve` process on one
host, not the capacity of a tuned production web server. The main limitations
are the following:

- **[Shared CPUs](#results):** `binserve` and `wrk` share the same CPUs, and
  `wrk` runs up to four threads per CPU, so `rps` is not the maximum a
  dedicated server would reach. The Navigator also publishes CPU-time-adjusted
  scores.
- **Large Servers:** This benchmark does not scale well on servers with 100 or
  more vCPUs because of a listener bottleneck in the single `binserve`
  process.
- **[Static Files Over HTTP](#workload):** This benchmark serves five static
  files, filled with zero bytes, from `binserve`'s in-memory file cache over
  plain HTTP. It does not measure dynamic content or HTTPS.
- **[Loopback Only](#purpose):** `wrk` and `binserve` exchange requests over
  the loopback interface of the same machine, so no traffic crosses a network
  card or a real network. Results do not include network throughput or
  latency, and a server type's network performance does not affect them.
- **Untuned Server:** `binserve` runs with the settings in
  [`binserve.json`](./binserve/binserve.json) and its defaults otherwise.
  Results do not show how a tuned web server, such as nginx, would perform on
  the same server type.
- **Single, Short Runs:** Each configuration runs once for 10 seconds, with no
  warmup and no repetitions.
- **CPU Count:** The thread and connection counts follow the CPUs the
  container may run on. Docker's `--cpuset-cpus` changes them, but `--cpus`
  does not: with `--cpus`, this benchmark still uses the host's CPU count.
- **Error Checks:** `failed` counts only responses with a status other than
  2xx or 3xx, and `wrk`'s connect, read, write, and timeout errors are not
  recorded. If `wrk`'s output cannot be parsed, this benchmark prints the error
  to stderr, writes the row with an `rps` of 0, and still exits with code 0, as
  if the run had succeeded. The Navigator publishes such a row only if every
  thread count in its group returned zero.

## Usage

Run the image via Docker:

```bash
docker run --rm --privileged ghcr.io/sparecores/benchmark-web:main
```

The harness runs `binserve` and `wrk` at the highest scheduling priority
(`nice -n -20`), which needs the `CAP_SYS_NICE` capability that `--privileged`
grants. Without it, they run at normal priority. The harness also raises its
open-file limit to 1,048,576 before starting them.

Production runs the container the same way and also sets host networking
(`--network host`). With host networking, `binserve` listens on port 8080 of
the host's loopback interface, so that port must be free. If it is not,
`binserve` cannot listen on it, and this benchmark usually stops with
`Couldn't start server` and exit code 1. Without host networking, the traffic
stays on loopback inside the container.

The container entrypoint runs this benchmark under `resource-tracker`, which
samples resource usage during the run. Its data is not published.

Measurement takes about 12 minutes: 69 configurations of 10 seconds each, plus
server startup. A production run on an `aws/m5.large` took 11 minutes 42 seconds
([run metadata](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/static_web/meta.json)).

The run metadata also records the tool versions in its `version` field, as one
string, for example `binserve 0.2.1, wrk 4.2.0`. The image builds `binserve`
from commit `1e32f94` and `wrk` from version 4.2.0.

### Settings

This benchmark reads no environment variables or command-line options. The
file sizes, thread and connection counts, and run time are fixed in
[`benchmark.py`](./benchmark.py); see [Workload](#workload). To change the
thread and connection counts, limit the container's CPUs with
`--cpuset-cpus`.

### Results

This benchmark prints a comma-separated table to stdout: a header row with
quoted column names, then one row per configuration.

```text
"size","threads","connections","rps","latency","failed","server_usr","server_sys","client_usr","client_sys"
```

The columns mean the following:

- `size` is the label of the served file in KiB, such as `"1k"`.
- `threads` and `connections` are the `wrk` thread count and the number of
  open connections.
- `rps` is the requests per second that `wrk` reports, as an integer.
- `latency` is the average latency from `wrk`'s thread statistics in seconds.
- `failed` is the number of responses with a status other than 2xx or 3xx.
- `server_usr` and `server_sys` are the user and system CPU seconds that
  `binserve` used during the configuration.
- `client_usr` and `client_sys` are the user and system CPU seconds that
  `wrk` used during the configuration.

For example, on an `aws/m5.large` (2 vCPUs), serving the `1k` file with two
threads, `rps` rises from 43,808 with two connections to 64,606 with 32
connections, while `latency` rises from 0.05 ms to 0.55 ms
([production output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/static_web/stdout)).

The Navigator groups the rows by file size and by connections per vCPU
(`connections` divided by the server's vCPU count), so that servers with
different vCPU counts can be compared. In each group, it keeps the row with the
highest `rps`, whatever its thread count, and publishes the following five
scores from it:

- **`static_web:rps`:** `rps` in requests per second.
- **`static_web:rps-extrapolated`:** `rps` multiplied by the ratio of all CPU
  time (server and client) to the server's CPU time in requests per second.
- **`static_web:throughput` and `static_web:throughput-extrapolated`:** the
  two scores above multiplied by the file size in bytes per second.
- **`static_web:latency`:** `latency` of the same row in seconds; lower is
  better.

In the `aws/m5.large` example, for the `1k` file at 16 connections per vCPU,
the row with two threads (64,606 requests/s) is kept over those with
four and eight threads. Note that the published latency is that of the
highest-`rps` row, not the lowest latency in the group. The Navigator's Web UI
shows these scores with charts on each server type's page, for example
[`aws/g5g.xlarge`](https://sparecores.com/server/aws/g5g.xlarge).

## Workload

This benchmark runs one workload: a sweep of file sizes, `wrk` threads, and
connections against one `binserve` process. [`benchmark.py`](./benchmark.py)
first runs the following steps:

1. Create five files in `/usr/local/binserve/data/`, with sizes of 1, 16, 64, 256, and
   512 KiB, filled with zero bytes.
2. Start one `binserve` process with
   [`binserve.json`](./binserve/binserve.json), which listens on
   `127.0.0.1:8080`, routes `/1k` to `/512k` to the files, and turns on the
   in-memory file cache and turns off logging. This process serves every
   configuration.
3. Wait for `binserve` to be ready: send a `HEAD /1k` request up to 10 times,
   1 second apart, until it answers `200 OK`. If it never does, print
   `Couldn't start server` and exit with code 1.

Then, for each file size from the smallest to the largest, the harness runs
`wrk` once for each combination of the following counts:

- **Threads:** 1, 2, or 4 per CPU.
- **Connections:** 1, 2, 4, 8, 16, or 32 per CPU, but at most 16 per CPU for
  the 256 KiB and 512 KiB files. A comment in `benchmark.py` says smaller files
  likely need more connections to saturate the machine.

Combinations with fewer connections than threads are skipped. This leaves 15
configurations for each of the three smaller files and 12 for each of the two
larger ones, 69 in all, whatever the CPU count. Each configuration runs the
following command, one after another, with no warmup:

```bash
wrk -d 10 -t <threads> -c <connections> http://localhost:8080/<size>
```

Before and after each run, the harness records the CPU times of `binserve` and
of `wrk`, which give the CPU columns of the row.

## References

See [References](./docs/references.md) for the terms specific to this image.
