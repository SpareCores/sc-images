# benchmark-redis

This benchmark measures Redis 7.4 SET throughput and latency with
`memtier_benchmark`. It starts one `redis-server` and one `memtier_benchmark`
client for each CPU, all in the same container and connected over loopback,
and runs every configuration for 10 seconds.

It reports the total operations per second and the average latency of the
Redis SET command at six pipeline depths, from 1 to 512, together with the CPU
time the Redis servers and the clients used.
[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes these
as Redis server+client speed, Redis server (extrapolated) speed, and Redis
latency.

## Purpose

This benchmark shows how fast a server type serves a simple in-memory
key-value workload, and how throughput and latency change as each client
connection sends more requests at once.

Because each Redis server shares its CPU with its `memtier_benchmark` client,
this benchmark records their CPU time separately, so the servers' share of the
work can be separated from the clients' share. See [Results](#results) for how
the Navigator uses it.

## Limitations

These results compare Redis SET performance on one host, not the throughput of
a production Redis deployment. The main limitations are the following:

- **Shared CPUs:** Each Redis server shares its CPU with a client, so `rps` is
  not the maximum a dedicated Redis server would reach. See
  [Results](#results) for the CPU-time-adjusted score the Navigator publishes.
- **[Workload](#workload):** Only the SET command is measured, with
  `memtier_benchmark`'s default key pattern, key range, value size, and
  connection count. GET and mixed workloads are not measured.
- **Loopback Only:** Clients and servers run on the same host and talk over
  loopback, so this benchmark does not measure network throughput or latency.
- **Untuned Servers:** Each `redis-server` starts with only a data directory
  and a port, so Redis 7.4.0 defaults apply, including persistence, memory, and
  threading settings.
- **Single Short Run:** Each configuration runs once for 10s, with no warmup
  and no repetitions.
- **CPU Count:** The number of server and client pairs follows the CPUs the
  container may run on. Docker's `--cpuset-cpus` changes it, but `--cpus` does
  not: with `--cpus`, this benchmark still starts one pair per host CPU.
- **Error Checks:** The harness does not check `memtier_benchmark` exit codes,
  and it continues even if some servers do not answer the flush before a
  configuration. A client that prints no result adds nothing to `rps` but still
  counts in the `latency` average.

## Usage

Run the image via Docker:

```bash
docker run --rm --privileged ghcr.io/sparecores/benchmark-redis:main
```

The harness runs the Redis servers and clients at the highest scheduling
priority (`nice -n -20`), which needs the `CAP_SYS_NICE` capability that
`--privileged` grants. Production runs the container the same way and also
sets host networking (`--network host`); this benchmark's traffic itself stays
on loopback inside the container.

The entrypoint runs this benchmark under `resource-tracker`, which samples
resource usage during the run. Its data is not published.

Measurement takes 60 seconds (six configurations of 10 seconds each), plus
server startup and a flush of all servers before each configuration.

### Settings

This benchmark reads no environment variables or command-line options. The
operation, pipeline depths, run time, and ports are fixed in
[`benchmark.py`](./benchmark.py); see [Workload](#workload). To change the
number of server and client pairs, limit the container's CPUs with
`--cpuset-cpus`.

### Results

This benchmark prints a comma-separated table to stdout: a header row with
quoted column names, then one row per pipeline depth. Nothing else is written
besides Redis data under `/data` in the container.

```text
"operation","threads","pipeline","rps","latency","server_usr","server_sys","client_usr","client_sys"
```

The columns mean the following:

- `operation` is always `"SET"`, and `threads` is always `1`, the
  `memtier_benchmark` thread count per client.
- `pipeline` is the `memtier_benchmark` `--pipeline` value: how many requests
  each connection sends before waiting for replies.
- `rps` is the sum of SET operations per second over all clients, as an
  integer.
- `latency` is the average SET latency in milliseconds: the mean of each
  client's average latency, not weighted by its operation count.
- `server_usr` and `server_sys` are the user and system CPU seconds that all
  Redis servers used during the configuration.
- `client_usr` and `client_sys` are the user and system CPU seconds of the
  `memtier_benchmark` clients of the configuration.

For example, on an `aws/m5.large` (2 vCPUs), `rps` rises from 69,962 at
pipeline 1 to 685,686 at pipeline 512, while `latency` rises from 1.43 ms to
74.36 ms
([production output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/redis/stdout)).

The Navigator publishes the following three scores for each pipeline depth:

- **Redis server+client speed:** `rps`, in operations per second.
- **Redis server (extrapolated) speed:** `rps` multiplied by the ratio of all
  CPU time (servers and clients) to the servers' CPU time, in operations per
  second.
- **Redis latency:** `latency`, in milliseconds; lower is better.

Each score carries a note with the server and client CPU seconds. Production
records the Redis version from `redis-server -v`. `memtier_benchmark` is
installed from the Redis package repository without a version pin, so a
rebuild of the image can change it, and production does not record its
version.

## Workload

This benchmark runs one workload: a sweep of Redis pipelining depths for SET
operations. [`benchmark.py`](./benchmark.py) starts one `redis-server` per
CPU, on ports 7000 and up, each with its own data directory under `/data`. The
servers run until the container exits.

For each pipeline depth (1, 4, 16, 64, 256, and 512), in this order, the
harness runs the following steps:

1. Flush every server with `redis-cli flushall`, retrying up to 10 times,
   1 second apart, until all servers answer. This is also the only wait for
   the servers to start.
2. Record the CPU times of the servers and the clients' running totals.
3. Start one client per server, all at once:

   ```bash
   memtier_benchmark -4 -p <port> --ratio=1:0 --test-time=10 --hide-histogram -t 1 --pipeline=<depth>
   ```

   `--ratio=1:0` sends SET operations only.
4. Wait for every client to exit, read the SET results from each client's
   summary table, and record the CPU times again.

## Design History

An earlier version of the harness also ran each pipeline depth with two and four
`memtier_benchmark` threads per client, which produced 18 rows instead of six.
We cut it to one thread in August 2024: with one server and client pair per
CPU, more threads do not improve performance. A few published results still
come from the earlier version, for example
[`aws/c6g.16xlarge`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c6g.16xlarge/0/redis/stdout).
For each pipeline depth, the Navigator keeps the row with the highest `rps`,
regardless of the thread count.

## References

See [References](./docs/references.md) for the terms specific to this image.
