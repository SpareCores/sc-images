# Usage

## Standalone Mode

Leave `SC_DB_HOST` empty to automatically start a local PostgreSQL server inside
the Docker container:

```bash
docker run --rm ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

**Note:** Docker `--memory`, `--cpus`, and `--cpuset-cpus` limits do not change
the tuning. With `--memory`, PostgreSQL is still sized for the host's memory and
can be killed for running out of memory; with `--cpus` or `--cpuset-cpus`, set
`SC_DB_VCPUS` to match.

The harness tunes the local server at startup, based on the identified vCPU
count and system memory size, using the following algorithm, implemented in
[`pgtune_leopard.py`](../pgtune_leopard.py):

- Take the host system memory (Docker `--memory` limits are ignored), `M`, from
  `MemTotal` in `/proc/meminfo`, rounded down to whole GiB (at least `1`).
- Take the vCPU count, `C`, from the optional `SC_DB_VCPUS` environment
  variable, which defaults to the host's logical CPU count, even when Docker
  limits the container's CPUs.
- Generate PostgreSQL settings as [`pgtune`](https://pgtune.leopard.in.ua/) does
  with its form defaults: PostgreSQL 18, Linux, web application, SSD storage,
  database size mid-RAM, and an automatic connection count.
- Set `shared_buffers` to `M/4` and `effective_cache_size` to `3M/4`.
- Set `huge_pages` to `try` when `shared_buffers` is at least 2 GB, otherwise
  `off`.
- Set `maintenance_work_mem` to `M/16`, capped at 8 GB. Set
  `autovacuum_work_mem` to 2 GB when `maintenance_work_mem` reaches 2 GB.
- Set `work_mem` to `(M - shared_buffers) / (3 × (200 + W))`, with a 4 MB
  minimum, where `W` is `C` when `C` ≥ 4 and 1 otherwise.
- Set `wal_buffers` to 3% of `shared_buffers`, with a 32 kB minimum and a 16 MB
  cap; values between 14 and 16 MB are rounded up to 16 MB.
- Use fixed values for the following settings:

    - `min_wal_size`: `1 GB`
    - `max_wal_size`: `4 GB`
    - `checkpoint_completion_target`: `0.9`
    - `default_statistics_target`: `100`
    - `random_page_cost`: `1.1`
    - `effective_io_concurrency`: `200`
    - `jit`: `off`
    - `wal_compression`: `lz4`
    - `io_method`: `io_uring`

- When `C` ≥ 4, set `max_worker_processes` and `max_parallel_workers` to `C`,
  and `max_parallel_workers_per_gather` and `max_parallel_maintenance_workers`
  to `C`/2 rounded up, capped at 4.
- Set `autovacuum_max_workers` to 4 when `C` ≥ 16 and to 5 when `C` ≥ 32;
  otherwise, keep the PostgreSQL default.
- Override `pgtune`'s `max_connections` of 200 with 3,122 (the highest client
  count, 3,072, plus 50 reserved connections). `work_mem` is still sized for 200
  connections.
- Set `synchronous_commit` to `off` when `SC_DURABILITY=async`, and `on`
  otherwise.

The output records the generated settings in `postgres.requested_gucs` and a
`pgtune` form link in `pgtune_share_url`. The link shows `pgtune`'s default of
200 connections, not the 3,122 override. `pgbench_ro` applies further [workload
settings](#workload-settings) on top of these.

## Remote Mode

Set `SC_DB_HOST` to benchmark an existing PostgreSQL 18 server, such as a DBaaS
(Database as a Service) instance, from the same or a separate machine. Before
you run, check the following prerequisites:

- The `SC_DB_USER` role can create databases (`CREATEDB`) and terminate other
  sessions on the benchmark database: it must be a superuser, a member of
  `pg_signal_backend`, or the role those sessions run as.
- The benchmark database, `SC_PGBENCH_DB` (default `pgbench`), holds no data you
  need. Restoring a cached dump drops and recreates it; building the dataset
  drops and recreates its benchmark tables.
- The `SC_DB_USER` role owns the benchmark database or is a superuser, so it can
  set the `pgbench_ro` [workload settings](#workload-settings).
- The client can reach `SC_CDN_BASE_URL` over HTTPS (Hypertext Transfer Protocol
  Secure). The image restores a cached dataset dump from there and builds the
  dataset itself only when no dump exists for it; the run fails if the CDN
  (Content Delivery Network) is unreachable.
- You know the database server's vCPU count. Set it in `SC_DB_VCPUS`, because
  the default is the client host's logical CPU count, and on a separate client
  VM that produces the wrong concurrency points.

Set the host, credentials, and server size:

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_USER=<username> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_DB_VCPUS=<database-vcpus> \
  -e SC_DB_MEM_GIB=<database-memory-gib> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

`SC_DB_MEM_GIB` is only recorded in the output; for a remote server it is `null`
unless you set it.

The image does not tune a remote server's configuration; the only settings it
changes are the [workload settings](#workload-settings) on the benchmark
database.

## Workload Settings

In both modes, `pgbench_ro` sets the following defaults
([`benchmark.py`](../benchmark.py)) on its benchmark database with `ALTER
DATABASE ... SET`, after building or restoring the dataset:

- `jit`: `off`
- `work_mem`: `64MB`
- `max_parallel_workers_per_gather`: `0`

They apply to every `pgbench` session and override the server-level values,
including the standalone `pgtune` settings. `pgbench_tpcb` changes no settings.

## Run Duration

A run prepares the dataset, then measures throughput at a series of concurrency
points (client counts) in the following steps:

1. Restore the dataset from a cached dump on the CDN, or build it if no dump
   exists for it.
2. Warm up once for `SC_WARMUP_SECONDS` (default 120 s) at the first concurrency
   point. Before each later point, run a settle period of `SC_SETTLE_SECONDS`
   (default 60 s) at that point's client count. Neither is measured. With
   `SC_WARMUP_ONCE=false`, every point gets the full warmup.
3. Measure each concurrency point for `SC_RUN_SECONDS` (default 300 s), sampling
   1% of transaction latencies.
4. Report the highest TPM (Transactions Per Minute) of all concurrency points
   (and scale factors) as the score.

For `pgbench_ro`, the concurrency points are `{1, V/2, V, 2·V}`, where `V` is
`SC_DB_VCPUS` and `V/2` is rounded down. Duplicates collapse on small servers,
so the default duration depends on `V` as follows, excluding dataset
preparation:

| `V` | Concurrency points | Duration |
| --- | --- | --- |
| 1 | 1, 2 | 13 minutes (120 + 60 + 2 × 300 s) |
| 2 | 1, 2, 4 | 19 minutes (120 + 2 × 60 + 3 × 300 s) |
| 3 | 1, 3, 6 | 19 minutes (120 + 2 × 60 + 3 × 300 s) |
| 4 or more | 1, V/2, V, 2·V | 25 minutes (120 + 3 × 60 + 4 × 300 s) |

For example, a production run on a 96 vCPU AWS (Amazon Web Services)
`m9g.24xlarge` measured 1, 48, 96, and 192 clients and took 25 minutes 25
seconds in total ([recorded
output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m9g.24xlarge/pgbench_postgres_ro_durable/stdout)),
including restoring the dataset from the CDN and starting the server. Building
the dataset instead takes longer.

The following settings also change the number of concurrency points, and so the
duration:

- `SC_PROFILE_VUS` replaces the default concurrency points for `pgbench_ro`.
- Points above the server's `max_connections` minus 50 are dropped, which can
  shorten runs against a remote server with a low connection limit.
- `pgbench_tpcb` prepares and measures each scale factor in `SC_SCALEFACTORS` in
  turn, with a settle period before every point after the first warmup.

Each `pgbench` call times out after its duration plus 600 seconds. Each dataset
build or restore command times out after 4 hours; the optional dump upload has
no timeout.

### Why 5 Minutes

This is deliberately not a long-running benchmark. We tested 5-, 10-, 15-, and
30-minute measurement windows with five interleaved trials each, using BenchBase
Wikipedia on three GCP (Google Cloud Platform) server types and `pgbench -S` on
a fourth. Longer windows moved mean throughput by less than 2% and did not
reduce run-to-run variation, which comes from load, OS, and noisy-neighbor
effects rather than from too short an average. See [Measurement
duration](../CHANGELOG.md#measurement-duration) for the results.

As a result, each score is a single 5-minute sample. Repeated runs on the same
server type varied by a CV (Coefficient of Variation) of about 0.5-4% in these
tests, so treat smaller differences between server types as noise. These
figures come from the BenchBase Wikipedia and `pgbench -S` runs, not from
`pgbench_ro`.

## Production Setup

Spare Cores runs this image through `sc-inspector` orchestration with the
following setup. A plain `docker run` differs from it as noted below.

### Container Settings

Production runs apply no `sysctl` or other host OS tweaks. The container runs
privileged with the following settings:

- host networking
- `seccomp=unconfined`
- high `nofile` and unlimited `memlock` ulimits (unlocking huge pages and
  `io_uring`)

In standalone mode, the harness itself starts `postgres` under `nice -n -20`.
The higher priority takes effect only when the container has `CAP_SYS_NICE`, as
under `--privileged`; with a plain `docker run`, the server runs at normal
priority.

### Topology

IaaS (Infrastructure as a Service) runs place the client and database on the
same node. DBaaS runs use a separate client VM that always reaches the database
over private VPC (Virtual Private Cloud) addresses. On AWS, both are in the same
availability zone; other vendors may place them in different zones of the same
region.

## Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `SC_WORKLOAD` | Workload: `pgbench_ro` or `pgbench_tpcb`. | `pgbench_ro` |
| `SC_DB_HOST` | Remote database host; empty starts local PostgreSQL. | Empty |
| `SC_DB_PORT` | Database port. Remote mode only: leave it unset in standalone mode, which always uses `5432`. | `5432` |
| `SC_DB_USER` | Database role. Remote mode only: leave it unset in standalone mode, which always uses `postgres`. | `postgres` |
| `SC_DB_PASSWORD` | Database password. | `postgres` |
| `SC_DB_NAME` | Admin database used for setup and settings queries. | `postgres` |
| `SC_PGBENCH_DB` | Database used by this benchmark; its contents are dropped and recreated. | `pgbench` |
| `SC_DB_SSLMODE` | SSL (Secure Sockets Layer) mode for the dataset restore, `pg_dump`, and the connection that drops and recreates the benchmark database. `pgbench` and the setup queries do not use it. | `prefer` |
| `SC_CPU_SCALE` | `pgbench_ro` transaction work multiplier. | `1` |
| `SC_SCALEFACTORS` | Comma-separated `pgbench_tpcb` scale factors. | Unset |
| `SC_SCALEFACTOR` | `pgbench_tpcb` scale factor when `SC_SCALEFACTORS` is unset or empty. | `65` |
| `SC_PROFILE_VUS` | Comma-separated `pgbench_ro` concurrency points. For `pgbench_tpcb`, it only sets the default `SC_PROFILE_MAX_CLIENTS`. | Derived from database vCPUs |
| `SC_PROFILE_SEARCH` | Allow adaptive concurrency search for `pgbench_tpcb`; forced off for `pgbench_ro`. At the default scale factor, it takes effect only when `SC_PROFILE_MAX_CLIENTS` is above the highest anchor. | True for `pgbench_tpcb`; false for `pgbench_ro` |
| `SC_PROFILE_IMPROVE_PCT` | Throughput improvement threshold for `pgbench_tpcb` search. | `5.0` |
| `SC_PROFILE_MAX_CLIENTS` | Client-count limit for the planned profile. `pgbench_tpcb` search can extend past it while throughput improves, up to the scale factor. Raising it above the highest anchor lets `pgbench_tpcb` search add points only when a ladder rung lies between that anchor and the scale factor. | Highest anchor |
| `SC_PROFILE_HARD_MAX_CLIENTS` | `pgbench_tpcb` only: the highest client count search can reach, also capped by the scale factor and by `max_connections` minus 50. Has no effect on `pgbench_ro`. | Highest anchor for `pgbench_ro`; `3072` for `pgbench_tpcb` |
| `SC_RUN_SECONDS` | Measurement duration per concurrency point. | `300` |
| `SC_WARMUP_SECONDS` | Initial warmup duration. | `120` |
| `SC_SETTLE_SECONDS` | Settle duration between concurrency points. | `60` |
| `SC_WARMUP_ONCE` | Use one full warmup, then settle between concurrency points. | `true` |
| `SC_DB_VCPUS` | Database vCPU count used for concurrency anchors and local tuning. Set it for remote databases: the default is the client host's logical CPU count. | `os.cpu_count()` or `2` |
| `SC_CLIENT_VCPUS` | Client vCPU count recorded in output. | `os.cpu_count()` or `2` |
| `SC_DB_MEM_GIB` | Database memory in GiB recorded in output. Set it for remote databases. | Detected locally; unset for remote databases |
| `SC_DURABILITY` | Local server durability; `async` disables `synchronous_commit`. | `durable` |
| `SC_TOPOLOGY` | Topology recorded in output. | `single_vm` locally; otherwise `multi_vm` |
| `SC_CDN_BASE_URL` | CDN base URL for cached dataset dumps. | `https://cdn.sparecores.net/sc-inspector` |
| `SC_CDN_DATASET_POST_B64` | Optional base64-encoded presigned upload configuration for dataset dumps. | Unset |
| `SC_CDN_UPLOAD` | Permit dataset upload when a valid upload configuration is supplied. | `1` |

See [Results](../README.md#results) for the output fields.
