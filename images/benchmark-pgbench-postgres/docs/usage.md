# Usage

## Standalone Mode

Leave `SC_DB_HOST` empty to automatically start a local PostgreSQL server inside
the Docker container:

```bash
docker run --rm ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

The local server will automatically tune its settings based on the identified
vCPU count and system memory size using the following algorithm:

- Take the host system memory (Docker `--memory` limits are ignored), `M`, from
  `MemTotal` in `/proc/meminfo`, rounded down to whole GiB (at least `1`).
- Take the vCPU count, `C`, from the optional `SC_DB_VCPUS` environment
  variable, which defaults to the container's vCPU count.
- Generate PostgreSQL settings as [pgtune](https://pgtune.leopard.in.ua/) does
  with its form defaults: PostgreSQL 18, Linux, web application, SSD storage,
  database size mid-RAM, and an automatic connection count.
- Set `shared_buffers` to `M/4` and `effective_cache_size` to `3M/4`.
- Set `huge_pages` to `try` when `shared_buffers` is at least 2 GB, otherwise
  `off`.
- Set `maintenance_work_mem` to `M/16`, capped at 8 GB. Set
  `autovacuum_work_mem` to 2 GB when `maintenance_work_mem` reaches 2 GB.
- Set `work_mem` to `(M − shared_buffers) / (3 × (200 + W))`, with a 4 MB
  minimum, where `W` is `C` when `C` ≥ 4 and 1 otherwise.
- Set `wal_buffers` to 3% of `shared_buffers`, capped at 16 MB.
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

The output records the generated settings in `postgres.requested_gucs` and the
matching `pgtune` form link in `pgtune_share_url`. `pgbench_ro` applies further
[workload settings](#workload-settings) on top of these.

## Remote Mode

Set `SC_DB_HOST` to benchmark an existing PostgreSQL server, such as a DBaaS
(Database as a Service) instance, from a separate client. Before you run,
check the following prerequisites:

- The `SC_DB_USER` role can create databases (`CREATEDB`) and terminate other
  sessions on the benchmark database, as its owner, a member of
  `pg_signal_backend`, or a superuser.
- The benchmark database, `SC_PGBENCH_DB` (default `pgbench`), holds no data
  you need. Restoring a cached dump drops and recreates it; building the
  dataset drops and recreates its benchmark tables.
- The `SC_DB_USER` role owns the benchmark database or is a superuser, so it
  can set the `pgbench_ro` [workload settings](#workload-settings).
- The client can reach `SC_CDN_BASE_URL` over HTTPS. The image restores a
  cached dataset dump from there and builds the dataset itself only when no
  dump exists for it; the run fails if the CDN is unreachable.
- You know the database server's vCPU count. Set it in `SC_DB_VCPUS`, because
  the default is the client container's CPU count, and on a separate client VM
  that produces the wrong concurrency points.

Set the host, credentials, and server size:

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_DB_VCPUS=<database-vcpus> \
  -e SC_DB_MEM_GIB=<database-memory-gib> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

`SC_DB_MEM_GIB` is only recorded in the output; for a remote server it is
`null` unless you set it.

The image does not tune a remote server's configuration; the only settings it
changes are the [workload settings](#workload-settings) on the benchmark
database.

## Workload Settings

In both modes, `pgbench_ro` sets the following defaults on its benchmark
database with `ALTER DATABASE … SET`, after building or restoring the dataset:

- `jit`: `off`
- `work_mem`: `64MB`
- `max_parallel_workers_per_gather`: `0`

They apply to every `pgbench` session and override the server-level values,
including the standalone `pgtune` settings. `pgbench_tpcb` changes no settings.

## Run Duration

A run prepares the dataset, then measures throughput at a series of
concurrency points (client counts) in the following steps:

1. Restore the dataset from a cached dump on the CDN, or build it if no dump
   exists for it.
2. Warm up once for `SC_WARMUP_SECONDS` (default 120 s) at the first
   concurrency point. Before each later point, run a settle period of
   `SC_SETTLE_SECONDS` (default 60 s) at that point's client count. Neither is
   measured. With `SC_WARMUP_ONCE=false`, every point gets the full warmup.
3. Measure each concurrency point for `SC_RUN_SECONDS` (default 300 s),
   sampling 1% of transaction latencies.
4. Report the highest TPM of all concurrency points (and scale factors) as the
   score.

For `pgbench_ro`, the concurrency points are `{1, V/2, V, 2·V}`, where `V` is
`SC_DB_VCPUS`. Duplicates collapse on small servers, so the default duration
depends on `V` as follows, excluding dataset preparation:

| `V` | Concurrency points | Duration |
| --- | --- | --- |
| 1 | 1, 2 | 13 minutes (120 + 60 + 2 × 300 s) |
| 2 | 1, 2, 4 | 19 minutes (120 + 2 × 60 + 3 × 300 s) |
| 3 | 1, 3, 6 | 19 minutes (120 + 2 × 60 + 3 × 300 s) |
| 4 or more | 1, V/2, V, 2·V | 25 minutes (120 + 3 × 60 + 4 × 300 s) |

For example, a standalone run on a 96-vCPU AWS `m9g.24xlarge` measured 1, 48,
96, and 192 clients and took 25 minutes 25 seconds in total. Restoring the
dataset from the CDN and starting the server accounted for the extra 25
seconds; building the dataset instead takes longer.

The following settings also change the number of concurrency points, and so
the duration:

- `SC_PROFILE_VUS` replaces the default concurrency points.
- Points above the server's `max_connections` minus 50 are dropped, which can
  shorten runs against a remote server with a low connection limit.
- `pgbench_tpcb` prepares and measures each scale factor in `SC_SCALEFACTORS`
  in turn, with a settle period before every point after the first warmup.

Each `pgbench` call times out after its duration plus 600 seconds, and dataset
preparation times out after 4 hours.

## Key Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `SC_WORKLOAD` | Workload: `pgbench_ro` or `pgbench_tpcb`. | `pgbench_ro` |
| `SC_DB_HOST` | Remote database host; empty starts local PostgreSQL. | Empty |
| `SC_DB_PORT` | Database port. | `5432` |
| `SC_DB_USER` | Database role. | `postgres` |
| `SC_DB_PASSWORD` | Database password. | `postgres` |
| `SC_DB_NAME` | Admin database used for setup and settings queries. | `postgres` |
| `SC_PGBENCH_DB` | Database used by the benchmark; its contents are dropped and recreated. | `pgbench` |
| `SC_DB_SSLMODE` | SSL (Secure Sockets Layer) mode for database and dataset-dump connections. | `prefer` |
| `SC_CPU_SCALE` | `pgbench_ro` transaction work multiplier. | `1` |
| `SC_SCALEFACTORS` | Comma-separated `pgbench_tpcb` scale factors. | Unset |
| `SC_SCALEFACTOR` | `pgbench_tpcb` scale factor when `SC_SCALEFACTORS` is unset or empty. | `65` |
| `SC_PROFILE_VUS` | Comma-separated concurrency anchors. | Derived from database vCPUs |
| `SC_PROFILE_SEARCH` | Allow adaptive concurrency search; forced off for `pgbench_ro`. | True for `pgbench_tpcb`; false for `pgbench_ro` |
| `SC_PROFILE_IMPROVE_PCT` | Throughput improvement threshold for TPC-B (Transaction Processing Performance Council Benchmark B) search. | `5.0` |
| `SC_PROFILE_MAX_CLIENTS` | Maximum client count for the profile. | Highest anchor |
| `SC_PROFILE_HARD_MAX_CLIENTS` | Hard concurrency ceiling. | Highest anchor for `pgbench_ro`; `3072` for TPC-B. |
| `SC_RUN_SECONDS` | Measurement duration per concurrency rung. | `300` |
| `SC_WARMUP_SECONDS` | Initial warmup duration. | `120` |
| `SC_SETTLE_SECONDS` | Settle duration between concurrency rungs. | `60` |
| `SC_WARMUP_ONCE` | Use one full warmup, then settle between rungs. | `true` |
| `SC_DB_VCPUS` | Database vCPU count used for concurrency anchors and local tuning. Set it for remote databases: the default is the client's CPU count. | `os.cpu_count() or 2` |
| `SC_CLIENT_VCPUS` | Client vCPU count recorded in output. | `os.cpu_count()` or `2` |
| `SC_DB_MEM_GIB` | Database memory in GiB recorded in output. Set it for remote databases. | Detected locally; unset for remote databases |
| `SC_DURABILITY` | Local server durability; `async` disables `synchronous_commit`. | `durable` |
| `SC_TOPOLOGY` | Topology recorded in output. | `single_vm` locally; otherwise `multi_vm` |
| `SC_CDN_BASE_URL` | CDN (Content Delivery Network) base URL for cached dataset dumps. | `https://cdn.sparecores.net/sc-inspector` |
| `SC_CDN_DATASET_POST_B64` | Optional base64-encoded presigned upload configuration for dataset dumps. | Unset |
| `SC_CDN_UPLOAD` | Permit dataset upload when a valid upload configuration is supplied. | `1` |

The process prints one indented, key-sorted JSON object to
stdout. It includes a per-concurrency `profile` array and a headline `score`
in TPM (transactions per minute). Profile behavior depends on the workload:

- `pgbench_ro` reports TPM only, not TPS (transactions per second), and uses
  the fixed concurrency profile `{1, V/2, V, 2·V}`. It caps `pgbench` worker
  jobs at 32 per run, independently of the client count.
- `pgbench_tpcb` uses geometric concurrency anchors with optional adaptive
  search.
