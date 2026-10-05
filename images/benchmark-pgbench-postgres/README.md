# benchmark-pgbench-postgres

This PostgreSQL benchmark client measures cloud server database performance with
`pgbench`. In standalone mode, the image starts a local PostgreSQL 18 server. In
remote mode, it connects to the configured host without checking or enforcing
the remote server version. The client can run on the same node as the database
or on a separate machine.

This benchmark reports throughput and concurrency behavior in TPM (Transactions
Per Minute), as well as latency across varying client counts, so users can see both peak performance
and how scalability changes with load on self-managed and managed PostgreSQL on
the same hardware.

## Purpose

Our [Navigator](https://sparecores.com/servers) project publishes empirical
performance measurements for more than 5,000 cloud server types. We needed an
RDBMS ([Relational Database Management
System](https://en.wikipedia.org/wiki/Relational_database)) benchmark that
tracks relevant metrics directly rather than relying on proxies. Many of our
other database benchmarks have close-enough approximations, but they aren't
fully accurate; PassMark database operations don't scale to larger instances,
Redis is not relational, and raw CPU speed and memory bandwidth are proxies
rather than database workloads.

The same benchmark client measures two deployment models:

- **IaaS (Infrastructure as a Service):** Self-hosted PostgreSQL, with the
  client and database on the same node.
- **DBaaS:** Provider-managed PostgreSQL, with a
  separate client VM; the provider provisions, manages, and tunes the database
  engine.

## Limitations

These results compare PostgreSQL server CPU and memory performance, not application
throughput. The main limitations are:

- **Workload:** `pgbench_ro` uses synthetic transactions and uniform data; the
  results do not specifically predict any given application's throughput, but aim to provide meaningful comparisons across different server types. See the
  [workload limitations](./docs/limitations.md#disclaimer-what-this-benchmark-does-not-deliver).
- **Disk and network:** The score excludes disk performance and does not measure
  network throughput or latency. See [deliberate exclusions](./docs/limitations.md#deliberate-exclusions).
- **Engine tuning:** When `SC_DB_HOST` is unset, the image starts PostgreSQL
  locally and tunes it with `pgtune`. A remote `SC_DB_HOST` target is not tuned
  by this benchmark. See [operational
  details](./docs/limitations.md#operational-details).
  
  **Note**: DBaaS engines are not tuned by the benchmark, and minor versions may
    vary when providers do not support pinning. See
  [Minor Engine Versions](./docs/limitations.md#minor-engine-versions) for details.

## Usage

Run the image via Docker. Without `SC_DB_HOST`, it starts a local PostgreSQL 18
server in the container (standalone mode):

```bash
docker run --rm ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

Set `SC_DB_HOST` to benchmark a remote PostgreSQL server instead. A remote run
needs more setup: database privileges, a disposable benchmark database, and the
server's vCPU count. Follow [Remote Mode](./docs/usage.md#remote-mode) for more
details and settings.

Both modes tries to download the dataset from the Spare Cores CDN (Content
Delivery Network) over HTTPS.

A default run takes about 25 minutes on servers with 4 or more vCPUs: a short
warmup, then 5 minutes of measurement at each of four client counts. See
[Run Duration](./docs/usage.md#run-duration) for the steps and how to change
them.

### Settings

Common settings for `pgbench_ro` are listed here; see the [full environment-variable
reference](./docs/usage.md#key-environment-variables) for all options and
defaults:

- `SC_DB_HOST`, `SC_DB_PORT`, `SC_DB_USER`, and `SC_DB_PASSWORD` configure a  
  remote connection. `SC_DB_SSLMODE` controls SSL mode.  
- `SC_DB_VCPUS` sets the database vCPU count used to derive concurrency points  
  and the local server settings in standalone mode.  
- `SC_CPU_SCALE` changes `pgbench_ro` transaction work.  
- `SC_RUN_SECONDS`, `SC_WARMUP_SECONDS`, and `SC_SETTLE_SECONDS` control  
  measurement and warmup timing.

### Results

The process prints one JSON object to stdout. Interpret its main fields as
follows:

- `score` is the highest TPM (transactions per minute) result; `score_unit` is
  `tpm`, and `peak_concurrency` is the client count for that result.
- Each `sizes[]` entry represents a workload size and has its own score and
  `profile`. `pgbench_ro` entries identify `cpu_scale`; `pgbench_tpcb` entries
  identify `scalefactor`.
- Each `profile[]` entry is one concurrency measurement. `concurrency` is the
  client count and `jobs` is the worker count. For `pgbench_ro`, `V` is the
  database vCPU count; it uses fixed points `{1, V/2, V, 2·V}` and caps worker
  jobs at 32. `pgbench_tpcb` uses geometric anchors with optional adaptive
  search.
- `latency_ms` contains sampled p50, p95, and p99 latency and an average, in
  milliseconds. The harness samples 1% of transaction latency logs.

## Workloads

The image supports the following workloads with different transaction patterns:

- [`pgbench_ro`](./docs/workloads.md#pgbench_ro) (default) runs a custom,
  read-only transaction over a fixed schema, spreading work across eight
  PostgreSQL subsystems. `SC_CPU_SCALE` changes transaction work without
  resizing the schema.
- [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb) runs [`pgbench`](https://www.postgresql.org/docs/current/pgbench.html)'s built-in
  `tpcb-like` transaction mix against a schema initialized at one or more
  scale factors.

For more information on available workloads, see
[Workloads](./docs/workloads.md).

## Design History

We first compared `sysbench`, HammerDB TPROC-C, BenchBase, and `pgbench`. The
experiments exposed practical limits: some suites required substantial client
resources, while write-heavy workloads got bottlenecked by storage and WAL.
On `tmpfs`, write-heavy OLTP results improved by 10–25%, showing how much disk
behavior could influence a score. But `tmpfs` was unavailable for DBaaS, and
warehouse or scale-factor sizing could not cover the range from 1 vCPU to
thousands. `pgbench` looked promising as a simpler, more scalable basis for a
cross-provider benchmark.

Early `pgbench -S` runs were dominated by network latency, and profiling found
that the first custom transaction over-weighted regex work. Dataset comparisons
exposed storage effects in write-heavy suites, while 21 PostgreSQL configuration
experiments on a 32-vCPU host showed that tuning could improve throughput by
about 20%. The `pgbench_ro` redesign uses a fixed, cache-resident schema and
spreads CPU work across PostgreSQL subsystems; `pgbench_tpcb` remains a
conventional reference workload.

The workload remains synthetic: its generated data is uniform and it does not
predict application throughput or measure disk and network performance. See the
[changelog](./CHANGELOG.md) for experiment results, validation, and calibration
details.

## References

See the full list of [References](./docs/references.md) used in this
documentation.
