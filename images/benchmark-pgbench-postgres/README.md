# benchmark-pgbench-postgres

This PostgreSQL benchmark client measures how well cloud servers handle
PostgreSQL database workloads using `pgbench` against a local PostgreSQL 18
instance. Remote servers use what the host provides, but DBaaS (Database as a
Service) targets are set to PostgreSQL major version 18 (without minor version
pinning). It is
designed to be agnostic to whether the client and the database are hosted on the
same machine or separately.

This benchmark reports throughput and concurrency behavior, typically in
transactions per minute and across varying client counts, so users can see both
peak performance and how scalability changes with load.

## Purpose

Our [Navigator](https://sparecores.com/servers) project publishes empirical
performance measurements for more than 5,000 cloud server types. We needed an
RDBMS ([Relational Database Management
System](https://en.wikipedia.org/wiki/Relational_database)) benchmark that
tracks relevant metrics rather than relying on proxies. Many other database
benchmarks lack actual, proper database measurements; PassMark database
operations don't scale to larger instances, Redis is not relational, and raw CPU
speed and memory bandwidth are proxies rather than database workloads.

The same benchmark client measures two deployment models:

- **IaaS (Infrastructure as a Service):** Self-hosted PostgreSQL, with the
  client and database on the same node.
- **DBaaS:** Provider-managed PostgreSQL, with a
  separate client VM; the provider provisions, manages, and tunes the database
  engine.

The benchmark reports a headline score in TPM (transactions per minute) and a
concurrency profile showing throughput at different client counts. These results
support comparisons of self-managed and managed PostgreSQL on the same hardware.

## Limitations

These results compare PostgreSQL server CPU and memory behavior, not application
throughput. The main limitations are:

- **Workload:** `pgbench_ro` uses synthetic transactions and uniform data; the
  results do not predict any specific application's throughput. See the
  [workload limitations](./docs/limitations.md#disclaimer-what-this-benchmark-does-not-deliver).
- **Disk and network:** The score excludes disk performance and does not measure
  network throughput or latency. See [deliberate exclusions](./docs/limitations.md#deliberate-exclusions).
- **Managed database configuration:** DBaaS engines are not tuned by the
  benchmark, and minor versions may vary when providers do not support pinning.
  See [exclusions](./docs/limitations.md#exclusions).
- **Standalone tuning:** When `SC_DB_HOST` is unset, the image starts PostgreSQL
  locally and tunes it with `pgtune`. A remote `SC_DB_HOST` target is not tuned
  by the benchmark. See [operational details](./docs/limitations.md#operational-details).

## Usage

This benchmark can be run via Docker:

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

For more information on how to use this benchmark, see [Usage](./docs/usage.md).

## Workloads

`benchmark-pgbench-postgres` can be used with the following workloads:

- [`pgbench_ro`](./docs/workloads.md#pgbench_ro) - our custom-built, read-only
  PostgreSQL benchmark sized to fit in `shared_buffers`, with a monolithic SQL
  script that runs several blocks, each touching a different PostgreSQL
  subsystem.
- [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb) -
  [`pgbench`](https://www.postgresql.org/docs/current/pgbench.html)'s built-in
  script with a standard schema

For more information on available workloads, see
[Workloads](./docs/workloads.md).

## Design History

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
