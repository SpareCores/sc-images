# benchmark-pgbench-postgres

This `pgbench`-driven PostgreSQL benchmark client measures how well cloud
servers handle real PostgreSQL database workloads using `pgbench` against a
local or remote PostgreSQL 18 instance. It is designed to be agnostic to whether
the client and the database are hosted on the same machine or separately.

This benchmark reports throughput and concurrency behavior, typically in
transactions per minute and across varying client counts, so users can see both
peak performance and how scalability changes with load.

## Purpose

Our [Navigator](https://sparecores.com/servers) project publishes empirical
performance measurements for more than 5,000 cloud server types. It needed an
RDBMS ([Relational Database Management
System](https://en.wikipedia.org/wiki/Relational_database)) benchmark that
scales beyond 32 vCPUs.

In many other benchmarks, the missing piece is the lack of actual, proper
database measurements - PassMark database operations don't scale to larger
instances, Redis is not relational, and raw CPU speed and memory bandwidth are
proxies rather than database workloads.

The same benchmark client measures two deployment models:

- **IaaS (Infrastructure as a Service):** Self-hosted PostgreSQL, with the
  client and database on the same node.
- **DBaaS (Database as a Service):** Provider-managed PostgreSQL, with a
  separate client VM; the provider provisions, manages, and tunes the database
  engine.

The benchmark reports a headline score in TPM (transactions per minute) and a
concurrency profile showing throughput at different client counts. These results
support comparisons of self-managed and managed PostgreSQL on similar hardware.

## Limitations

Most published database benchmarks focus more narrowly on storage, network
throughput, a single database operation, or a specific production workload. This
benchmark instead provides a controlled, comparable measure of PostgreSQL server
performance across cloud infrastructure. As such, it is
[distinct](./docs/limitations.md#disclaimer-what-this-benchmark-does-not-deliver)
from most typical benchmarks. Because of this, disk-speed scoring, minor-version
pinning where DBaaS providers don’t allow it, and direct engine-config control
had to be [excluded](./docs/limitations.md#exclusions) to ensure it actually
measures what we set out for it to do.

As detailed in our [design
constraints](./docs/limitations.md#design-constraints), `pgbench_ro` uses a
small dataset that fits entirely into memory, so disk I/O does not influence
results. Its workload is read-only, avoiding WAL, checkpoints, and database
writes. Transactions are intentionally CPU-heavy and run for more than 100 ms,
minimizing the impact of network latency and ensuring this benchmark primarily
measures PostgreSQL server CPU and memory performance.

Operational runs tune IaaS PostgreSQL with `pgtune`, leave DBaaS and host OS
settings unchanged, and place client and server in the same zone over private
networking. This is because we aim not to measure the tuning of the managed
service or the host, but the actual performance of the measured servers. See
[operational details](./docs/limitations.md#operational-details) for more.

For more information on this benchmark's limits, see
[Limitations](./docs/limitations.md).

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

- [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb) -
  [`pgbench`](https://www.postgresql.org/docs/current/pgbench.html)'s built-in
  script with a standard schema
- [`pgbench_ro`](./docs/workloads.md#pgbench_ro) - our custom-built, read-only
  PostgreSQL benchmark sized to fit in `shared_buffers`, with a monolithic SQL
  script that runs several blocks, each touching a different PostgreSQL
  subsystem.

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
