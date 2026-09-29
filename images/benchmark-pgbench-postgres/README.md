# benchmark-pgbench-postgres

This `pgbench`-driven PostgreSQL benchmark client measures how well cloud servers handle real PostgreSQL database workloads using `pgbench` against a remote PostgreSQL 18 instance. It is designed to be agnostic to whether the client and the database are hosted on the same machine or separately.

This benchmark reports throughput and concurrency behavior, typically in transactions per minute and across varying client counts, so users can see both peak performance and how scalability changes with load.

## Purpose

[Spare Cores](https://sparecores.com) monitors and publishes empirical performance data for over 5,000 cloud server types as part of our [Navigator](https://sparecores.com/servers) project. We needed a proper RBDMS (Relational Database Management System) benchmark that tracks more relevant metrics than relying on proxies like our prior PassMark database operations, Redis, and other related measurements such as raw CPU speed or memory bandwidth.

For more information on the project’s intent and motivation, see [Purpose](./docs/purpose.md).

## Limitations

Most published database benchmarks focus more narrowly on storage, network throughput, a single database operation, or a specific production workload. This benchmark instead provides a controlled, comparable measure of PostgreSQL server performance across cloud infrastructure. As such, it is [distinct](./docs/limitations.md#disclaimer-what-this-benchmark-does-not-deliver) from most typical benchmarks. Because of this, disk-speed scoring, minor-version pinning where DBaaS providers don’t allow it, and direct engine-config control had to be [excluded](./docs/limitations.md#exclusions) to ensure it actually measures what we set out for it to do.

As detailed in our [design constraints](./docs/limitations.md#design-constraints), `benchmark-pgbench-postgres` uses a small dataset that fits entirely into memory, so disk I/O does not influence results. Its workload is read-only, avoiding WAL, checkpoints, and database writes. Transactions are intentionally CPU-heavy and run for more than 100 ms, minimizing the impact of network latency and ensuring this benchmark primarily measures PostgreSQL server CPU and memory performance.

Operational runs tune IaaS PostgreSQL with pgtune, leave DBaaS and host OS settings unchanged, and place client and server in the same zone over private networking. This is because we aim not to measure the tuning of the managed service or the host, but the actual performance of the measured servers. See [operational details](./docs/limitations.md#operational-details) for more.

For more information on this benchmark's limits, see [Limitations](./docs/limitations.md).

## Usage

Most published database benchmarks focus more narrowly on storage, network throughput, a single database operation, or a specific production workload. This benchmark instead provides a controlled, comparable measure of PostgreSQL server performance across cloud infrastructure. As such, it is [distinct](./docs/limitations.md#disclaimer-what-this-benchmark-does-not-deliver) from most typical benchmarks. Because of this, some usual metrics had to be [excluded](./docs/limitations.md#exclusions) to ensure it actually measures what we set out for it to do.

For more information on how to use this benchmark, see [Usage](./docs/usage.md).

## Workloads

`benchmark-pgbench-postgres` can be used with the following workloads:

- [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb) - [`pgbench`](https://www.postgresql.org/docs/current/pgbench.html)'s built-in script with a standard schema
- [`pgbench_ro`](./docs/workloads.md#pgbench_ro) - our custom-built, read-only PostgreSQL benchmark sized to fit in `shared_buffers`, with a monolithic SQL script that runs several blocks, each touching a different PostgreSQL subsystem.

For more information on available workloads, see [Workloads](./docs/workloads.md).

## Design History

This benchmark was developed by comparing `pgbench` with other database suites, testing storage and configuration effects, and measuring the impact of network latency. Lightweight workloads proved too sensitive for RTT and often measured storage or network behavior instead of server performance.

The [first custom workload](./docs/design-history.md#v1-from-trivial-pgbench--s-to-a-cached-multi-query-script) introduced a cached, CPU-heavy transaction to reduce these effects. Profiling then exposed an imbalanced workload, data-generation issues, and missing coverage of important PostgreSQL execution paths.

The [current `pgbench_ro` workload](./docs/design-history.md#v2-rebalance-across-postgresql-subsystems) addresses these issues with a small dataset that fits in memory and several calibrated SQL blocks covering various PostgreSQL subsystems. It uses fixed concurrency levels and reports transactions per minute, while `pgbench_tpcb` remains available as a conventional TPC-B-style reference workload.

See the full [Design History](./docs/design-history.md) for the [investigation](./docs/design-history.md#what-was-actually-wrong-with-it), [calibration findings](./docs/design-history.md#calibration-gotchas-found-along-the-way-worth-knowing-before-touching-this-again), and [validation results](./docs/design-history.md#validation).

## References

See the full list of [References](./docs/references.md) used in this documentation.
