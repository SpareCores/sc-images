# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### A Production Workload

Transactions are synthetic proxies: deliberately balanced mixes of PostgreSQL
subsystems. See [Workloads](./workloads.md) for details.

This benchmark does *not* predict the specific performance of any given
application. It instead gives a general sense of the relative RDBMS (Relational
Database Management System) performance expected of a server type.

### Disk I/O Speed

The scores do *not* measure storage I/O (Input/Output) performance. Database
throughput usually hinges first on disk IOPS (Input/Output Operations Per
Second), then on bandwidth. In the cloud, that disk is almost always
network-attached block storage, provisioned independently of the server type and
entirely up to the user, so it says little about the server itself.

Deploying volumes with high-enough IOPS to never bottleneck across the more than
5,000 server types on [Navigator](https://sparecores.com/servers) would also be
prohibitively expensive. Because of this, we exclude disk speed from the
measurement and score the database engine's CPU and memory performance.

### Network Performance

The scores say nothing about a server's network throughput or latency. With a
remote client, both bandwidth and especially RTT (Round-Trip Time) between
client and server can dominate chatty workloads. Placing the client close to
the server is not enough on its own: occasional latency glitches still distort
lightweight-workload results. The workload is therefore designed so the
remaining RTT is a rounding error; see
[CPU-Heavy Transactions](#cpu-heavy-transactions).

### Uniform Data

Data distribution is uniform rather than Zipfian. The product catalog has
20,000 products, of which order items reference only the first 5,000, but
customer, order, and order-item values use `g % k` modular arithmetic rather
than a realistic power-law distribution with a few high-activity customers. See
[Workloads](./workloads.md) for details.

## Scope Decisions

Most published database benchmarks focus on storage, network throughput, a
single database operation, or one production workload. Our approach focuses on
the CPU and memory speed of the server instead: we keep the workload and the
PostgreSQL major version constant and vary the hardware across thousands of
server types.

### Minor Engine Versions

Only the PostgreSQL major version is fixed. DBaaS (Database as a Service)
providers may apply minor PostgreSQL upgrades automatically. This benchmark does
not check or enforce the version of a remote target; version selection and
pinning are deployment responsibilities. Standalone runs use whichever
PostgreSQL 18 minor release the `postgres:18` base image had when the image was
built.

### Instance Size Range

From small instances (e.g. 1 vCPU and 2 GiB of RAM) to large nodes with hundreds
of vCPUs, the same workload must produce meaningful, comparable numbers.

HammerDB TPROC-C ([HammerDB's transaction processing
workload](https://www.hammerdb.com/docs/)) and similar suites size their
datasets by warehouse count, and `pgbench` by scale factor. Neither sizing
scheme can fulfill this purpose. Because of this, we chose a fixed-size
workload that can run on every server type on
[Navigator](https://sparecores.com/servers) with at least 2 GiB of RAM, with
larger servers being taxed by concurrency rather than workload size.

### Engine Configuration

The vendor tunes the managed engine, and the harness cannot assume superuser
access or GUC ([Grand Unified
Configuration](https://www.postgresql.org/docs/current/config-setting.html))
control on DBaaS. For this benchmark, we deliberately do not tune the DBaaS
engine: the tuning of the managed service is part of what is being measured, so
the score reflects the vendor's configuration.

### JIT and Parallel Query

For `pgbench_ro`, the harness disables PostgreSQL's JIT ([Just-in-Time
Compilation](https://www.postgresql.org/docs/current/jit.html)) and [parallel
query](https://www.postgresql.org/docs/current/parallel-query.html), and sets
`work_mem` to `64MB`; see [Workload Settings](./usage.md#workload-settings).
These settings are not applied to `pgbench_tpcb`.

This benchmark measures raw engine and CPU behavior, *not* JIT compilation
variance or Gather scalability. Those are treated as a separate testing axis.

### A Single Monolithic Statement

One `SELECT` whose eight query blocks are CTEs ([Common Table
Expressions](https://www.postgresql.org/docs/current/queries-with.html))
combined with `UNION ALL` is deliberate: it keeps one `pgbench` transaction
equal to one network round trip. This makes the `pgbench_ro` workload resilient
to RTT simulated with `netem` ([Network
Emulator](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html));
see [Latency and pipelining](../CHANGELOG.md#latency-and-pipelining) for the
experiments. The tradeoff is that per-block planner GUCs (e.g. forcing Merge
Join specifically) are not possible without affecting every block.

### Pre-Calibrated Weights

The time shares of the eight query blocks in the `pgbench_ro` transaction were
manually calibrated and remain fixed for all runs. See the
[block timings](../CHANGELOG.md#current-pgbench_ro-workload-v2-subsystem-rebalance)
and the [recalibration procedure](../CHANGELOG.md#recalibration-procedure) for
details.

## Design Constraints

With the help of the [benchANT](https://benchant.com) team, over many iterations
with different tools and configurations, we identified the following core
principles.

### Memory-Fit, Small Dataset

This benchmark is designed to use a small dataset of about 303 MiB. On the
smallest nodes, the dataset does not fit in memory, which can add some disk
overhead, so production runs this benchmark only on servers with at least
2 GiB of RAM, as reported by the vendor. There, the dataset is meant to stay in
memory (`shared_buffers` plus the OS page cache), so that the disk is not read
again after warmup. Because the dataset is small by design, large instances are
exercised through concurrency rather than data volume.

### Read-Only Workload

The measured `pgbench_ro` transaction writes no data, so in the steady state of
this benchmark it generates no WAL ([Write-Ahead
Logging](https://www.postgresql.org/docs/current/wal-intro.html)) records.

### CPU-Heavy Transactions

Each transaction takes about 70 to 100 ms of server work per connection. With 1
to 5 ms of RTT against roughly 100 ms of server work, network time stays a small
fraction of each transaction and cannot dominate. Experiments showed that
lightweight read-only transactions are sensitive to network delay, so the
default workload uses a heavier cached transaction; see [Latency and
pipelining](../CHANGELOG.md#latency-and-pipelining) for the results.
