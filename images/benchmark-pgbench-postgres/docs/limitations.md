# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### A Production Workload

Transactions are synthetic proxies; deliberately balanced mixes of PostgreSQL
subsystems. See [Workloads](./workloads.md) for details.

This benchmark does *not* predict the throughput of any specific application. It
instead gives a general sense of the relative RDBMS (Relational Database
Management System)
performance expected of a server type.

### Disk I/O Speed

Block storage is provisioned independently of the instance type in most clouds,
so it is not a property of the server being ranked. The scores do *not* measure
storage performance. See [Deliberate Exclusions](#deliberate-exclusions) for
details.

### Network Performance

The design actively minimizes RTT (Round-Trip Time) sensitivity, so the scores
say nothing about a server's network throughput or latency (we publish separate
benchmarks for this purpose).

### Other Limitations

Data distribution is uniform rather than [Zipfian](https://en.wikipedia.org/wiki/Zipf%27s_law):
although the product catalog has over 20,000 products, customer, order, and
order-item values use `g % k` modular arithmetic rather than a realistic
power-law distribution with a few high-activity customers. See
[Workloads](./workloads.md) for details.

For `pgbench_ro`, the harness disables PostgreSQL's JIT ([Just-in-Time
Compilation](https://www.postgresql.org/docs/current/jit.html)) and [parallel
query](https://www.postgresql.org/docs/current/parallel-query.html), and sets
`work_mem` to `64MB`. These settings are not applied to `pgbench_tpcb`.

The dataset is small by design, so large instances are exercised through
concurrency rather than data volume.

## Exclusions

Most published database benchmarks compare *database engines*, engine versions,
or config tuning on fixed hardware. Our approach is the inverse: keep the engine
constant and vary the hardware across thousands of server types. This inversion
is the design's primary aim.

### External Limitations

#### Minor Engine Versions

Many DBaaS (Database as a Service) providers we benchmark do not allow pinning
the minor engine version (minor upgrades are applied automatically), so only
the *major* PostgreSQL version is fixed across IaaS (Infrastructure as a
Service) and DBaaS runs.

### Deliberate Exclusions

#### Disk Speed

Database throughput usually hinges first on disk IOPS (Input/Output operations
per second), then on bandwidth. In the cloud, that disk is almost always
network-attached block storage, provisioned independently of the server type and
entirely up to the user. As such, it says little about the server
itself.

Deploying volumes with high-enough IOPS to never bottleneck across a
~5,000-server fleet would also be prohibitively expensive. Because of this, we
exclude disk speed from the measurement and score the database engine's CPU and
memory performance.

#### RTT

With a remote client, both bandwidth and especially latency between client and
server can dominate chatty workloads. To avoid this, we minimize RTT via
placement, then design the workload so the remaining RTT is a rounding error.

### Design Considerations

#### Massive Workload Size Range

From the smallest cloud server instances with a single vCPU and mere megabytes
of memory to industrial-scale machines with thousands of vCPUs and terabytes of
memory, the same workload must produce meaningful, comparable numbers.
  
Available warehouse and scale-factor sizing schemes can't quite fulfill this
purpose. Because of this, we chose a fixed-size workload that could run on all
servers of the fleet, with larger servers being taxed by concurrency rather than
workload size.

#### Engine Config

Some DBaaS providers allow for config control, while others forbid it. In the
latter case, the vendor tunes the managed engine, so the
harness cannot assume superuser access or GUC ([Grand Unified
Configuration](https://www.postgresql.org/docs/current/config-setting.html))
control. For this benchmark, we deliberately do not tune the DBaaS engine, so the score reflects the engine's default behavior.

### Deliberately Out of Scope

The following are deliberately left out of scope, either because they detract
from this benchmark's overall purpose, or because they go against some other
design consideration.

#### `jit` and `max_parallel_workers_per_gather` Stay Off, Matching the Original Design's Rationale

This benchmark measures raw engine/CPU behavior,
*not* LLVM ([Low Level Virtual Machine](https://llvm.org/)) JIT jitter or Gather scalability. Those are treated as a separate
testing axis.

#### A Single Monolithic Statement

One `SELECT` with 8 CTEs, one `UNION ALL`
is deliberate: it keeps one `pgbench` transaction equal to one network round
trip. This makes the cached-RO redesign resilient to RTT simulated with
`netem` (Network Emulator; [documentation](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html));
see [Latency and pipelining](../CHANGELOG.md#latency-and-pipelining) for the
experiments. The tradeoff is that per-block planner GUCs (e.g. forcing Merge
Join specifically) aren't possible without affecting every block.

#### Pre-Calibrated Weights

Weights are calibrated on one local Docker `postgres:18` instance.
They are, however, not recalibrated automatically on every run.
See [Recalibration procedure](../CHANGELOG.md#recalibration-procedure) for details.

## Design Constraints

With the help of the [benchANT](https://benchant.com) team, over many iterations
with different tools and configs, we identified the following core principles.

### Memory-Fit, Small Dataset

This benchmark uses a small dataset of ~260–320 MB, stored comfortably in
`shared_buffers`, even on the smallest instances. After warmup, the disk is
not read again.

### Read-Only Workload

No WAL ([Write-Ahead Logging](https://www.postgresql.org/docs/current/wal-intro.html)), no
checkpoints, and no disk-write paths for `pgbench_ro`.

### CPU-Heavy Transactions

100+ ms of server work per transaction per
connection. This minimizes network round-trip time to ~0.2–4% of the total
service time and prevents it from dominating. This is [a necessary
constraint](#a-necessary-constraint), evidenced by our lab measurements.

### A Necessary Constraint

Experiments showed that lightweight read-only transactions are sensitive to
network delay, so the default workload uses a heavier cached transaction; see
[Latency and pipelining](../CHANGELOG.md#latency-and-pipelining) for the results.

## Operational Details

This benchmark can be used for the following production runs:

### IaaS Server Tuning

When `SC_DB_HOST` is unset, the benchmark starts PostgreSQL in the same
container and generates server settings with
[pgtune](https://pgtune.leopard.in.ua/) using the host's RAM and CPU count.
Remote servers specified with `SC_DB_HOST` are not tuned, as the benchmark is measuring the vendor's managed service tuning.

The pgtune form defaults are:

- Linux
- PostgreSQL 18
- web application
- SSD storage
- mid-RAM database size
- automatic connection count

The benchmark then applies these local-server overrides:

- `synchronous_commit` is `off` when `SC_DURABILITY=async`, and `on` otherwise.
- `max_connections` is raised to at least 3,122: 3,072 client connections plus
  50 reserved connections.

### No DBaaS Tuning

The vendor-managed config is left untouched by design, as the tuning of the
managed service is part of what is being measured.

### No OS-Level Tuning

We don't use `sysctl` or other host tweaks.

Tuning works on the container-level only. The server runs privileged with the
following:

- host networking
- `seccomp=unconfined`
- high `nofile` and unlimited `memlock` ulimits (unlocking huge pages and
  `io_uring`)
- `postgres` process at `nice -n -20`

### Topology

Client and server VMs are deployed in the same availability zone of the same
region, talking over private VPC (Virtual Private Cloud) addresses to minimize
RTT.

**Note**: This alone is insufficient. Occasional latency glitches still distort
lightweight-workload results even when deployed in the same zone. Because of
this the workload itself must be RTT-tolerant.

### Timing

This benchmark uses the following default timing settings:

- 120 s warmup (once)
- 60 s settle between concurrency rungs
- 300 s measurement per rung

See [Key Environment Variables](./usage.md#key-environment-variables) for
configuration options.
