# benchmark-pgbench-postgres

This benchmark measures cloud server database performance with `pgbench`. In
standalone mode, the image starts a local PostgreSQL 18 server. In remote mode,
it connects to the configured host without checking or enforcing the remote
server version. The image can run on the same node as the database or on a
separate machine.

This benchmark reports throughput in TPM ([Transactions Per
Minute](https://en.wikipedia.org/wiki/Transactions_per_second)) and latency
at several client counts. Together, they show peak performance and how it scales
with load, for self-managed and managed PostgreSQL on the same hardware.

## Purpose

Our [Navigator](https://sparecores.com/servers) project publishes empirical
performance measurements for more than 5,000 cloud server types. We needed an
RDBMS ([Relational Database Management
System](https://en.wikipedia.org/wiki/Relational_database)) benchmark that
tracks relevant metrics directly rather than relying on proxies. Navigator
previously relied on stand-ins: PassMark database operations, which do not scale
to 32+ vCPUs; Redis, which is not relational; and raw CPU speed and memory
bandwidth, which are useful proxies but not database workloads.

The main design goal is one methodology that scales across instance sizes, from
small instances (e.g. 1 vCPU and 2 GiB of RAM) to large nodes with hundreds of
vCPUs.

This benchmark measures two deployment models:

- **IaaS ([Infrastructure as a Service](https://en.wikipedia.org/wiki/Infrastructure_as_a_service)):**
  Self-hosted PostgreSQL, with the client and database on the same node.
- **DBaaS ([Database as a Service](https://en.wikipedia.org/wiki/Cloud_database)):**
  Provider-managed PostgreSQL, with a separate client VM; the provider
  provisions, manages, and tunes the database engine.

## Limitations

These results compare PostgreSQL server CPU and memory performance, not
application throughput. The main limitations are:

- **Workload:** `pgbench_ro` uses synthetic transactions and uniform data; the
  results do not specifically predict any given application's throughput, but
  aim to provide meaningful comparisons across different server types. See the
  [workload
  limitations](./docs/limitations.md#disclaimer-what-this-benchmark-does-not-deliver).
- **Disk and network:** The score excludes disk I/O (Input/Output) performance
  and does not measure network throughput or latency. See [Disk I/O
  Speed](./docs/limitations.md#disk-io-speed) and [Network
  Performance](./docs/limitations.md#network-performance).
- **Memory:** The dataset takes about 303 MiB, and production runs this
  benchmark only on servers with at least 2 GiB of RAM. The image enforces no
  minimum, so on smaller hosts disk reads can influence the score. See
  [Memory-Fit, Small Dataset](./docs/limitations.md#memory-fit-small-dataset).
- **Engine tuning:** When `SC_DB_HOST` is unset, the image starts PostgreSQL
  locally and tunes it with `pgtune`. A remote `SC_DB_HOST` target is not tuned
  by this benchmark. See [Standalone Mode](./docs/usage.md#standalone-mode) for
  the tuning and
  [Engine Configuration](./docs/limitations.md#engine-configuration) for why
  managed engines are left untuned.

  **Note:** Only the PostgreSQL major version is fixed: minor versions may vary
  between builds of the image and on DBaaS providers that do not support
  pinning. See [Minor Engine
  Versions](./docs/limitations.md#minor-engine-versions) for details.

## Usage

Run the image via Docker. Without `SC_DB_HOST`, it starts a local PostgreSQL 18
server in the container (standalone mode):

```bash
docker run --rm ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

Do not limit memory with `--memory`: the local server is tuned for the host's
full memory. See [Standalone Mode](./docs/usage.md#standalone-mode).

Production runs the container privileged, which lets the harness start
`postgres` at a higher priority (`nice -n -20`); with a plain `docker run`, it
runs at normal priority. See [Production
Setup](./docs/usage.md#production-setup) before you compare your scores with
Navigator's.

Set `SC_DB_HOST` to benchmark a remote PostgreSQL server instead. A remote run
needs more setup: database privileges, a disposable benchmark database, and the
server's vCPU count. Follow [Remote Mode](./docs/usage.md#remote-mode) for more
details and settings.

Both modes need outbound HTTPS ([Hypertext Transfer Protocol
Secure](https://en.wikipedia.org/wiki/HTTPS)) access to the Spare Cores CDN
([Content Delivery Network](https://en.wikipedia.org/wiki/Content_delivery_network))
to download the dataset; the run fails if the CDN is unreachable.

A default run takes about 25 minutes on servers with 4 or more vCPUs: a 2-minute
warmup, 1-minute settle periods, and 5 minutes of measurement at each of four
client counts. See [Run
Duration](./docs/usage.md#run-duration) for the steps and how to change them.

### Settings

Common settings for `pgbench_ro` are listed here; see the [full
environment-variable reference](./docs/usage.md#environment-variables) for
all options and defaults:

- `SC_DB_HOST`, `SC_DB_PORT`, `SC_DB_USER`, and `SC_DB_PASSWORD` configure a
  remote connection; leave `SC_DB_PORT` and `SC_DB_USER` unset in standalone
  mode. `SC_DB_SSLMODE` sets the SSL ([Secure Sockets
  Layer](https://www.postgresql.org/docs/current/ssl-tcp.html)) mode
  for the dataset restore and dump connections only.
- `SC_DB_VCPUS` sets the database vCPU count used to derive concurrency points
  and the local server settings in standalone mode. Set it to the database
  server's vCPU count for remote targets; the default is the client host's
  logical CPU count.
- `SC_CPU_SCALE` changes `pgbench_ro` transaction work.
- `SC_RUN_SECONDS`, `SC_WARMUP_SECONDS`, and `SC_SETTLE_SECONDS` control
  measurement and warmup timing.

### Results

The process prints nothing until the run ends, then prints one JSON object to
stdout; a default run can stay silent for about 25 minutes. Errors go to
stderr. In standalone mode, the PostgreSQL server log is written to
`/tmp/pg-server.log` inside the container, so drop `--rm` if you need it after
the run. Interpret the JSON object's main fields as follows:

- `score` is the highest TPM result; `score_unit` is `tpm`, and
  `peak_concurrency` is the client count for that result.
- Each `sizes[]` entry represents a workload size and has its own score and
  `profile`. `pgbench_ro` entries identify `cpu_scale`; `pgbench_tpcb` entries
  identify `scalefactor`.
- Each `profile[]` entry is one concurrency measurement. `concurrency` is the
  client count and `jobs` is the worker count. For `pgbench_ro`, `V` is the
  database vCPU count; it uses fixed points `{1, V/2, V, 2·V}` (`V/2` rounded
  down) and caps worker jobs at 32. For how `pgbench_tpcb` picks its client
  counts, see
  [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb).
- `latency_ms` contains sampled p50, p95, and p99 (50th, 95th, and 99th
  percentile) latency and an average, in milliseconds. The harness samples 1% of
  transaction latency logs.
- `schema_gib` (`0.17`) in `pgbench_ro` output is a legacy constant, not the
  dataset size; the restored database is about 303 MiB.

## Workloads

The image supports the following workloads with different transaction patterns:

- [`pgbench_ro`](./docs/workloads.md#pgbench_ro) (default) runs a custom,
  read-only transaction over a fixed schema, spreading work across eight
  PostgreSQL subsystems. `SC_CPU_SCALE` changes transaction work without
  resizing the schema.
- [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb) runs
  [`pgbench`](https://www.postgresql.org/docs/current/pgbench.html)'s built-in
  `tpcb-like` transaction mix against a schema initialized at one or more scale
  factors. It is disk-limited and not used in production.

For more information on available workloads, see
[Workloads](./docs/workloads.md).

## Design History

We first compared `sysbench`, HammerDB TPROC-C ([HammerDB's transaction
processing workload](https://www.hammerdb.com/docs/)), BenchBase, and
`pgbench`. The experiments showed that write-heavy workloads got bottlenecked by
storage and WAL
([Write-Ahead Logging](https://www.postgresql.org/docs/current/wal-intro.html)).
On `tmpfs`, write-heavy OLTP ([Online Transaction
Processing](https://en.wikipedia.org/wiki/Online_transaction_processing))
results improved by 10-25%, showing how much disk behavior could influence a
score. But `tmpfs` was unavailable for DBaaS, and sizing by warehouse count
(TPROC-C's dataset-size unit) or scale factor could not cover the range from
1 vCPU to hundreds. `pgbench` looked promising as a simpler, more scalable basis
for a cross-provider benchmark.

Early `pgbench -S` runs were dominated by network latency, and profiling found
that the first custom transaction over-weighted regex work. A set of 21
PostgreSQL configuration experiments on a 32 vCPU host showed that tuning could
improve throughput by about 20%. The `pgbench_ro` redesign uses a fixed,
cache-resident schema and spreads CPU work across PostgreSQL subsystems.
`pgbench_tpcb` is kept as a conventional reference workload, but it is
disk-limited and unused in production.

See the [changelog](./CHANGELOG.md) for experiment results, validation, and
calibration details.

## References

See the full list of [references](./docs/references.md) used in this
documentation.
