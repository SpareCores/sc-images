# benchmark-pgbench-postgres

This `pgbench`-driven PostgreSQL benchmark client for the Spare Cores fleet runs against a `postgres:18` server over the network (`SC_DB_HOST`). It simulates how a real application communicates with a managed or remote database instead of hosting the client and the server together.
# Purpose

[Spare Cores](https://sparecores.com) monitors and publishes empirical performance data for over 5,000 cloud server types as part of the [Navigator](https://sparecores.com/servers) project. Among the measured metrics are the following and more:
- Memory bandwidth
- OpenSSL speed
- Compression algorithms
- Redis and static web-serving throughput
- LLM inference speed

In many other benchmarks, the missing piece is the lack of actual, proper database measurements - a gap this benchmark closes by scoring how how cloud servers perform under [Relational Database Management System](https://en.wikipedia.org/wiki/Relational_database) (RDBMS) workloads.

In this benchmark, two deployment models are measured with the same client:

- **Infrastructure as a Service (IaaS)**: self-hosted PostgreSQL on a cloud VM, driven by a separate client VM.
- **Database as a Service (DBaaS)**: the cloud vendor's managed PostgreSQL offering - similar hardware, but the vendor provisions, manages, and tunes the engine.

This benchmark helps users compare same-hardware, self-managed, and managed instances through a single, comparable headline score in TPM, and a concurrency profile that measures throughput at various numbers of connected clients.
# Limitations
## Disclaimer: what this benchmark does *NOT* deliver

- **A Production workload:**
	- Transactions are synthetic proxies; deliberately balanced mixes of PostgreSQL subsystems. See [Workloads](#workloads) for details.
	- The benchmark does *not* predict the throughput of any specific application.
	- It instead gives a general sense of the relative RDBMS performance expected of a server type.
- **Disk I/O speed:**
	- Block storage is provisioned independently of the instance type in most clouds, so it is not a property of the server being ranked. The scores do *not* measure storage performance. See [Deliberate Exclusions](#exclusions) for details.
- **Network performance:**
	- The design actively minimizes Round-Trip Time (RTT) sensitivity, so the scores say nothing about a server's network throughput or latency (we publish separate benchmarks for this purpose).
- **Other limitations**:
	- Data distribution in the database test tables is uniform rather than [Zipfian](https://en.wikipedia.org/wiki/Zipf%27s_law). See [](#workloads) for details.
	- PostgreSQL's [Just-in-Time Compilation](https://www.postgresql.org/docs/current/jit.html) (JIT) and [Parallel Query](https://www.postgresql.org/docs/current/parallel-query.html) features are disabled to measure raw engine and CPU behavior.
	- The dataset is small by design, so large instances are exercised through concurrency rather than data volume<sup>[1](Supplemental.md#1)</sup>.
## Exclusions

Most published database benchmarks compare *database engines*, engine versions, or config tuning on fixed hardware. Our approach is the inverse: keep the engine constant and vary the hardware across thousands of server types. This inversion is the design's primary drive. 

- ***External Limitations:***
	- **Minor Engine Versions:** Many DBaaS providers we benchmark do not allow pinning the minor engine version (minor upgrades are applied automatically), so only the *major* PostgreSQL version is fixed across IaaS and DBaaS runs.
- ***Deliberate Exclusions:***
	- **Disk Speed:** Database throughput usually hinges on the underlying disk (Input/Output operations per second (IOPS) first, bandwidth second). In the cloud, that disk is almost always network-attached block storage, provisioned independently of the server type and entirely up to the user. As such, it says little about the server itself. Deploying volumes with high-enough IOPS to never bottleneck across a ~5,000-server fleet would also be prohibitively expensive. Because of this, we exclude disk speed from the measurement and score the database engine's CPU and memory performance.
	- **RTT:** With a remote client, both bandwidth and especially latency between client and server can dominate chatty workloads. To avoid this, we minimize RTT via placement, then design the workload so the remaining RTT is a rounding error.
- ***Design Considerations:***
	- **Massive workload size range:** From the smallest cloud server instances with a singe vCPU and mere megabytes of memory to industrial-scale machines with thousands of vCPUs and terabytes of memory, the same workload must produce meaningful, comparable numbers.
	  Available warehouse and scale-factor sizing schemes can't quite fulfill this purpose. Because of this, we chose a fixed-size workload that could run on all servers of the fleet, with larger servers being taxed by concurrency rather than workload size.
	- **Engine config:** DBaaS forbids config control. The vendor tunes the managed engine, so the harness cannot assume superuser access or [Grand Unified Configuration](https://www.postgresql.org/docs/current/config-setting.html) (GUC) control.
## Design Constraints

With the help of the [benchANT](https://benchant.com) team, over many iterations with different tools and configs, we identified the following core principles:
1. **Memory-fit, small dataset:** ~260–320 MB, stored comfortably in `shared_buffers`, even on the smallest instances. After warmup, the disk is not read again.
2. **Read-only workload:** No [Write-Ahead Logging](https://www.postgresql.org/docs/current/wal-intro.html) (WAL), no checkpoints, and no disk-write paths.
3. **CPU-heavy transactions:** 100+ ms of server work per transaction per connection. This minimizes network round-trip time to ~0.2–4% of the total service time and prevents it from dominating<sup>[2](Supplemental.md#2)</sup>.
# Operational Details

The benchmark can be used for the following production runs:
- **IaaS server tuning**: Postgres GUCs are generated per host by [pgtune](https://pgtune.leopard.in.ua/).
	- Form defaults:
		- web application
		- SSD
		- Postgres 18 (with the host's RAM and CPU count)
		- `sc-inspector` orchestration
		- `synchronous_commit` is set by the durable/async task variant
		- `max_connections` is raised to cover the concurrency profile
- **No DBaaS tuning**: The vendor-managed config is left untouched by design, as the tuning of the managed service is part of what is being measured.
- **No OS-level tuning**:
	- No `sysctl` or other host tweaks.
	- Container-level only. The server runs privileged with the following:
		- host networking
		- `seccomp=unconfined`
		- high `nofile` and unlimited `memlock` ulimits (unlocking huge pages and `io_uring`)
		- postgres process at `nice -n -20`
- **Topology**: Client and server VMs are deployed in the same availability zone of the same region, talking over private VPC addresses to minimize RTT<sup>[3](Supplemental.md#3)</sup>.
- **Timing**: See [Key Environment Variables](#key-environment-variables) for details.
	- 120 s warmup (once)
	- 60 s settle between concurrency rungs
	- 300 s measurement per rung
## Usage
### Run Script
To use the benchmark, run the following bash script:

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```
### Key Environment Variables

| Var | Meaning | Default |
| - | - | - |
| `SC_WORKLOAD` | `pgbench_ro` (cached CPU-heavy custom script) or `pgbench_tpcb` (built-in `tpcb-like`) | `pgbench_ro` |
| `SC_DB_HOST` / `SC_DB_PORT` / `SC_DB_USER` / `SC_DB_PASSWORD` | connection | — / `5432` / `postgres` / `postgres` |
| `SC_CPU_SCALE` | `pgbench_ro` work multiplier (`-D scale=N`) | `1` |
| `SC_SCALEFACTOR(S)` | `pgbench_tpcb` `-i -s` size(s) | `65` |
| `SC_RUN_SECONDS` / `SC_WARMUP_SECONDS` / `SC_SETTLE_SECONDS` | measurement/warmup timing | `300` / `120` / `60` |
See `benchmark.py` docstring for the full list.

The script outputs one JSON document if prompted with stdout (`benchmark: pgbench_postgres`), with the following information:
- per-concurrency `profile` array
- headline `score` in transactions/minute (TPM)
	- `pgbench_ro` reports TPM only
	- no TPS
	- forces a fixed concurrency profile `{1, V/2, V, 2·V}` instead of `pgbench_tpcb`'s geometric search

***
## Workloads

### `pgbench_tpcb`

This workload is pgbench's built-in `tpcb-like` script (`-b tpcb-like`) with a standard `pgbench -i -s N` schema.
Standard TPC-B-style OLTP mix (mostly-write, network- and lock-sensitive). See the [official documentation](https://www.postgresql.org/docs/current/pgbench.html) for details.
### `pgbench_ro`

This schema uses a cached CPU-heavy SQL workload.

A custom, read-only PostgreSQL benchmark sized to fit in `shared_buffers`, so the benchmark is dominated by CPU work (parse, plan, execute, join, aggregate, text/JSON/array processing) rather than disk I/O.
It creates the following test data, taking up roughly 260–320 MB in memory:
- 20k products
- 50k customers
- 250k orders
- 750k order items

This benchmark can be run via the `pgbench -D scale=N -f ro_cpu_txn.sql` command.
- `-D scale=N` linearly scales the row-count knobs inside the transaction (wider slices, bigger joins) without touching the underlying dataset, so a single fixed schema can represent a range of CPU intensities.
- It uses fixed concurrency points instead of a geometric search to work with the uniformity of the test database.

The transaction is intentionally a single SQL script that runs several blocks, each touching a different Postgres subsystem:
- `q_idx`: btree index scan + nested loop + window agg
- `q_hashjoin`: hash join + hash aggregate over a time slice
- `q_regex`: regex + `md5()`
- `q_fts`: full-text search via `tsvector`/GIN
- `q_array`: array containment with GIN
- `q_stats`: ordered-set/statistical aggregates
- `q_toast`: TOAST ([The Oversized-Attribute Storage Technique](https://www.postgresql.org/docs/current/storage-toast.html)) fetch/decompression
- `q_seqscan`: plain sequential scan + aggregate

At the end of the transaction, the script does the following:
- builds a single `md5(string_agg(...))` checksum
- unites all eight block outputs (`UNION ALL`)
- returns one final headline score for easy comparison

This prevents the planner from optimizing away the work, and makes it a realistic, read-only CPU benchmark, rather than a trivial constant-time query.