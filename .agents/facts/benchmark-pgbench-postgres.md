# Facts: benchmark-pgbench-postgres

Source commit: 2c78dc82fe6489547b5275c26508e8d423c7c753 (uncommitted changes)
Inputs: images/benchmark-pgbench-postgres ':(exclude)images/benchmark-pgbench-postgres/README.md' ':(exclude)images/benchmark-pgbench-postgres/docs' ':(exclude)images/benchmark-pgbench-postgres/CHANGELOG.md' images/benchmark-postgres-server ':(exclude)images/benchmark-postgres-server/README.md' ':(exclude)images/benchmark-postgres-server/docs' ':(exclude)images/benchmark-postgres-server/CHANGELOG.md' .agents/context/shared.md .agents/context/benchmark-pgbench-postgres.md

- Uncommitted Inputs path: `.agents/context/benchmark-pgbench-postgres.md` (new maintainer answers appended at the end). Uncommitted docs (`README.md`, `docs/usage.md`, `docs/limitations.md`) are excluded from Inputs and do not count.

## What is measured

- Two workloads, selected by `SC_WORKLOAD`: `pgbench_ro` (default) and `pgbench_tpcb`; any other value raises an error. (`benchmark.py:964-966`)
- Headline metric: TPM. `pgbench`'s `tps = ... (without initial connection time)` is multiplied by 60, rounded to an integer, and stored as both `tpm` and `score`; no TPS field is emitted. (`benchmark.py:78-80`, `benchmark.py:393-400`)
- Per size: score is the highest-TPM profile point; summary score is the best size. Ties keep the first entry (Python `max`). (`benchmark.py:936-955`, `benchmark.py:1068-1096`)
- What varies between runs (by design): server hardware; the concurrency points derived from `SC_DB_VCPUS`. (`benchmark.py:667-674`, `benchmark.py:998-1007`)
- What is held constant for `pgbench_ro`: one fixed schema (no scale factor), transaction script, `-D scale` work multiplier (default 1), `pgbench -M prepared -n`, and three database-level settings (`jit=off`, `work_mem='64MB'`, `max_parallel_workers_per_gather=0`). (`benchmark.py:490-523`, `benchmark.py:597-618`, `benchmark.py:721-724`, `ro_cpu_setup.sql:18-178`)
- Engine version: local server comes from `FROM postgres:18` (major pinned, minor floats with the base tag). Remote server version is not checked; it is only recorded in standalone mode via `collect_postgres_repro`. (`Dockerfile:4`, `benchmark.py:968-988`, `benchmark.py:1109-1126`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Is the PostgreSQL version held constant across runs?")
- Disk and network are excluded by design, not measured; the workload is meant to be CPU and memory bound on a cached dataset. (`ro_cpu_setup.sql:1-4`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is disk performance excluded?" and "Why is network performance excluded, and how is RTT handled?")
- Read-only: the measured `pgbench_ro` transaction is a single `SELECT`; writes happen only during dataset build or restore. "No WAL" applies to the measured steady state. (`ro_cpu_txn.sql:67-342`, `ro_cpu_setup.sql:18-178`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does a measured `pgbench_ro` run write WAL?")
- Purpose and gap filled: source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why does this benchmark exist?" and "What is the primary advantage versus other database benchmarks?"; `.agents/context/shared.md`, "What are these images for?"
- Design goal on instance range: source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the primary advantage versus other database benchmarks?" (smallest example there is 1 vCPU and 1 GB RAM; see Conflicts).
- Production server selection: `pgbench_ro` runs in production only on servers with at least 2 GB RAM; the code has no RAM check or guard. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?"; `benchmark.py:152-158`, `benchmark.py:963-1131`)
- `pgbench_tpcb` is kept but unused in production (disk limited); `pgbench_ro` is the only production PostgreSQL benchmark. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is `pgbench_tpcb` kept?")
- IaaS and DBaaS runs target the same underlying server types. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Do IaaS and DBaaS runs use the same hardware?")
- Design consultation with benchANT. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Who helped shape the design?")

## Workload

### Topology

- Empty `SC_DB_HOST`: standalone; the container starts its own `postgres` and connects to `127.0.0.1`. Non-empty: remote; the host is used as is. (`benchmark.py:968-988`)
- Code comment: client cost observed below 1% CPU and about 10 MB RSS on an 8-vCPU run, so a separate client VM was judged wasteful. (`benchmark.py:140-147`)
- Production: IaaS runs client and database on one node; DBaaS uses a separate client VM over private VPC, same AZ on AWS, possibly different zones of one region elsewhere. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Where is the benchmarking client run?" and "Are the DBaaS server and its benchmarking client always placed in the same availability zone and connected over private VPC addresses?")
- `SC_TOPOLOGY` is only a label: defaults to `single_vm` in standalone (`setdefault`, so a user value wins) and to `multi_vm` in the summary otherwise. (`benchmark.py:988`, `benchmark.py:1072`)

### Standalone server startup and tuning

- Command: `nice -n -20 docker-entrypoint.sh postgres -c listen_addresses=* <pgtune GUCs>`, with `POSTGRES_USER=postgres` and `POSTGRES_PASSWORD=<SC_DB_PASSWORD>`; server log goes to `/tmp/pg-server.log`. (`benchmark.py:149`, `benchmark.py:206-224`)
- Readiness: polls `psql -h 127.0.0.1 -p 5432 -U postgres` up to 120 s. Port 5432 and user `postgres` are hard-coded here, while later client calls use `SC_DB_PORT` and `SC_DB_USER`, so standalone needs those left at defaults. (`benchmark.py:227-241`, `benchmark.py:969-970`)
- Memory input: `MemTotal` from `/proc/meminfo` (host value; cgroup `--memory` limits not read), floored to whole GiB, minimum 1. (`benchmark.py:152-158`, `pgtune_leopard.py:375-388`)
- CPU input: `SC_DB_VCPUS`, default `os.cpu_count() or 2`. (`benchmark.py:981`, `benchmark.py:183`)
- pgtune port with site defaults: PostgreSQL 18, Linux, `web`, SSD, `mid_ram`, auto connections. (`pgtune_leopard.py:1-17`, `pgtune_leopard.py:33-39`)
- Resulting settings (M = floored memory, C = vCPUs):

| GUC | Value | Citation |
| --- | --- | --- |
| `max_connections` | pgtune 200, then raised to 3122 (ladder top 3072 plus 50) | `pgtune_leopard.py:117-121`, `benchmark.py:161-167`, `benchmark.py:187-192` |
| `shared_buffers` | M/4 | `pgtune_leopard.py:124-130` |
| `effective_cache_size` | 3M/4 | `pgtune_leopard.py:141-147` |
| `maintenance_work_mem` | M/16, capped at 8 GB | `pgtune_leopard.py:150-164` |
| `huge_pages` | `try` if `shared_buffers` at least 2 GB, else `off` | `pgtune_leopard.py:134-138` |
| `min_wal_size` / `max_wal_size` | 1 GB / 4 GB | `pgtune_leopard.py:167-180` |
| `checkpoint_completion_target` | 0.9 | `pgtune_leopard.py:182` |
| `wal_buffers` | 3% of `shared_buffers`, capped at 16 MB, bumped to 16 MB when between 14 and 16 MB, minimum 32 kB | `pgtune_leopard.py:184-193` |
| `default_statistics_target` | 100 | `pgtune_leopard.py:195` |
| `random_page_cost` | 1.1 | `pgtune_leopard.py:197-204` |
| `effective_io_concurrency` | 200 | `pgtune_leopard.py:206-208` |
| `work_mem` | (M minus `shared_buffers`) / ((200 + W) x 3), minimum 4 MB; W = C when C >= 4, else 1; uses 200 even after the 3122 override | `pgtune_leopard.py:225-244` |
| `max_worker_processes`, `max_parallel_workers` | C, only when C >= 4 | `pgtune_leopard.py:210-218` |
| `max_parallel_workers_per_gather`, `max_parallel_maintenance_workers` | ceil(C/2) capped at 4, only when C >= 4 | `pgtune_leopard.py:211-223` |
| `jit` | `off` | `pgtune_leopard.py:246-248` |
| `wal_compression` | `lz4` | `pgtune_leopard.py:250-254` |
| `autovacuum_max_workers` | 4 if C >= 16, 5 if C >= 32, else not set | `pgtune_leopard.py:256-261` |
| `autovacuum_work_mem` | 2 GB when `maintenance_work_mem` >= 2 GB | `pgtune_leopard.py:263-266` |
| `io_method` | `io_uring` | `pgtune_leopard.py:268-275` |
| `synchronous_commit` | `off` if `SC_DURABILITY=async`, else `on` | `benchmark.py:186` |

- `pgtune_share_url` encodes memory and CPU but no `connectionNum`, so it reproduces pgtune's 200 connections, not the 3122 override. (`pgtune_leopard.py:56-69`, `benchmark.py:187-193`)
- Remote servers are never retuned by the harness; DBaaS tuning is part of what is measured. (`benchmark.py:979-988`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is the DBaaS engine not tuned?")
- Host-level flags (privileged, host networking, `seccomp=unconfined`, ulimits) come from orchestration; no `sysctl` changes. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does production apply OS-level tuning?"; `.agents/context/shared.md`, "Where is a production run defined?")

### Dataset preparation

- Every size: `CREATE DATABASE "<SC_PGBENCH_DB>"` with "already exists" tolerated. (`benchmark.py:776-799`, `benchmark.py:824`)
- CDN check: HTTP HEAD on `<SC_CDN_BASE_URL>/<filename>`; 403 or 404 means miss; network errors and other HTTP errors propagate and fail the run (both modes). (`db_dataset_cache.py:67-77`, `db_dataset_cache.py:309-312`)
- Dump filenames: RO `pgbench-ro-cpu-v1.sql.zst`; TPC-B `pgbench-init-sf<N>.sql.zst`. Key has no PostgreSQL version. (`db_dataset_cache.py:37-39`, `db_dataset_cache.py:370-385`)
- Hit: terminate other sessions on the benchmark DB, `DROP DATABASE IF EXISTS`, `CREATE DATABASE`, then `bash -c "curl -fsSL <url> | zstd -d | psql -v ON_ERROR_STOP=1"`. The pipeline has no `pipefail`, so only `psql`'s exit status is checked. (`db_dataset_cache.py:116-138`, `db_dataset_cache.py:255-284`)
- After restore, no explicit `ANALYZE` or `VACUUM` runs; the code only re-applies the RO database settings. `pg_dump` flags: `--no-owner --no-privileges --format=plain`. (`benchmark.py:840-854`, `db_dataset_cache.py:145-172`)
- Deferred by the maintainer and not for the docs: planner statistics and hint bits after a CDN restore; truncated CDN downloads. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Miss: build natively (RO: `psql -f ro_cpu_setup.sql` plus database settings; TPC-B: `pgbench -i -s <scale>`), then stream `pg_dump | zstd -T0 | curl` presigned POST if upload is enabled; upload errors are recorded, not fatal. (`benchmark.py:526-574`, `db_dataset_cache.py:181-252`, `db_dataset_cache.py:326-351`)
- Upload key prefix is the constant `sc-inspector/`, independent of `SC_CDN_BASE_URL`. (`db_dataset_cache.py:21`, `db_dataset_cache.py:32-34`, `db_dataset_cache.py:220-222`)
- Timeouts: init and restore 14,400 s per command; each `pgbench` call duration plus 600 s; readiness 120 s. (`benchmark.py:467`, `benchmark.py:549`, `benchmark.py:636`, `db_dataset_cache.py:80`, `benchmark.py:227`)

### `pgbench_ro` schema (`ro_cpu_setup.sql`)

- Tables and rows: `ro_cpu_product` 20,000; `ro_cpu_customer` 50,000; `ro_cpu_order` 250,000 (5 per customer); `ro_cpu_order_item` 750,000 (3 per order). (`ro_cpu_setup.sql:68-92`, `ro_cpu_setup.sql:94-106`, `ro_cpu_setup.sql:108-140`, `ro_cpu_setup.sql:142-151`)
- Order items reference only products 1 to 5,000 (cold catalog tail). (`ro_cpu_setup.sql:68-70`, `ro_cpu_setup.sql:147`)
- Status formula mixes in the order's sequence so each customer's 5 orders cover all 5 statuses (fixes the v1 bug). (`ro_cpu_setup.sql:108-122`)
- `spec_blob` is about 4.5 KB of concatenated md5 strings (140 x 32 chars) to force out-of-line TOAST. (`ro_cpu_setup.sql:32-35`, `ro_cpu_setup.sql:91`)
- `search_doc` is a stored generated `tsvector`. (`ro_cpu_setup.sql:54-56`)
- Indexes: PKs, 6 B-tree, 2 expression, GIN on `search_doc` and `tags`, BRIN on `ordered_at`. (`ro_cpu_setup.sql:153-169`)
- Ends with `ANALYZE` on all four tables and `GRANT SELECT ... TO PUBLIC`. (`ro_cpu_setup.sql:171-176`)
- Size: comment estimates 260 to 320 MB data plus indexes and says it fits in `shared_buffers`; maintainer reports `pg_database_size` of 303 MiB after a fresh import on PostgreSQL 18. (`ro_cpu_setup.sql:1-4`, `ro_cpu_setup.sql:16`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the size of the benchmarking database?")
- Memory fit by floored `MemTotal` (pgtune `shared_buffers` = M/4): M = 1 GiB gives 256 MB (below 303 MiB); M = 2 GiB gives 512 MB (above). Because `MemTotal` is floored, a VM whose `MemTotal` is slightly under 2 GiB is tuned as M = 1. (`benchmark.py:152-158`, `pgtune_leopard.py:124-130`, `pgtune_leopard.py:384`)
- Maintainer: below the 2 GB production minimum the dataset does not fully fit in `shared_buffers`, with possible disk overhead. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?")
- Data distribution is modular (`g % k`), a known simplification. (`ro_cpu_setup.sql:71-151`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Is the uniform data distribution intentional?")

### `pgbench_ro` transaction (`ro_cpu_txn.sql`)

- One `SELECT md5(string_agg(x, '|' ORDER BY x))` over a `WITH` of 25 CTEs; 8 tagged block CTEs (`q_idx`, `q_hashjoin`, `q_regex`, `q_fts`, `q_array`, `q_stats`, `q_toast`, `q_seqscan`) combined with 7 `UNION ALL` operators. (`ro_cpu_txn.sql:67-342`)
- Random inputs per transaction: customer 1 to 50,000; order 1 to 250,000; product 1 to 5,000; region, tag, and FTS term indices. (`ro_cpu_txn.sql:37-44`)
- `-D scale=N` multiplies: `regex_width` 3600, `hj_window_sec` 1500, `fts_lim` 40, `array_slice_width` 48000, `stats_width` 8000, `toast_n` 700. Fixed: `array_lim` 200, `q_idx`, `q_seqscan`. (`ro_cpu_txn.sql:46-65`)
- Scale ceiling: `hj_start_sec = random(0, 250000 - 1500 * scale)` has an empty range for `SC_CPU_SCALE` >= 167. Maintainer deferred an upper-bound guard and excluded it from the docs. (`ro_cpu_txn.sql:52-53`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Calibrated block times at scale 1 (comment, local Docker Postgres 18): `q_idx` 0.1, `q_hashjoin` 24, `q_regex` 15, `q_fts` 12, `q_array` 8, `q_stats` 13, `q_toast` 4, `q_seqscan` 2 ms; total about 78 ms; no block above about 30%. Whether weights were also calibrated on cloud servers is unknown and excluded from the docs. (`ro_cpu_txn.sql:10-18`, `ro_cpu_txn.sql:46-50`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Comment: variable sums are precomputed in `\set` because under `-M prepared` SQL-side `:a + :b` becomes `$N + $M` with unknown types. (`ro_cpu_txn.sql:54-57`)
- Generic-plan risk under `-M prepared`: maintainer considers it probably fine, to be checked later, and excludes it from the docs. (`benchmark.py:611-612`, `profile_v2_breakdown.sql:14-23`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Single-statement rationale: one transaction equals one round trip; trade-off is no per-block planner GUCs. (`ro_cpu_txn.sql:23-29`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is the `pgbench_ro` transaction a single statement?")
- Database-level settings via `ALTER DATABASE ... SET`: `jit=off`, `work_mem='64MB'`, `max_parallel_workers_per_gather=0`; applied during build and again after every prepare (CDN restores lose them). (`benchmark.py:490-523`, `benchmark.py:572-574`, `benchmark.py:850-854`; rationale source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why are JIT and parallel query disabled for `pgbench_ro`?")
- `profile_v2_breakdown.sql`: dev-only calibration helper, not copied into the image. (`profile_v2_breakdown.sql:1-3`, `Dockerfile:23`)

### `pgbench_tpcb`

- `pgbench -b tpcb-like` against `pgbench -i -s <scale>`; scales from `SC_SCALEFACTORS` (CSV) else `SC_SCALEFACTOR` (default 65); each scale is prepared and measured in turn. (`benchmark.py:526-551`, `benchmark.py:619-620`, `benchmark.py:721-728`, `benchmark.py:1024-1066`)
- No database settings are changed for TPC-B. (`benchmark.py:825-854`)

### `pgbench` invocation (both workloads)

- `pgbench -h -p -U -c <clients> -j <jobs> -T <seconds> -M prepared -n`; RO jobs = min(clients, 32), TPC-B jobs = clients. (`benchmark.py:42`, `benchmark.py:593-614`)
- Measured runs add `-P 5 --progress-timestamp` and `-l --log-prefix=... --sampling-rate=0.01`; warmup and settle runs add neither. (`benchmark.py:87`, `benchmark.py:623-632`, `benchmark.py:869-898`)
- A nonzero `pgbench` exit is accepted if a TPS line was parsed. (`benchmark.py:635-642`)

### Concurrency

- Ladder: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072. `rung(x)` picks the nearest rung, ties to the smaller. (`benchmark.py:51-76`, `benchmark.py:662-664`)
- RO default points: sorted set {1, V//2 (min 1), V, 2V}, V = `SC_DB_VCPUS`; `SC_PROFILE_VUS` replaces them; points above `SC_PROFILE_MAX_CLIENTS` (default: highest point) are dropped; search forced off. (`benchmark.py:667-669`, `benchmark.py:707-710`, `benchmark.py:996-1008`)
- Resulting RO points: V=1 gives 1, 2; V=2 gives 1, 2, 4; V=3 gives 1, 3, 6; V>=4 gives 4 distinct points. (`benchmark.py:667-669`)
- Fixed-profile rationale: source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why does `pgbench_ro` use a fixed concurrency profile?"
- Connection cap: `conn_cap = max_connections - 50` (from `SHOW max_connections` on the admin DB) caps max clients, hard max, and host anchors. (`benchmark.py:752-773`, `benchmark.py:1015-1020`)
- TPC-B per scale: `anchor_v = min(SC_DB_VCPUS, scale)`; anchors = {1, rung(anchor_v/4), rung(anchor_v/2), rung(anchor_v)}; `max_clients = min(scale, host_max_clients)`. `host_anchors` (and therefore `SC_PROFILE_VUS`) is not used to build TPC-B anchors; it only sets the default `SC_PROFILE_MAX_CLIENTS` and the summary `profile_vus`. (`benchmark.py:672-674`, `benchmark.py:698-718`, `benchmark.py:1004-1008`, `benchmark.py:1083`)
- `choose_concurrency_plan`: keeps anchors <= `max_clients`; with search on, appends every ladder rung above the highest kept anchor up to `max_clients`. (`benchmark.py:677-688`)
- Run loop (TPC-B with search): at a non-anchor point, stop if TPM < previous peak x (1 + `SC_PROFILE_IMPROVE_PCT`/100); if the last planned point is a non-anchor that clears the threshold, append the next rung up to `hard_max = min(SC_PROFILE_HARD_MAX_CLIENTS, scale, conn_cap)`. Anchor points never trigger either rule. (`benchmark.py:912-934`, `benchmark.py:1009-1012`, `benchmark.py:1033-1037`)
- What search actually adds (derived from the plan functions):
  - Default scale 65 with default `SC_PROFILE_MAX_CLIENTS`: never adds a point for any V; the plan equals the anchors, all points are anchors, so the run-loop rules never fire. Examples: V=16 gives 1, 4, 8, 16; V>=65 gives 1, 16, 32, 64. (`benchmark.py:672-718`, `benchmark.py:1008`)
  - Non-default scale with default `SC_PROFILE_MAX_CLIENTS`: when rung(min(V, scale)) > scale, the top anchor is dropped and the one ladder rung between the V/2 anchor and the scale is added as a search point. Examples: scale 57, V>=57 gives 1, 16, 32, 48 (anchor 64 dropped); scale 11, V>=11 gives 1, 3, 6, 8. The run-loop extension cannot add more, because `max_clients` and `hard_max` are then both capped by the scale. (`benchmark.py:677-688`, `benchmark.py:712-717`, `benchmark.py:924-933`, `benchmark.py:1033-1037`)
  - Raised `SC_PROFILE_MAX_CLIENTS` (below scale): the plan gets ladder rungs up to it, and the run loop can then extend past it up to `min(scale, SC_PROFILE_HARD_MAX_CLIENTS, conn_cap)` while each point improves by the threshold. Example: V=16, scale 65, `SC_PROFILE_MAX_CLIENTS=32` plans 1, 4, 8, 16, 24, 32, then may add 48 and 64. (`benchmark.py:677-688`, `benchmark.py:924-933`)
  - Raising `SC_PROFILE_MAX_CLIENTS` above the scale has no effect; TPC-B clients never exceed the scale factor (guard raises if the plan does). (`benchmark.py:712`, `benchmark.py:1038-1042`)

### Warmup and timing

- Per point: warmup or settle `pgbench` run at that point's client count, then a measured run of `SC_RUN_SECONDS`. First point gets `SC_WARMUP_SECONDS`; later points get `SC_SETTLE_SECONDS` when `SC_WARMUP_ONCE` is true, else the full warmup. A zero duration skips it. (`benchmark.py:863-898`, `benchmark.py:990-993`)
- Warmup state carries across TPC-B scales, so later scales only get settle runs. (`benchmark.py:1023`, `benchmark.py:1043-1063`)
- Default RO duration excluding preparation: V>=4 is 120 + 3 x 60 + 4 x 300 = 1500 s (25 min); V=2 or 3 is 1140 s (19 min); V=1 is 780 s (13 min). (`benchmark.py:667-669`, `benchmark.py:990-993`)
- Production timing example: the AWS `m9g.24xlarge` run is an actual production run recorded in `sc-inspector-data`. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Where does the `m9g.24xlarge` timing example come from?") At V=96 the code gives points 1, 48, 96, 192 and 1500 s of warmup plus measurement. (`benchmark.py:667-669`, `benchmark.py:990-993`)
- Production measurement window choice (5 min) is from history, not code; see CHANGELOG claims. (`CHANGELOG.md:147-182`)

### Latency

- Percentiles come from the sampled per-transaction logs: field 3 (microseconds) divided by 1000; linear-interpolated p50, p95, p99, mean, and sample count. (`benchmark.py:419-456`, `benchmark.py:647-658`)

## Parameters and defaults

| Name | Default | Meaning | Citation |
| --- | --- | --- | --- |
| `SC_WORKLOAD` | `pgbench_ro` | Workload selector (`pgbench_ro`, `pgbench_tpcb`) | `benchmark.py:964-966` |
| `SC_DB_HOST` | empty | Empty: standalone local server; set: remote host | `benchmark.py:968`, `benchmark.py:979-988` |
| `SC_DB_PORT` | `5432` | Client port (standalone server always listens on 5432) | `benchmark.py:969`, `benchmark.py:233` |
| `SC_DB_USER` | `postgres` | Client role (standalone server creates `postgres`) | `benchmark.py:970`, `benchmark.py:222` |
| `SC_DB_PASSWORD` | `postgres` | Password via `PGPASSWORD`; also standalone `POSTGRES_PASSWORD` | `benchmark.py:134-137`, `benchmark.py:221`, `benchmark.py:971` |
| `SC_DB_NAME` | `postgres` | Admin DB for `CREATE/DROP DATABASE`, `SHOW` queries | `benchmark.py:972`, `benchmark.py:1015-1016` |
| `SC_PGBENCH_DB` | `pgbench` | Benchmark DB; dropped and recreated on CDN restore | `benchmark.py:973`, `db_dataset_cache.py:255-284` |
| `SC_DB_SSLMODE` | `prefer` | Sets `PGSSLMODE` only for CDN restore and `pg_dump`, and `sslmode` for the psycopg drop/create connection (value `disable` is not passed there). Not applied to `pgbench`, `psql` setup and `SHOW` calls, or `collect_postgres_repro` | `db_dataset_cache.py:135-136`, `db_dataset_cache.py:153-155`, `db_dataset_cache.py:266-268`, `benchmark.py:134-137`, `benchmark.py:269-276` |
| `SC_CPU_SCALE` | `1` | RO `-D scale`; values >= 167 make the `random()` range empty | `benchmark.py:999`, `benchmark.py:618`, `ro_cpu_txn.sql:52-53` |
| `SC_SCALEFACTORS` | unset | CSV TPC-B scale factors (each min 1) | `benchmark.py:111-115`, `benchmark.py:725-727` |
| `SC_SCALEFACTOR` | `65` | TPC-B scale when `SC_SCALEFACTORS` empty | `benchmark.py:728` |
| `SC_PROFILE_VUS` | derived | RO: replaces points. TPC-B: not used for anchors; only seeds default max clients and summary `profile_vus` | `benchmark.py:1000-1008`, `benchmark.py:712-714` |
| `SC_PROFILE_SEARCH` | true for TPC-B; forced false for RO | Enables ladder tail and run-loop extension | `benchmark.py:995-997`, `benchmark.py:684-688`, `benchmark.py:912` |
| `SC_PROFILE_IMPROVE_PCT` | `5.0` | TPC-B search threshold, percent; summary shows `null` for RO | `benchmark.py:994`, `benchmark.py:1081` |
| `SC_PROFILE_MAX_CLIENTS` | highest host anchor | Plan cap; RO drops points above it; TPC-B cap is min(this, scale, `conn_cap`) | `benchmark.py:1008`, `benchmark.py:1018`, `benchmark.py:709`, `benchmark.py:712` |
| `SC_PROFILE_HARD_MAX_CLIENTS` | RO: highest anchor; TPC-B: 3072 | Run-loop extension ceiling; TPC-B also capped by scale and `conn_cap` | `benchmark.py:1009-1012`, `benchmark.py:1019`, `benchmark.py:1033-1037` |
| `SC_RUN_SECONDS` | `300` | Measured seconds per point | `benchmark.py:990` |
| `SC_WARMUP_SECONDS` | `120` | First (or every, if not warmup-once) warmup | `benchmark.py:991` |
| `SC_SETTLE_SECONDS` | `60` | Settle before later points | `benchmark.py:992` |
| `SC_WARMUP_ONCE` | `true` | Full warmup once, then settle | `benchmark.py:993` |
| `SC_DB_VCPUS` | `os.cpu_count() or 2` | Concurrency points; standalone pgtune CPU input | `benchmark.py:981`, `benchmark.py:998` |
| `SC_CLIENT_VCPUS` | `os.cpu_count() or 2` | Recorded only | `benchmark.py:1087` |
| `SC_DB_MEM_GIB` | standalone: detected (float GiB, unfloored); remote: unset gives `null` | Recorded only | `benchmark.py:987`, `benchmark.py:1088` |
| `SC_DURABILITY` | `durable` | Standalone: `async` sets `synchronous_commit=off`. Always echoed in summary, even for remote | `benchmark.py:980`, `benchmark.py:186`, `benchmark.py:1073` |
| `SC_TOPOLOGY` | standalone `single_vm`; else `multi_vm` | Recorded only | `benchmark.py:988`, `benchmark.py:1072` |
| `SC_CDN_BASE_URL` | `https://cdn.sparecores.net/sc-inspector` | Dump download base | `db_dataset_cache.py:42-48` |
| `SC_CDN_DATASET_POST_B64` | unset | Base64 JSON presigned POST (`url`, `fields`, `prefix`) | `db_dataset_cache.py:51-64` |
| `SC_CDN_UPLOAD` | `1` | `0`, `false`, `no` disable upload; upload also needs a valid POST config | `db_dataset_cache.py:291-294` |
| `TRACKER_QUIET` | `true` | Set in image for `resource-tracker` | `Dockerfile:25` |

- Boolean parsing (`env_bool`): `1`, `true`, `yes`, `on` (case-insensitive) are true; empty uses the default. (`benchmark.py:104-108`)
- Constants: `RO_MAX_JOBS=32`, `LATENCY_SAMPLE_RATE=0.01`, `MAX_CONNECTIONS_CLIENT_RESERVE=50`, `STORAGE_GIB=128`, `PGBENCH_RO_CPU_SCHEMA_GIB=0.17`, progress interval 5 s. (`benchmark.py:42`, `benchmark.py:47-49`, `benchmark.py:87`, `benchmark.py:624`, `benchmark.py:773`)

## Outputs and schema

- stdout: one JSON object (`indent=2`, `sort_keys=True`) printed after all measurements; exit code 0. Progress output is captured, and dataset-cache `logging.info` lines are not shown (no logging config), so stdout is silent until the end. (`benchmark.py:118-131`, `benchmark.py:1128-1131`, `db_dataset_cache.py:313`, `db_dataset_cache.py:326`)
- Errors go to stderr via exceptions; standalone server log stays in `/tmp/pg-server.log` inside the container. (`benchmark.py:149`, `benchmark.py:223`)
- Production: `resource-tracker` streams per-second resource metrics to Sentinel; metrics and raw JSON are not published. (`Dockerfile:22-26`; source: context, `.agents/context/shared.md`, "What wraps the main benchmarking process inside the container?" and "Are the resource-tracker metrics and raw JSON outputs published?")

### Top-level fields

| Field | Meaning / unit | Citation |
| --- | --- | --- |
| `benchmark` | `pgbench_postgres` | `benchmark.py:1070` |
| `workload`, `workload_kind` | workload name (both) | `benchmark.py:1071`, `benchmark.py:1107` |
| `topology`, `durability` | labels from env | `benchmark.py:1072-1073` |
| `synchronous_commit` | actual `SHOW` value | `benchmark.py:1074`, `benchmark.py:1015` |
| `max_connections`, `max_connections_client_cap` | actual `SHOW` value; minus 50 | `benchmark.py:1075-1076` |
| `run_seconds`, `warmup_seconds`, `settle_seconds`, `warmup_once` | configured timing | `benchmark.py:1077-1080` |
| `improve_pct` | TPC-B threshold or `null` | `benchmark.py:1081` |
| `profile_search`, `profile_vus`, `profile_max_clients`, `profile_hard_max_clients` | effective host-level settings after `conn_cap` | `benchmark.py:1082-1085` |
| `db_vcpus`, `client_vcpus`, `db_mem_gib` | recorded inputs | `benchmark.py:1086-1088` |
| `sizes` | list, see below | `benchmark.py:1089` |
| `peak_concurrency`, `score`, `score_unit` (`tpm`) | best size | `benchmark.py:1090-1092` |
| `latency_ms`, `latency_avg_ms`, `latency_stddev_ms` | best size's best point; ms | `benchmark.py:1093-1095` |
| RO only: `cpu_scale`, `peak_cpu_scale`, `schema_gib` (fixed 0.17) | | `benchmark.py:1097-1100` |
| TPC-B only: `scalefactors`, `peak_scalefactor`, `scalefactor` | | `benchmark.py:1101-1104` |
| `benchmark_image` | `ghcr.io/sparecores/benchmark-pgbench-postgres:main` (constant) | `benchmark.py:44`, `benchmark.py:1106` |
| Standalone only: `pg_image` (same constant), `storage_gib` (constant 128), `pgtune_share_url`, `postgres` | | `benchmark.py:45-47`, `benchmark.py:1109-1126` |

- `storage_gib`: describes the benchmarking environment, not a benchmark metric; in code it is the hard-coded `STORAGE_GIB = 128` (comment: the `sc-inspector` root volume size), never measured, emitted in every standalone run including a plain `docker run`. (`benchmark.py:45-47`, `benchmark.py:1113`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is `storage_gib`?")
- `schema_gib` is a leftover, not the dataset size (database is about 303 MiB). (`benchmark.py:48-49`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the size of the benchmarking database?")
- `postgres` object: `version`, `server_version`, `server_version_num`, `in_recovery`, `settings` (all GUCs, SHOW-style), `nondefault_settings` (per GUC: `setting`, `source`, `context`, `vartype`, `category`, optional `unit`, `setting_raw`, `short_desc`, `pending_restart`), `extensions`, `role_settings`, `requested_gucs`. Collected from the benchmark DB, so RO database-level settings appear with source `database`; omitted if collection fails. (`benchmark.py:254-390`, `benchmark.py:1117-1126`)

### `sizes[]` fields

- `dataset`, `profile`, `profile_vus` (anchors used for this size), `concurrency_plan` (final plan incl. run-loop additions), `profile_max_clients` (overwritten with the per-scale cap), `peak_concurrency`, `score`, `latency_ms`, `latency_avg_ms`, `latency_stddev_ms`, `stop_reason` (empty unless search stopped), `clients_capped_at_scale` (true for TPC-B, false for RO). RO adds `cpu_scale`; TPC-B adds `scalefactor`. (`benchmark.py:943-959`, `benchmark.py:1064-1065`)
- `dataset`: `dataset` (filename), `cdn_url`, `source` (`cdn` or `built`); built adds `uploaded` and maybe `upload_error`. (`db_dataset_cache.py:309-351`)

### `profile[]` fields

- `concurrency`, `jobs`, `anchor` (bool), `warmup_seconds` (warmup or settle length before it), `tpm`, `score`, `latency_avg_ms`, `latency_stddev_ms` (pgbench summary, ms), `tx_processed`, `tx_failed`, `initial_connection_ms`, `run_seconds` (wall clock of the measured call), `latency_ms` {`p50`, `p95`, `p99`, `avg`, `samples`}, optional `stop_reason`, `tpm_vs_final_peak_pct`. (`benchmark.py:393-416`, `benchmark.py:643-658`, `benchmark.py:903-910`, `benchmark.py:922`, `benchmark.py:938-942`)

## How to run

- Image: `ghcr.io/sparecores/benchmark-pgbench-postgres:main`. (`benchmark.py:44`; source: context, `.agents/context/shared.md`, "What is the public image name?")
- Entrypoint: `resource-tracker -- python3 /benchmark/benchmark.py`; no CMD or arguments; all configuration via env. (`Dockerfile:26`)
- Standalone minimal: `docker run --rm ghcr.io/sparecores/benchmark-pgbench-postgres:main` (no env required). (`benchmark.py:968-988`)
- `nice -n -20` in standalone needs a container allowed to raise priority (for example privileged); otherwise `nice` cannot apply it. (`benchmark.py:210-214`, `Dockerfile:6`)
- Remote minimal: set `SC_DB_HOST`, credentials, and `SC_DB_VCPUS` (default is the client's CPU count). (`benchmark.py:968-973`, `benchmark.py:998`)
- Remote privileges implied by code: `CREATE DATABASE`; `DROP DATABASE` of the benchmark DB and `pg_terminate_backend` on its sessions (CDN restore path); `ALTER DATABASE ... SET` (RO); `CREATE TABLE` or `pgbench -i` (build path). (`benchmark.py:500-502`, `benchmark.py:776-799`, `db_dataset_cache.py:279-284`, `ro_cpu_setup.sql:20-66`)
- Network: outbound HTTPS to `SC_CDN_BASE_URL` required in both modes. (`db_dataset_cache.py:67-77`, `db_dataset_cache.py:312`)
- Memory: no minimum enforced in code; production uses servers with at least 2 GB RAM. (`benchmark.py:152-158`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?")
- Image contents: `ca-certificates`, `curl`, `python3`, `python3-psycopg`, `zstd`, plus `benchmark.py`, `db_dataset_cache.py`, `pgtune_leopard.py`, `ro_cpu_setup.sql`, `ro_cpu_txn.sql` in `/benchmark/`. (`Dockerfile:14-23`)

## Dependencies and platforms

- `DEPENDS_ON`: `resource-tracker`. (`DEPENDS_ON:1`)
- Build arg `RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main-${ARCH}`; Dockerfile default `...:main`. (`BUILD_ARGS:1`, `Dockerfile:1-3`)
- Base image: `postgres:18`, running as root at build. (`Dockerfile:4-6`)
- Platforms: `amd64`, `arm64`. Build context: `images/benchmark-pgbench-postgres`. No `ZRAM` or `SCCACHE` files. (`PLATFORMS:1-2`, `CONTEXT:1`)
- Sibling `benchmark-postgres-server`: `postgres:18` plus `resource-tracker`, entrypoint `resource-tracker -- nice -n -20 docker-entrypoint.sh`, CMD `postgres`; no `PLATFORMS` or `CONTEXT` file; this image does not depend on it. (`images/benchmark-postgres-server/Dockerfile:1-9`, `images/benchmark-postgres-server/DEPENDS_ON:1`, `images/benchmark-postgres-server/BUILD_ARGS:1`)
- Family: this README is the PostgreSQL manual; the server image gets no separate methodology write-up. (source: context, `.agents/context/shared.md`, "Which folders share one manual?")
- Vendored or mirrored code that must stay in sync with `sc-inspector`: `pgtune_leopard.py`, `max_connections_for_vcpus`, `pg_guc_settings`, `collect_postgres_repro`, the ladder, and `PGBENCH_RO_CPU_SCHEMA_GIB`. (`pgtune_leopard.py:2`, `benchmark.py:48-51`, `benchmark.py:164`, `benchmark.py:179`, `benchmark.py:265`)
- `__pycache__/` in the image folder is Git-ignored and not part of the build inputs (`Dockerfile:23` copies named files only).

## Previous Review Findings

None found.

- The previous facts file (source commit 5dbe4999cdb350e2c7b0b3a68f9a033b682e4e3c) contained no bullets marked `(added in review)` and no Previous Review Findings section, so nothing was rechecked, kept, or removed.

## Open questions for maintainers

1. **Should the docs name 1 GB or 2 GB of RAM as the smallest instance the design targets?**
   - Why we ask: the design-goal answer gives "1 vCPU and 1 GB of RAM" as the small end, but production only runs `pgbench_ro` on servers with at least 2 GB, because the dataset does not fit in `shared_buffers` below that. The docs need one number, or both with a clear distinction (design goal vs. production minimum).
   - Code: `pgtune_leopard.py:124-130` (`shared_buffers` = memory / 4), `ro_cpu_setup.sql:16` (dataset estimate)
2. **On a nominal 2 GB server whose `MemTotal` is slightly below 2 GiB, does the dataset still fit in `shared_buffers`?**
   - Why we ask: the harness rounds `MemTotal` down to whole GiB before tuning. Kernels usually report a bit less than the nominal RAM, so a "2 GB" VM can be tuned as 1 GiB, giving 256 MB `shared_buffers`, below the 303 MiB database. That is the case the 2 GB production minimum is meant to avoid. Is this expected, and is the production minimum measured in nominal GB, GiB, or `MemTotal`?
   - Code: `benchmark.py:152-158`, `pgtune_leopard.py:375-388`, `pgtune_leopard.py:124-130`

### Conflicts

1. Smallest instance size (context vs. docs).
   - Context: design goal scales "from small instances (e.g. 1 vCPU and 1 GB of RAM)" (`.agents/context/benchmark-pgbench-postgres.md`, "What is the primary advantage versus other database benchmarks?"); production minimum is 2 GB RAM (`.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?").
   - Docs: the design goal is stated as "1 vCPU and 2 GB of RAM" (`README.md:25-27`, `docs/limitations.md:61-62`), which matches the production minimum but not the design-goal answer.
   - Code: enforces no minimum; at floored 1 GiB, `shared_buffers` is 256 MB. (`benchmark.py:152-158`, `pgtune_leopard.py:124-130`, `pgtune_leopard.py:384`)
2. `storage_gib` described as a measurement (context vs. code).
   - Context: "a measurement of the benchmarking environment" (`.agents/context/benchmark-pgbench-postgres.md`, "What is `storage_gib`?").
   - Code: hard-coded constant 128, never measured; emitted in every standalone run regardless of the real disk. (`benchmark.py:45-47`, `benchmark.py:1113`)
3. TPC-B search at default settings (docs vs. code).
   - Docs: search "adds points only when `SC_PROFILE_MAX_CLIENTS` is raised above the highest anchor" (`README.md:113-115`, `docs/usage.md:217`, `docs/usage.md:219`).
   - Code: at default scale 65 this holds. At other scale factors with default `SC_PROFILE_MAX_CLIENTS`, when rung(min(V, scale)) > scale the top anchor is dropped and one search rung is added (scale 57, V>=57: 1, 16, 32, 48). When raised, the run loop can extend past `SC_PROFILE_MAX_CLIENTS` up to min(scale, hard max, `conn_cap`); raising it above the scale does nothing. (`benchmark.py:677-688`, `benchmark.py:712-717`, `benchmark.py:924-933`, `benchmark.py:1033-1037`)
4. TPC-B anchors and `SC_PROFILE_VUS` (docs vs. code).
   - Docs: `SC_PROFILE_VUS` "replaces the default concurrency points" (`docs/usage.md:150`) and is "Comma-separated concurrency anchors" (`docs/usage.md:216`); TPC-B anchors are {1, V/4, V/2, V} with V the database vCPU count (`README.md:111-114`).
   - Code: TPC-B anchors use V = min(`SC_DB_VCPUS`, scale) and ignore `SC_PROFILE_VUS`, which only seeds the default max clients and summary `profile_vus`. (`benchmark.py:712-714`, `benchmark.py:1004-1008`)
5. `SC_PROFILE_HARD_MAX_CLIENTS` default (docs incomplete).
   - Docs: 3072 for TPC-B (`docs/usage.md:220`).
   - Code: 3072, then capped by `conn_cap` and per scale by the scale factor. (`benchmark.py:1009-1012`, `benchmark.py:1019`, `benchmark.py:1033-1037`)
6. `SC_DB_SSLMODE` scope (docs vs. code).
   - Docs: "SSL (Secure Sockets Layer) mode for database and dataset-dump connections" (`docs/usage.md:212`); "controls SSL (Secure Sockets Layer) mode" (`README.md:91`).
   - Code: only CDN restore, `pg_dump`, and the drop/create connection use it; `pgbench`, `psql` setup and `SHOW` calls, and `collect_postgres_repro` use libpq defaults. (`db_dataset_cache.py:135-136`, `db_dataset_cache.py:153-155`, `db_dataset_cache.py:266-268`, `benchmark.py:134-137`, `benchmark.py:269-276`)
7. Literal substitution vs. prepared mode (CHANGELOG and code comment vs. code).
   - CHANGELOG and code comment: `pgbench` substitutes variables as literals before planning, so the generic-plan issue does not affect production (`CHANGELOG.md:45-51`, `profile_v2_breakdown.sql:20-23`).
   - Code: production uses `-M prepared` (bind parameters), acknowledged in `ro_cpu_txn.sql:54-56`; the CHANGELOG recalibration command omits `-M prepared` (`CHANGELOG.md:83-84`). (`benchmark.py:611-612`)
   - Context: the generic-plan risk is deferred and need not be covered in the docs (`.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?"); the CHANGELOG claim still contradicts the code.
8. `nice -n -20` ownership (context vs. code).
   - Context: PostgreSQL priority via `nice -n -20` is a container setting applied by `sc-inspector` orchestration (`.agents/context/benchmark-pgbench-postgres.md`, "Does production apply OS-level tuning?"; `.agents/context/shared.md`, "Where is a production run defined?").
   - Code: this image's harness launches `postgres` under `nice -n -20` itself, and the sibling server image's entrypoint does too; orchestration only supplies the privilege. (`benchmark.py:210-214`, `images/benchmark-postgres-server/Dockerfile:8`)
   - Docs follow the code (`docs/usage.md:188-191`).
9. CTE and `UNION ALL` count (context and docs vs. code).
   - Context and docs: "one `SELECT` with eight CTEs and one `UNION ALL`" (`.agents/context/benchmark-pgbench-postgres.md`, "Why is the `pgbench_ro` transaction a single statement?"; `docs/limitations.md:92`).
   - Code: 25 CTEs, of which 8 are tagged blocks, joined by 7 `UNION ALL` operators. (`ro_cpu_txn.sql:67-342`)
10. `PGBENCH_RO_CPU_SCHEMA_GIB` as an environment variable (context vs. code).
    - Context: describes it as an environment variable used historically to tune dataset size (`.agents/context/benchmark-pgbench-postgres.md`, "What is the size of the benchmarking database?").
    - Code: a module constant, never read from the environment, emitted as `schema_gib`. (`benchmark.py:48-49`, `benchmark.py:1100`)
11. Terminate privileges for remote runs (docs vs. code and PostgreSQL rules).
    - Docs: the role can terminate sessions "as its owner, a member of `pg_signal_backend`, or a superuser" (`docs/usage.md:67-69`).
    - Code: calls `pg_terminate_backend` on other sessions of the benchmark DB (`db_dataset_cache.py:279-282`). PostgreSQL grants this to members of the session's role, `pg_signal_backend`, or superusers; database ownership alone does not qualify.
12. `wal_buffers` rule (docs incomplete).
    - Docs: 3% of `shared_buffers`, capped at 16 MB (`docs/usage.md:33`).
    - Code: also rounds 14 to 16 MB up to 16 MB and enforces a 32 kB minimum. (`pgtune_leopard.py:184-193`)
