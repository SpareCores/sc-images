# Facts: benchmark-pgbench-postgres

Source commit: afae74e3d7bd46afc0fde89952d90345fe80d31d
Inputs: images/benchmark-pgbench-postgres ':(exclude)images/benchmark-pgbench-postgres/README.md' ':(exclude)images/benchmark-pgbench-postgres/docs' ':(exclude)images/benchmark-pgbench-postgres/CHANGELOG.md' images/benchmark-postgres-server ':(exclude)images/benchmark-postgres-server/README.md' ':(exclude)images/benchmark-postgres-server/docs' ':(exclude)images/benchmark-postgres-server/CHANGELOG.md' .agents/context/shared.md .agents/context/benchmark-pgbench-postgres.md

- All Inputs pathspecs are clean at HEAD (`git status --short` on them is empty). Uncommitted docs (`README.md`, `docs/usage.md`, `docs/limitations.md`, `docs/workloads.md`, `docs/references.md`, `CHANGELOG.md`) are excluded from Inputs and do not count; they were read from disk for the Conflicts check.
- Only Inputs changes since the previous extraction (`d764abf`): `.agents/context/benchmark-pgbench-postgres.md` and `.agents/context/shared.md` in `17f9ce9` and `afae74e` (new answers: no Navigator network benchmark; `nice -n -20` is started by the image; `PGBENCH_RO_CPU_SCHEMA_GIB` was a setting whose old constant is still emitted as `schema_gib`; CTE wording; `storage_gib` is a fixed label; keep a one-line `schema_gib` warning; `MemTotal` rounding accepted). No code changed; all code citations were re-verified against the current files.
- `.github/` has no `pgbench` or `postgres` specific build behavior (no matches), so it was not read further.

## What is measured

- Two workloads, selected by `SC_WORKLOAD`: `pgbench_ro` (default) and `pgbench_tpcb`; any other value raises an error. (`benchmark.py:964-966`)
- Headline metric: TPM. `pgbench`'s `tps = ... (without initial connection time)` is multiplied by 60, rounded to an integer, and stored as both `tpm` and `score`; no TPS field is emitted. (`benchmark.py:78-80`, `benchmark.py:393-400`)
- Per size: score is the highest-TPM profile point; summary score is the best size. Ties keep the first entry (Python `max`). (`benchmark.py:936-955`, `benchmark.py:1068-1096`)
- What varies between runs (by design): server hardware; the concurrency points derived from `SC_DB_VCPUS`. (`benchmark.py:667-674`, `benchmark.py:998-1007`)
- What is held constant for `pgbench_ro`: one fixed schema (no scale factor), transaction script, `-D scale` work multiplier (default 1), `pgbench -M prepared -n`, and three database-level settings (`jit=off`, `work_mem='64MB'`, `max_parallel_workers_per_gather=0`). (`benchmark.py:490-523`, `benchmark.py:597-618`, `benchmark.py:721-724`, `ro_cpu_setup.sql:18-178`)
- Engine version: local server comes from `FROM postgres:18` (major pinned, minor floats with the base tag). Remote server version is not checked; version details are only recorded in standalone mode via `collect_postgres_repro`. (`Dockerfile:4`, `benchmark.py:968-988`, `benchmark.py:1109-1126`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Is the PostgreSQL version held constant across runs?")
- Disk and network are excluded by design, not measured; the workload is meant to be CPU and memory bound on a cached dataset. (`ro_cpu_setup.sql:1-4`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is disk performance excluded?" and "Why is network performance excluded, and how is RTT handled?")
- No Spare Cores image measures network throughput or latency, and Navigator does not publish network benchmarks; docs must not point to one. (source: context, `.agents/context/shared.md`, "Does Navigator publish network benchmarks?")
- Read-only: the measured `pgbench_ro` transaction is a single `SELECT`; writes happen only during dataset build or restore. "No WAL" applies to the measured steady state. (`ro_cpu_txn.sql:67-342`, `ro_cpu_setup.sql:18-178`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does a measured `pgbench_ro` run write WAL?")
- Purpose and gap filled: source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why does this benchmark exist?" and "What is the primary advantage versus other database benchmarks?"; `.agents/context/shared.md`, "What are these images for?"
- Design goal on instance range: from small instances (example given: 1 vCPU, 2 GB RAM) up to hundreds of vCPUs; the 2 GiB production minimum is also the smallest instance size the design targets. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the primary advantage versus other database benchmarks?" and "Which servers is `pgbench_ro` run on in production?")
- Production server selection: `pgbench_ro` runs in production only on servers with at least 2 GiB RAM as reported by the vendor (nominal size, not `MemTotal`). The code has no RAM check or guard. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?"; `benchmark.py:152-158`, `benchmark.py:963-1131`)
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
- The image starts PostgreSQL under `nice -n -20` itself; orchestration's privileged mode is what lets the higher priority take effect. (`benchmark.py:209-214`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does production apply OS-level tuning?")
- Readiness: polls `psql -h 127.0.0.1 -p 5432 -U postgres` up to 120 s. Port 5432 and user `postgres` are hard-coded here, while later client calls use `SC_DB_PORT` and `SC_DB_USER`, so standalone needs those left at defaults. (`benchmark.py:227-241`, `benchmark.py:969-970`)
- Memory input: `MemTotal` from `/proc/meminfo` (host value; cgroup `--memory` limits not read), floored to whole GiB, minimum 1. (`benchmark.py:152-158`, `pgtune_leopard.py:375-388`)
- CPU input: `SC_DB_VCPUS`, default `os.cpu_count() or 2`. `os.cpu_count()` reports the system's logical CPUs, not a Docker `--cpus` quota or `--cpuset-cpus` set. (`benchmark.py:981`, `benchmark.py:183`)
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
- The local server is stopped only after the JSON is printed; on an exception it is not explicitly stopped (the container exits). (`benchmark.py:1128-1131`, `benchmark.py:244-251`)

### Dataset preparation

- Every size: `CREATE DATABASE "<SC_PGBENCH_DB>"` with "already exists" tolerated. (`benchmark.py:776-799`, `benchmark.py:824`)
- CDN check: HTTP HEAD on `<SC_CDN_BASE_URL>/<filename>` with a 30 s timeout; 403 or 404 means miss; network errors and other HTTP errors propagate and fail the run (both modes). (`db_dataset_cache.py:67-77`, `db_dataset_cache.py:309-312`)
- Dump filenames: RO `pgbench-ro-cpu-v1.sql.zst`; TPC-B `pgbench-init-sf<N>.sql.zst`. Key has no PostgreSQL version. (`db_dataset_cache.py:37-39`, `db_dataset_cache.py:370-385`)
- Hit: terminate other sessions on the benchmark DB, `DROP DATABASE IF EXISTS`, `CREATE DATABASE`, then `bash -c "curl -fsSL <url> | zstd -d | psql -v ON_ERROR_STOP=1"`. The pipeline has no `pipefail`, so only `psql`'s exit status is checked. (`db_dataset_cache.py:116-138`, `db_dataset_cache.py:255-284`)
- After restore, no explicit `ANALYZE` or `VACUUM` runs; the code only re-applies the RO database settings. `pg_dump` flags: `--no-owner --no-privileges --format=plain`. (`benchmark.py:840-854`, `db_dataset_cache.py:145-172`)
- Deferred by the maintainer and not for the docs: planner statistics and hint bits after a CDN restore; truncated CDN downloads. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Miss: build natively (RO: `psql -f ro_cpu_setup.sql` plus database settings; TPC-B: `pgbench -i -s <scale>`), then stream `pg_dump | zstd -T0 | curl` presigned POST if upload is enabled; upload errors are recorded, not fatal. (`benchmark.py:526-574`, `db_dataset_cache.py:181-252`, `db_dataset_cache.py:326-351`)
- Upload key prefix is the constant `sc-inspector/`, independent of `SC_CDN_BASE_URL`. (`db_dataset_cache.py:21`, `db_dataset_cache.py:32-34`, `db_dataset_cache.py:220-222`)
- Timeouts are per command: RO setup `psql -f`, `pgbench -i`, and CDN restore 14,400 s each; `CREATE DATABASE` 120 s; each `ALTER DATABASE` 60 s; each `pgbench` call duration plus 600 s; readiness 120 s. The `pg_dump | zstd | curl` upload stream has no timeout. (`benchmark.py:467`, `benchmark.py:549`, `benchmark.py:521`, `benchmark.py:794`, `benchmark.py:636`, `db_dataset_cache.py:80`, `db_dataset_cache.py:199-238`, `benchmark.py:227`)

### `pgbench_ro` schema (`ro_cpu_setup.sql`)

- Tables and rows: `ro_cpu_product` 20,000; `ro_cpu_customer` 50,000; `ro_cpu_order` 250,000 (5 per customer); `ro_cpu_order_item` 750,000 (3 per order). (`ro_cpu_setup.sql:68-92`, `ro_cpu_setup.sql:94-106`, `ro_cpu_setup.sql:108-140`, `ro_cpu_setup.sql:142-151`)
- Order items reference only products 1 to 5,000 (cold catalog tail). (`ro_cpu_setup.sql:68-70`, `ro_cpu_setup.sql:147`)
- Status formula mixes in the order's sequence so each customer's 5 orders cover all 5 statuses (fixes the v1 bug). (`ro_cpu_setup.sql:108-122`)
- `spec_blob` is about 4.5 KB of concatenated md5 strings (140 x 32 chars) to force out-of-line TOAST. (`ro_cpu_setup.sql:32-35`, `ro_cpu_setup.sql:91`)
- `search_doc` is a stored generated `tsvector`. (`ro_cpu_setup.sql:54-56`)
- Indexes: PKs, 6 B-tree, 2 expression, GIN on `search_doc` and `tags`, BRIN on `ordered_at`. (`ro_cpu_setup.sql:153-169`)
- Ends with `ANALYZE` on all four tables and `GRANT SELECT ... TO PUBLIC`, inside one transaction, then prints an informational size report. (`ro_cpu_setup.sql:18`, `ro_cpu_setup.sql:171-189`)
- Size: comment estimates 260 to 320 MB data plus indexes and says it fits in `shared_buffers`; maintainer reports `pg_database_size` of 303 MiB after a fresh import on PostgreSQL 18. (`ro_cpu_setup.sql:1-4`, `ro_cpu_setup.sql:16`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the size of the benchmarking database?")
- Memory fit by floored `MemTotal` (pgtune `shared_buffers` = M/4): M = 1 GiB gives 256 MB (below 303 MiB); M = 2 GiB gives 512 MB (above). (`benchmark.py:152-158`, `pgtune_leopard.py:124-130`, `pgtune_leopard.py:384`)
- A server sold with 2 GiB usually reports slightly less `MemTotal`, so it is tuned as M = 1 GiB and `shared_buffers` (256 MB) ends up below the 303 MiB database; the maintainer accepts this, no code fix planned. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does the `MemTotal` rounding need a code fix?"; `benchmark.py:152-158`, `pgtune_leopard.py:384`)
- Maintainer: below the 2 GiB vendor-reported production minimum the dataset does not fit in memory, with possible disk overhead. At 2 GiB it may not fit in `shared_buffers` alone and relies on the OS page cache as well, consistent with the `MemTotal` answer above. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?")
- Data distribution is modular (`g % k`), a known simplification. (`ro_cpu_setup.sql:71-151`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Is the uniform data distribution intentional?")
- Table replacement on native build: see Previous Review Findings.

### `pgbench_ro` transaction (`ro_cpu_txn.sql`)

- One `SELECT md5(string_agg(x, '|' ORDER BY x))` over a `WITH` of 25 CTEs; 8 tagged block CTEs (`q_idx`, `q_hashjoin`, `q_regex`, `q_fts`, `q_array`, `q_stats`, `q_toast`, `q_seqscan`) combined with 7 `UNION ALL` operators. (`ro_cpu_txn.sql:67-342`) Context now describes the same shape: eight query blocks as CTEs combined with `UNION ALL`. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is the `pgbench_ro` transaction a single statement?")
- Random inputs per transaction: customer 1 to 50,000; order 1 to 250,000; product 1 to 5,000; region, tag, and FTS term indices. (`ro_cpu_txn.sql:37-44`)
- `-D scale=N` multiplies: `regex_width` 3600, `hj_window_sec` 1500, `fts_lim` 40, `array_slice_width` 48000, `stats_width` 8000, `toast_n` 700. Fixed: `array_lim` 200, `q_idx`, `q_seqscan`. (`ro_cpu_txn.sql:46-65`)
- Scale ceiling: `hj_start_sec = random(0, 250000 - 1500 * scale)` has an empty range for `SC_CPU_SCALE` >= 167. Maintainer deferred an upper-bound guard and excluded it from the docs. (`ro_cpu_txn.sql:52-53`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Calibrated block times at scale 1 (code comment, local Docker Postgres 18): `q_idx` 0.1, `q_hashjoin` 24, `q_regex` 15, `q_fts` 12, `q_array` 8, `q_stats` 13, `q_toast` 4, `q_seqscan` 2 ms; total about 78 ms; no block above about 30%. (`ro_cpu_txn.sql:10-18`, `ro_cpu_txn.sql:46-50`)
- Calibration location: first on a local machine, then on a cloud test instance (which changed between providers); maintainer says this detail does not matter for the docs. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Where were the `pgbench_ro` block weights calibrated?")
- Comment: variable sums are precomputed in `\set` because under `-M prepared` SQL-side `:a + :b` becomes `$N + $M` with unknown types. (`ro_cpu_txn.sql:54-57`)
- Generic-plan risk under `-M prepared`: maintainer considers it probably fine, to be checked later, and excludes it from the docs. (`benchmark.py:611-612`, `profile_v2_breakdown.sql:14-23`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which open edge cases are deliberately left out of the docs?")
- Code-internal inconsistency (not in published docs): the `profile_v2_breakdown.sql` comment says real `pgbench` uses the simple query protocol and substitutes variables as literal text, while the harness runs `-M prepared`. (`profile_v2_breakdown.sql:20-23`, `benchmark.py:611-612`)
- Single-statement rationale: one transaction equals one round trip; trade-off is no per-block planner GUCs. (`ro_cpu_txn.sql:23-29`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why is the `pgbench_ro` transaction a single statement?")
- Database-level settings via `ALTER DATABASE ... SET`: `jit=off`, `work_mem='64MB'`, `max_parallel_workers_per_gather=0`; applied during build and again after every prepare (CDN restores lose them). (`benchmark.py:490-523`, `benchmark.py:572-574`, `benchmark.py:850-854`; rationale source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why are JIT and parallel query disabled for `pgbench_ro`?")
- `profile_v2_breakdown.sql`: dev-only calibration helper, not copied into the image; it sets `jit`, `work_mem`, `max_parallel_workers_per_gather`, and `plan_cache_mode=force_custom_plan` per session. (`profile_v2_breakdown.sql:1-3`, `profile_v2_breakdown.sql:25-28`, `Dockerfile:23`)

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
- `SC_PROFILE_HARD_MAX_CLIENTS` has no effect on RO points; it is used only by the TPC-B search branch and echoed in the summary. (`benchmark.py:912-933`, `benchmark.py:1085`)
- Fixed-profile rationale: source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Why does `pgbench_ro` use a fixed concurrency profile?"
- Connection cap: `conn_cap = max_connections - 50` (from `SHOW max_connections` on the admin DB) caps max clients, hard max, and host anchors; if every anchor is above it, the list becomes [1]. (`benchmark.py:752-773`, `benchmark.py:1015-1020`)
- TPC-B per scale: `anchor_v = min(SC_DB_VCPUS, scale)`; anchors = {1, rung(anchor_v/4), rung(anchor_v/2), rung(anchor_v)}; `max_clients = min(scale, host_max_clients)`. `host_anchors` (and therefore `SC_PROFILE_VUS`) is not used to build TPC-B anchors; it only sets the default `SC_PROFILE_MAX_CLIENTS` and the summary `profile_vus`. (`benchmark.py:672-674`, `benchmark.py:698-718`, `benchmark.py:1004-1008`, `benchmark.py:1083`)
- `choose_concurrency_plan`: keeps anchors <= `max_clients`; with search on, appends every ladder rung above the highest kept anchor up to `max_clients`. (`benchmark.py:677-688`)
- Run loop (TPC-B with search): at a non-anchor point, stop if TPM < previous peak x (1 + `SC_PROFILE_IMPROVE_PCT`/100); if the last planned point is a non-anchor that clears the threshold, append the next rung up to `hard_max = min(SC_PROFILE_HARD_MAX_CLIENTS, scale, conn_cap)`. Anchor points never trigger either rule. (`benchmark.py:912-934`, `benchmark.py:1009-1012`, `benchmark.py:1033-1037`)
- What search actually adds (derived from the plan functions):
  - Default scale 65 with default `SC_PROFILE_MAX_CLIENTS`: never adds a point for any V; the plan equals the anchors, all points are anchors, so the run-loop rules never fire. Examples: V=16 gives 1, 4, 8, 16; V>=57 gives 1, 16, 32, 64. (`benchmark.py:672-718`, `benchmark.py:1008`)
  - Default scale 65 with raised `SC_PROFILE_MAX_CLIENTS`: adds ladder rungs between the highest anchor and min(raised value, 65). When V>=57 the highest anchor is already 64 and no rung lies in (64, 65], so raising it adds nothing. (`benchmark.py:662-664`, `benchmark.py:677-688`, `benchmark.py:712`)
  - Non-default scale with default `SC_PROFILE_MAX_CLIENTS`: when rung(min(V, scale)) > scale, the top anchor is dropped and the one ladder rung between the V/2 anchor and the scale is added as a search point. Examples: scale 57, V>=57 gives 1, 16, 32, 48 (anchor 64 dropped); scale 11, V>=11 gives 1, 3, 6, 8. The run-loop extension cannot add more, because `max_clients` and `hard_max` are then both capped by the scale. (`benchmark.py:677-688`, `benchmark.py:712-717`, `benchmark.py:924-933`, `benchmark.py:1033-1037`)
  - Raised `SC_PROFILE_MAX_CLIENTS` (below scale): the plan gets ladder rungs up to it, and the run loop can then extend past it up to `min(scale, SC_PROFILE_HARD_MAX_CLIENTS, conn_cap)` while each point improves by the threshold. Example: V=16, scale 65, `SC_PROFILE_MAX_CLIENTS=32` plans 1, 4, 8, 16, 24, 32, then may add 48 and 64. (`benchmark.py:677-688`, `benchmark.py:924-933`)
  - Raising `SC_PROFILE_MAX_CLIENTS` above the scale has no effect beyond the scale; TPC-B clients never exceed the scale factor (guard raises if the plan does). (`benchmark.py:712`, `benchmark.py:1038-1042`)

### Warmup and timing

- Per point: warmup or settle `pgbench` run at that point's client count, then a measured run of `SC_RUN_SECONDS`. First point gets `SC_WARMUP_SECONDS`; later points get `SC_SETTLE_SECONDS` when `SC_WARMUP_ONCE` is true, else the full warmup. A zero duration skips it. (`benchmark.py:863-898`, `benchmark.py:990-993`)
- Warmup state carries across TPC-B scales, so later scales only get settle runs. (`benchmark.py:1023`, `benchmark.py:1043-1063`)
- Default RO duration excluding preparation: V>=4 is 120 + 3 x 60 + 4 x 300 = 1500 s (25 min); V=2 or 3 is 1140 s (19 min); V=1 is 780 s (13 min). (`benchmark.py:667-669`, `benchmark.py:990-993`)
- Wall time also includes non-measured steps (server start, readiness, dataset restore or build, settings queries, `pgbench` connection setup, final repro collection). (`benchmark.py:206-241`, `benchmark.py:824-898`, `benchmark.py:1015-1016`, `benchmark.py:1117-1124`) Maintainer: a breakdown of this extra time is not relevant for the docs. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does the `m9g.24xlarge` example's extra time need a breakdown?")
- Production timing example: the AWS `m9g.24xlarge` run is an actual production run recorded in `sc-inspector-data`. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Where does the `m9g.24xlarge` timing example come from?") At V=96 the code gives points 1, 48, 96, 192 and 1500 s of warmup plus measurement. (`benchmark.py:667-669`, `benchmark.py:990-993`)
- Production measurement window choice (5 min) is from history, not code; see CHANGELOG claims (Measurement Duration section).

### Latency

- Percentiles come from the sampled per-transaction logs: field 3 (microseconds) divided by 1000; linear-interpolated p50, p95, p99, mean, and sample count. (`benchmark.py:419-456`, `benchmark.py:647-658`)

### Code not in use (leftovers)

- Maintainer: the code may still contain leftovers from earlier experimentation; document what is in use, not leftovers. Some experiment artifacts live in `sc-db-benchmark-tmp`. (source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Should the docs cover leftovers from earlier experiments?")
- Code items with no effect on a run of this image:
  - `PGBENCH_RO_CPU_SCHEMA_GIB = 0.17`, emitted as `schema_gib`; not used for sizing or any decision. Maintainer: a former setting, now fixed; the docs keep a one-line warning that it is not the dataset size, because every `pgbench_ro` run prints it. (`benchmark.py:48-49`, `benchmark.py:1100`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the size of the benchmarking database?" and "Should the docs explain the `schema_gib` output field?")
  - `cdn_env_for_benchmark()`: defined, never called inside the image. (`db_dataset_cache.py:354-367`)
  - `DatasetSpec.tool` and `DatasetSpec.workload`: set, never read; only `filename` and `s3_key` are used. (`db_dataset_cache.py:24-34`, `db_dataset_cache.py:370-385`)
  - `pgbench_filename` docstring says RO and TPC-B share one schema, but RO uses its own `pgbench-ro-cpu-v1` dump. (`db_dataset_cache.py:37-39`, `db_dataset_cache.py:379-385`)
  - `max_connections_for_vcpus` ignores its `vcpus` argument. (`benchmark.py:161-167`)
  - `SC_PROFILE_HARD_MAX_CLIENTS` for RO: only echoed in the summary. (`benchmark.py:1009-1012`, `benchmark.py:1085`)
  - `profile_v2_breakdown.sql`: not shipped in the image. (`Dockerfile:23`, `profile_v2_breakdown.sql:1-3`)

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
| `SC_PROFILE_HARD_MAX_CLIENTS` | RO: highest anchor; TPC-B: 3072 | TPC-B run-loop extension ceiling, also capped by scale and `conn_cap`; no effect for RO beyond the summary | `benchmark.py:1009-1012`, `benchmark.py:1019`, `benchmark.py:1033-1037`, `benchmark.py:912-933` |
| `SC_RUN_SECONDS` | `300` | Measured seconds per point | `benchmark.py:990` |
| `SC_WARMUP_SECONDS` | `120` | First (or every, if not warmup-once) warmup | `benchmark.py:991` |
| `SC_SETTLE_SECONDS` | `60` | Settle before later points | `benchmark.py:992` |
| `SC_WARMUP_ONCE` | `true` | Full warmup once, then settle | `benchmark.py:993` |
| `SC_DB_VCPUS` | `os.cpu_count() or 2` (system logical CPUs, not cgroup quota) | Concurrency points; standalone pgtune CPU input | `benchmark.py:981`, `benchmark.py:998` |
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

- `storage_gib`: a fixed label for the benchmarking environment (the 128 GiB `sc-inspector` root volume), not a measurement and not a benchmark metric; hard-coded `STORAGE_GIB = 128`, emitted in every standalone run including a plain `docker run`. Code and context now agree. (`benchmark.py:45-47`, `benchmark.py:1113`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is `storage_gib`?")
- `schema_gib` is not the dataset size (database is about 303 MiB) and drives nothing in code; docs should explain it in one line. (`benchmark.py:48-49`, `benchmark.py:1100`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "What is the size of the benchmarking database?" and "Should the docs explain the `schema_gib` output field?")
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
- `nice -n -20` in standalone needs a container allowed to raise priority (for example privileged); otherwise `nice` cannot apply it and the server runs at normal priority. (`benchmark.py:210-214`, `Dockerfile:6`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Does production apply OS-level tuning?")
- Remote minimal: set `SC_DB_HOST`, credentials, and `SC_DB_VCPUS` (default is the client's CPU count). (`benchmark.py:968-973`, `benchmark.py:998`)
- Remote privileges implied by code: `CREATE DATABASE`; `DROP DATABASE` of the benchmark DB and `pg_terminate_backend` on its sessions (CDN restore path); `ALTER DATABASE ... SET` (RO); `CREATE TABLE` or `pgbench -i` (build path). (`benchmark.py:500-502`, `benchmark.py:776-799`, `db_dataset_cache.py:279-284`, `ro_cpu_setup.sql:20-66`)
- Network: outbound HTTPS to `SC_CDN_BASE_URL` required in both modes. (`db_dataset_cache.py:67-77`, `db_dataset_cache.py:312`)
- Memory: no minimum enforced in code; production uses servers with at least 2 GiB RAM as reported by the vendor. (`benchmark.py:152-158`; source: context, `.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?")
- Image contents: `ca-certificates`, `curl`, `python3`, `python3-psycopg`, `zstd`, plus `benchmark.py`, `db_dataset_cache.py`, `pgtune_leopard.py`, `ro_cpu_setup.sql`, `ro_cpu_txn.sql` in `/benchmark/`. (`Dockerfile:14-23`)

## Dependencies and platforms

- `DEPENDS_ON`: `resource-tracker`. (`DEPENDS_ON:1`)
- Build arg `RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main-${ARCH}`; Dockerfile default `...:main`. (`BUILD_ARGS:1`, `Dockerfile:1-3`)
- Base image: `postgres:18`, running as root at build. (`Dockerfile:4-6`)
- Platforms: `amd64`, `arm64`. Build context: `images/benchmark-pgbench-postgres`. No `ZRAM` or `SCCACHE` files. (`PLATFORMS:1-2`, `CONTEXT:1`)
- Sibling `benchmark-postgres-server`: `postgres:18` plus `resource-tracker`, entrypoint `resource-tracker -- nice -n -20 docker-entrypoint.sh`, CMD `postgres`; no `PLATFORMS` or `CONTEXT` file; this image does not depend on it. (`images/benchmark-postgres-server/Dockerfile:1-9`, `images/benchmark-postgres-server/DEPENDS_ON:1`, `images/benchmark-postgres-server/BUILD_ARGS:1`)
- Family: this README is the PostgreSQL manual; the server image gets no separate methodology write-up. (source: context, `.agents/context/shared.md`, "Which folders share one manual?")
- Vendored or mirrored code that must stay in sync with `sc-inspector`: `pgtune_leopard.py`, `max_connections_for_vcpus`, `pg_guc_settings`, `collect_postgres_repro`, the ladder, and `PGBENCH_RO_CPU_SCHEMA_GIB`. (`pgtune_leopard.py:2`, `benchmark.py:48-51`, `benchmark.py:164`, `benchmark.py:179`, `benchmark.py:265`)
- `__pycache__/` in the image folder is Git-ignored (`.gitignore:2`) and not part of the build inputs (`Dockerfile:23` copies named files only).

## Previous Review Findings

- Kept: the setup script opens a transaction and runs `DROP TABLE IF EXISTS ... CASCADE` for `ro_cpu_order_item`, `ro_cpu_order`, `ro_cpu_customer`, and `ro_cpu_product` before creating them, so a native RO build replaces any existing `ro_cpu_*` tables in `SC_PGBENCH_DB` without dropping the database. Rechecked at `afae74e`; unchanged at the cited lines. (`ro_cpu_setup.sql:18-23`, `ro_cpu_setup.sql:178`, `benchmark.py:554-574`)
  - Context for the finding: a native build runs only on a CDN miss; on a CDN hit the whole database is dropped and recreated instead. (`db_dataset_cache.py:312-328`, `db_dataset_cache.py:255-284`)
- The previous facts file had no bullets marked `(added in review)`; no other findings to recheck.

## Open questions for maintainers

None found.

### Conflicts

None found.

- Dropped as resolved since the previous facts file:
  - Dataset fit at the 2 GiB production minimum: context now says that at 2 GiB the dataset may not fit in `shared_buffers` alone and relies on the OS page cache as well (`.agents/context/benchmark-pgbench-postgres.md`, "Which servers is `pgbench_ro` run on in production?"), consistent with the `MemTotal` answer and `benchmark.py:152-158`, `pgtune_leopard.py:124-130`, `pgtune_leopard.py:384`; docs agree ("`shared_buffers` plus the OS page cache", `docs/limitations.md:117-122`).
  - `storage_gib` described as a measurement: context now calls it a fixed label, not a measurement, matching `benchmark.py:45-47`, `benchmark.py:1113` (`.agents/context/benchmark-pgbench-postgres.md`, "What is `storage_gib`?").
  - `nice -n -20` ownership: context now says the image starts PostgreSQL with `nice -n -20` and privileged mode lets it take effect, matching `benchmark.py:209-214` and `images/benchmark-postgres-server/Dockerfile:8`; docs agree (`docs/usage.md:194-197`).
  - CTE and `UNION ALL` count: context now says eight query blocks are CTEs combined with `UNION ALL`, matching `ro_cpu_txn.sql:67-342`; docs agree (`docs/limitations.md:95-96`).
  - `PGBENCH_RO_CPU_SCHEMA_GIB` as an environment variable: context now calls it a setting whose old constant is still emitted as `schema_gib`, matching `benchmark.py:48-49`, `benchmark.py:1100`.
  - Calibration location: docs now say only that the weights were manually calibrated, with no location (`docs/limitations.md:106-108`), consistent with the context answer that the location does not matter.
- Previous open questions now answered by context and removed:
  - `schema_gib` explanation: keep a one-line warning (`.agents/context/benchmark-pgbench-postgres.md`, "Should the docs explain the `schema_gib` output field?"); README does so (`README.md:125-126`).
  - Network benchmark link: Navigator publishes none (`.agents/context/shared.md`, "Does Navigator publish network benchmarks?"); docs now say only that the score does not measure network throughput or latency (`README.md:46-49`, `docs/limitations.md:29-35`).
