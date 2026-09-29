# Facts: benchmark-pgbench-postgres

Source commit: 2eba12e16dd378afb115c7b55a29e3232ee94397 (uncommitted changes)
Inputs: images/benchmark-pgbench-postgres, images/benchmark-postgres-server, .agents/context/shared.md, .agents/context/benchmark-pgbench-postgres.md

## What is measured

- Two workloads are selected by `SC_WORKLOAD` and default to `pgbench_ro`; the code accepts `pgbench_ro` and `pgbench_tpcb`. (`benchmark.py:1015-1018`, `benchmark.py:1042-1048`)
- The headline score is TPM (transactions per minute): `pgbench` output is parsed for `tps`, converted to `tpm` with `round(tps * 60)`, and stored as `score` as well. (`benchmark.py:80-90`, `benchmark.py:392-406`, `benchmark.py:1019-1040`)
- For a given size, the score is the max `tpm` in the concurrency profile; the overall document score is the max of these size scores. (`benchmark.py:900-955`, `benchmark.py:1051-1087`)
- `pgbench_ro` uses a fixed concurrency set `{1, V/2, V, 2·V}` from `SC_DB_VCPUS`; `SC_PROFILE_SEARCH` is forced off for this workload. (`benchmark.py:627-641`, `benchmark.py:1000-1018`)
- `pgbench_tpcb` uses geometric anchors plus an optional upward ladder; the search default is on only for TPC-B. (`benchmark.py:21-31`, `benchmark.py:64-77`, `benchmark.py:643-678`, `benchmark.py:1001-1008`)
- The `pgbench_ro` path applies one schema, `jit=off`, `work_mem='64MB'`, `max_parallel_workers_per_gather=0`, and uses `pgbench -M prepared -n`. The TPC-B path does not apply those ALTER DATABASE GUCs. (`benchmark.py:527-528`, `benchmark.py:537-556`, `benchmark.py:600-635`, `benchmark.py:817-836`)
- The image is based on `postgres:18` and does not pin a minor version. (`Dockerfile:1-6`)
- The module comment explicitly says a separate client VM is wasted, because pgbench uses under 1% CPU and ~10 MB RSS on an 8-vCPU run; the code therefore supports colocating client and server in the same host. (`benchmark.py:134-147`)
- The code supports both local standalone mode (empty `SC_DB_HOST`) and remote mode (non-empty `SC_DB_HOST`). Maintainer context says IaaS runs colocate client and server on the same node, while DBaaS uses a separate client VM. (`benchmark.py:961-985`; context: `.agents/context/benchmark-pgbench-postgres.md`, “Where is the benchmarking client run?”)

## Workload

- `pgbench_ro` executes `pgbench -D scale=<SC_CPU_SCALE> -f ro_cpu_txn.sql` with `jobs = min(clients, 32)`. (`benchmark.py:31-46`, `benchmark.py:559-638`)
- `ro_cpu_setup.sql` creates a schema for 20,000 products, 50,000 customers, 250,000 orders, and 750,000 line items; the comment states the footprint is about 260–320 MB plus indexes. (`ro_cpu_setup.sql:16`, `ro_cpu_setup.sql:68-70`, `ro_cpu_setup.sql:92`, `ro_cpu_setup.sql:94-106`, `ro_cpu_setup.sql:108`, `ro_cpu_setup.sql:140`, `ro_cpu_setup.sql:142-151`)
- `PGBENCH_RO_CPU_SCHEMA_GIB` is set to `0.17` and is emitted as `schema_gib` in the JSON output for `pgbench_ro`. (`benchmark.py:40-48`, `benchmark.py:1088-1094`)
- The custom transaction is one `SELECT md5(string_agg(...))` built from eight CTE blocks: `q_idx`, `q_hashjoin`, `q_regex`, `q_fts`, `q_array`, `q_stats`, `q_toast`, and `q_seqscan`. (`ro_cpu_txn.sql:67`, `ro_cpu_txn.sql:120-126`, `ro_cpu_txn.sql:334-341`)
- The SQL header is a calibration note rather than enforcement; it says the blocks total roughly 78 ms at scale 1. (`ro_cpu_txn.sql:10-18`)
- `-D scale` changes limit/slice widths inside the SQL script; it does not resize the tables. (`ro_cpu_txn.sql:3-5`)
- `pgbench_ro` initialization runs `psql -f ro_cpu_setup.sql` and then applies the three ALTER DATABASE settings; the scale list is a placeholder `[0]` because CPU scale is separate. (`benchmark.py:522-556`, `benchmark.py:720-730`)
- `pgbench_tpcb` initialization runs `pgbench -i -s <scale>` and then the built-in `-b tpcb-like` script; jobs equals clients, and the plan refuses `max(clients) > scale`. (`benchmark.py:525-548`, `benchmark.py:600-620`, `benchmark.py:719-729`, `benchmark.py:1029-1039`)
- Each concurrency rung does a warmup/settle run before a measured run; the first rung uses `SC_WARMUP_SECONDS` when `SC_WARMUP_ONCE` is true, otherwise it falls back to `SC_SETTLE_SECONDS`. Measurement runs use `SC_RUN_SECONDS`, `-P 5`, and a latency log. (`benchmark.py:867-919`, `benchmark.py:998-1001`)
- Latency log percentiles come from the sample log, stored as milliseconds; the parser records `p50`, `p95`, `p99`, `avg`, and `samples`. (`benchmark.py:82-117`, `benchmark.py:398-447`)
- `pgbench_tpcb` stops extending the ladder when a non-anchor rung fails to beat the previous peak by `SC_PROFILE_IMPROVE_PCT`; it may append the next geometric rung if the last rung still improves. (`benchmark.py:892-916`, `benchmark.py:31-31`)
- Dataset preparation skips CDN restore if an object already exists; otherwise it builds the dataset and may upload it. The RO dump name is `pgbench-ro-cpu-v1.sql.zst` and the TPC-B dump name is `pgbench-init-sf<N>.sql.zst`. (`db_dataset_cache.py:293-335`, `db_dataset_cache.py:370-385`, `benchmark.py:817-846`)
- After a CDN restore for the RO dump, the ALTER DATABASE GUCs are reapplied. (`benchmark.py:847-851`)
- If `SC_DB_HOST` is non-empty, the benchmark uses that host and does not start PostgreSQL. If empty, the same container starts `docker-entrypoint.sh postgres` on `127.0.0.1:5432` and sets `SC_TOPOLOGY=single_vm` when unset. (`benchmark.py:1009-1019`, `benchmark.py:154-186`, `benchmark.py:191-209`)
- Standalone GUC generation calls `pgtune_leopard.generate_for_host` with PostgreSQL 18 defaults, sets `synchronous_commit=off` when `SC_DURABILITY=async`, otherwise `on`, and raises `max_connections` to at least `3072 + 50`. (`benchmark.py:120-151`, `benchmark.py:154-186`, `pgtune_leopard.py:32-40`)
- Remote mode does not retune the server; it only queries `SHOW synchronous_commit` and `SHOW max_connections` on the admin database. (`benchmark.py:736-771`, `benchmark.py:1010-1018`)
- Client concurrency is capped at `max_connections - 50`; any anchors above the cap are dropped, and the plan falls back to `[1]` if nothing remains. (`benchmark.py:771-773`, `benchmark.py:1019-1029`)
- The summary field `topology` is driven by `SC_TOPOLOGY` or defaults to `multi_vm`; standalone mode sets it to `single_vm` only via `setdefault` before the summary is built. (`benchmark.py:1016-1019`, `benchmark.py:1071-1092`)

## Parameters and defaults

| Name | Default | Meaning | Citation |
| ------ | --------- | --------- | ---------- |
| `SC_WORKLOAD` | `pgbench_ro` | workload selector | `benchmark.py:1015-1018` |
| `SC_DB_HOST` | empty string | local Postgres when empty; remote host otherwise | `benchmark.py:1009-1019` |
| `SC_DB_PORT` | `5432` | target port | `benchmark.py:1009-1019` |
| `SC_DB_USER` | `postgres` | role name | `benchmark.py:1009-1019` |
| `SC_DB_PASSWORD` | `postgres` | password via `PGPASSWORD` | `benchmark.py:66-70`, `benchmark.py:1009-1019` |
| `SC_DB_NAME` | `postgres` | admin DB used for `CREATE DATABASE` and settings queries | `benchmark.py:1009-1019`, `benchmark.py:775-788` |
| `SC_PGBENCH_DB` | `pgbench` | benchmark database name | `benchmark.py:1009-1019` |
| `SC_DB_SSLMODE` | `prefer` | used when opening remote connections, CDN restore, and dump restore | `db_dataset_cache.py:135-136`, `db_dataset_cache.py:153-155`, `db_dataset_cache.py:266-268` |
| `SC_CPU_SCALE` | `1` | multiplier for `pgbench_ro` `-D scale` | `benchmark.py:1000-1014`, `benchmark.py:564-620` |
| `SC_SCALEFACTOR` | `65` | default TPC-B scale when `SC_SCALEFACTORS` is unset | `benchmark.py:666-681`, `benchmark.py:720-730` |
| `SC_SCALEFACTORS` | unset | CSV of TPC-B scales | `benchmark.py:677-681`, `benchmark.py:720-729` |
| `SC_PROFILE_VUS` | derived from vCPUs | explicit concurrency override | `benchmark.py:1000-1014` |
| `SC_PROFILE_SEARCH` | true for TPC-B, false for RO | whether to extend the geometric ladder | `benchmark.py:1000-1014` |
| `SC_PROFILE_IMPROVE_PCT` | `5.0` | stop threshold for TPC-B search | `benchmark.py:31-31`, `benchmark.py:1000-1014` |
| `SC_PROFILE_MAX_CLIENTS` | max anchor | maximum concurrency before connection cap | `benchmark.py:1000-1014`, `benchmark.py:1019-1029` |
| `SC_PROFILE_HARD_MAX_CLIENTS` | max anchor for RO, 3072 for TPC-B | hard cap for ladder expansion | `benchmark.py:1000-1014`, `benchmark.py:1029-1039` |
| `SC_RUN_SECONDS` | `300` | measurement duration | `benchmark.py:1000-1014` |
| `SC_WARMUP_SECONDS` | `120` | warmup duration for first rung | `benchmark.py:1000-1014` |
| `SC_SETTLE_SECONDS` | `60` | settle duration between rungs | `benchmark.py:1000-1014` |
| `SC_WARMUP_ONCE` | true | one real warmup, then settle | `benchmark.py:62-71`, `benchmark.py:1000-1014` |
| `SC_DB_VCPUS` | `os.cpu_count() or 2` | used for anchor math and standalone tuning | `benchmark.py:120-151`, `benchmark.py:1000-1014` |
| `SC_CLIENT_VCPUS` | `os.cpu_count() or 2` | recorded in output only | `benchmark.py:1082-1086` |
| `SC_DB_MEM_GIB` | unset; standalone fills from `/proc/meminfo` | recorded in output | `benchmark.py:118-126`, `benchmark.py:1082-1086` |
| `SC_DURABILITY` | `durable` | standalone `async` disables synchronous_commit | `benchmark.py:120-151`, `benchmark.py:1009-1019` |
| `SC_TOPOLOGY` | `multi_vm` if unset | reported topology label | `benchmark.py:1016-1019`, `benchmark.py:1071-1087` |
| `SC_CDN_BASE_URL` | `https://cdn.sparecores.net/sc-inspector` | dataset CDN prefix | `db_dataset_cache.py:42-44` |
| `SC_CDN_DATASET_POST_B64` | unset | presigned upload token | `db_dataset_cache.py:51-64`, `db_dataset_cache.py:291-294` |
| `SC_CDN_UPLOAD` | `1` if token is also set | enables upload | `db_dataset_cache.py:291-294` |

- Boolean env parsing accepts `1`, `true`, `yes`, and `on`; empty values use the default. (`benchmark.py:62-71`)
- The measurement progress interval is fixed to 5 seconds; it is not configurable by env var. (`benchmark.py:603-618`)
- Initialization timeout is 14400 seconds, and the per-run timeout is `seconds + 600`. (`benchmark.py:338-350`, `benchmark.py:557-568`, `benchmark.py:603-615`)

## Outputs and schema

- The benchmark prints one JSON object on stdout with `indent=2` and `sort_keys=True`. (`benchmark.py:1121-1123`)
- The process exits 0 after printing; standalone mode stops the local server after the summary is printed. (`benchmark.py:1121-1128`)
- Top-level fields always include: `benchmark`, `benchmark_image`, `workload`, `workload_kind`, `topology`, `durability`, `synchronous_commit`, `max_connections`, `max_connections_client_cap`, `run_seconds`, `warmup_seconds`, `settle_seconds`, `warmup_once`, `improve_pct`, `profile_search`, `profile_vus`, `profile_max_clients`, `profile_hard_max_clients`, `db_vcpus`, `client_vcpus`, `db_mem_gib`, `sizes`, `peak_concurrency`, `score`, `score_unit`, `latency_ms`, `latency_avg_ms`, and `latency_stddev_ms`. (`benchmark.py:1071-1105`)
- `pgbench_ro` adds `cpu_scale`, `peak_cpu_scale`, and `schema_gib`; `pgbench_tpcb` adds `scalefactors`, `peak_scalefactor`, and `scalefactor`. (`benchmark.py:1088-1101`)
- Standalone mode adds `pg_image`, `storage_gib`, `pgtune_share_url`, and a `postgres` object when collection succeeds. (`benchmark.py:38-48`, `benchmark.py:1102-1129`)
- The `postgres` object includes `version`, `server_version`, `server_version_num`, `in_recovery`, `settings`, `nondefault_settings`, `extensions`, `role_settings`, and `requested_gucs` when pgtune produced them. (`benchmark.py:317-387`)
- Each `sizes[]` entry includes `dataset`, `profile`, `profile_vus`, `concurrency_plan`, `profile_max_clients`, `peak_concurrency`, `score`, latency fields, and `stop_reason`; RO entries also include `cpu_scale`, and TPC-B entries include `scalefactor` and `clients_capped_at_scale`. (`benchmark.py:875-955`, `benchmark.py:1051-1087`)
- `dataset` metadata includes `dataset`, `cdn_url`, and `source` (`cdn` or `built`); if built locally it may also include `uploaded` and `upload_error`. (`db_dataset_cache.py:295-351`)
- Each `profile[]` entry includes `concurrency`, `jobs`, `anchor`, `warmup_seconds`, the parsed `pgbench` fields, `run_seconds`, optional `latency_ms`, optional `stop_reason`, and `tpm_vs_final_peak_pct`. (`benchmark.py:884-955`)
- Parsed fields include `tpm`, `score`, `latency_avg_ms`, `latency_stddev_ms`, `tx_processed`, `tx_failed`, and `initial_connection_ms`; failed `pgbench` invocations are retained only if the output still contains `tpm`. (`benchmark.py:88-116`, `benchmark.py:392-406`, `benchmark.py:603-615`)
- `latency_ms.avg`, `p50`, `p95`, and `p99` all come from the latency log and are stored in milliseconds. (`benchmark.py:391-447`)

## How to run

- The image tag is `ghcr.io/sparecores/benchmark-pgbench-postgres:main`. (`benchmark.py:40-48`)
- The entrypoint is `resource-tracker -- python3 /benchmark/benchmark.py`, and `TRACKER_QUIET=true` is set in the image. (`Dockerfile:17-22`, `Dockerfile:24-25`)
- The image copies `benchmark.py`, `db_dataset_cache.py`, `pgtune_leopard.py`, `ro_cpu_setup.sql`, and `ro_cpu_txn.sql`; `profile_v2_breakdown.sql` is not in the COPY list. (`Dockerfile:17-22`)
- A minimal remote run sets `SC_DB_HOST` and optionally `SC_DB_PASSWORD`; the workload defaults to `pgbench_ro`. (`benchmark.py:1009-1019`)
- A minimal standalone run leaves `SC_DB_HOST` unset; the same container starts PostgreSQL locally on `127.0.0.1:5432` and then runs the benchmark against it. (`benchmark.py:1009-1019`, `benchmark.py:154-186`, `benchmark.py:191-209`)
- Installed packages include `ca-certificates`, `curl`, `python3`, `python3-psycopg`, and `zstd`. (`Dockerfile:8-17`)
- The base `postgres:18` image provides `postgres`, `psql`, `pgbench`, and `pg_dump`, which are invoked by name rather than installed here. (`Dockerfile:1-6`, `benchmark.py:181-210`, `benchmark.py:546-560`, `benchmark.py:603-615`)

## Dependencies and platforms

- `DEPENDS_ON` contains only `resource-tracker`. (`images/benchmark-pgbench-postgres/DEPENDS_ON:1`)
- `BUILD_ARGS` defines `RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main-${ARCH}`; the Dockerfile default is `ghcr.io/sparecores/resource-tracker:main`. (`images/benchmark-pgbench-postgres/BUILD_ARGS:1`, `Dockerfile:1-4`, `Dockerfile:17-18`)
- `PLATFORMS` lists `amd64` and `arm64`. (`images/benchmark-pgbench-postgres/PLATFORMS:1-2`)
- `CONTEXT` is `images/benchmark-pgbench-postgres`. (`images/benchmark-pgbench-postgres/CONTEXT:1`)
- There is no `ZRAM` or `SCCACHE` file in this image folder.
- The sibling image `images/benchmark-postgres-server` is a separate server image that also uses `postgres:18` and a resource-tracker entrypoint; the pgbench image does not reference it in code or metadata. (`images/benchmark-postgres-server/Dockerfile:1-11`, `images/benchmark-postgres-server/DEPENDS_ON:1`)
- The dataset code uses the CDN prefix `sc-inspector`. (`db_dataset_cache.py:21`)
- The comments explicitly call out sync with sc-inspector helper trees such as `benchmark_tiers.py`, `postgres_multi.py`, and `pg_repro.py`, which are outside this repo. (`benchmark.py:28-31`, `benchmark.py:40-48`, `benchmark.py:120-151`, `benchmark.py:154-186`, `pgtune_leopard.py:1-2`)

## Conflicts between code and existing docs

- The README and `docs/limitations.md` describe the benchmark as read-only and no-WAL without limiting the claim to `pgbench_ro`. The code also supports `pgbench_tpcb`, which invokes the built-in `tpcb-like` workload and is described in `docs/workloads.md` as mostly-write. (`benchmark.py:612-617`, `benchmark.py:817-836`, `README.md:17`, `docs/limitations.md:42-48`, `docs/workloads.md:3-7`)
- `docs/usage.md` marks the `SC_DB_HOST` default as `—`, but the code defaults it to an empty string and starts local PostgreSQL when empty. (`benchmark.py:961-985`, `docs/usage.md:20`)
- `docs/purpose.md` describes IaaS as using a “colocated pair” of cloud and client VMs, while maintainer context says IaaS runs the client and database on the same node and DBaaS uses a separate client VM. (`docs/purpose.md:7`, context: `.agents/context/benchmark-pgbench-postgres.md`, “Where is the benchmarking client run?”)
- `docs/limitations.md` states deployment flags and same-zone private networking as current run behavior. The image code does not configure host flags; shared context places those choices in orchestration, outside this repository. Whether the listed flags are used in current fleet runs remains unanswered. (`docs/limitations.md:69-78`, context: `.agents/context/shared.md`, “Where is a production run defined?”)
- The RO schema SQL comment reports ~260–320 MB of data plus indexes, while benchmark output emits a fixed `schema_gib` value of `0.17`; the intended relationship between physical footprint and reported value is not established in the code. (`ro_cpu_setup.sql:16`, `benchmark.py:40-48`, `benchmark.py:1094-1097`, `docs/workloads.md:13`)
- `docs/usage.md` and `docs/design-history.md` use Windows-style image-relative paths for links to `benchmark.py` and `profile_v2_breakdown.sql`; those files are at the image root, above the `docs/` directory. (`docs/usage.md:25`, `docs/design-history.md:103`)
- The README’s Usage section repeats its preceding general limitations paragraph and sends the actual Docker command and output details to `docs/usage.md`, rather than presenting those essentials in the manual. (`README.md:15-27`, `docs/usage.md:3-31`; `AGENTS.md`, “README: the manual”)

## Open questions for maintainers

- Do current fleet orchestration settings use the privileged mode, host networking, `seccomp=unconfined`, `nofile` / `memlock` ulimits, and process priority described in `docs/limitations.md`, and where are those settings defined?
- What does `PGBENCH_RO_CPU_SCHEMA_GIB=0.17` represent relative to the schema SQL comment estimating 260–320 MB plus indexes?
- Does `resource-tracker` preserve the benchmark's JSON document intact on stdout in the actual container runtime?
