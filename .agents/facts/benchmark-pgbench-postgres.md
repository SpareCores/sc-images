# Facts: benchmark-pgbench-postgres

Source commit: 450857e706bdda2b17aed9c4c1d3b31249d23241
Inputs: images/benchmark-pgbench-postgres, images/benchmark-postgres-server, .agents/context/shared.md, .agents/context/benchmark-pgbench-postgres.md

Extracted before either context file existed. Claims below are from the
image code only.

## What is measured

- Two workloads, selected by `SC_WORKLOAD` (default `pgbench_ro`):
  `pgbench_ro` (custom SQL) and `pgbench_tpcb` (built-in `tpcb-like`).
  (`benchmark.py:3-8`, `benchmark.py:961-963`)
- Headline score is TPM (transactions/minute): `pgbench` "tps" × 60,
  rounded to int. The parsed summary stores `tpm` and `score`, not a
  `tps` field. (`benchmark.py:78-80`, `benchmark.py:390-397`)
- Per-size score is the max `tpm` across the concurrency profile. The
  document score is the max of those size scores. Unit string is `tpm`.
  (`benchmark.py:933-947`, `benchmark.py:1065-1089`)
- `pgbench_ro` concurrency is a fixed set `{1, V/2, V, 2·V}` from
  `SC_DB_VCPUS` (default `os.cpu_count()` or 2). Adaptive search is forced
  off. (`benchmark.py:664-666`, `benchmark.py:992-994`,
  `benchmark.py:704-707`)
- `pgbench_tpcb` uses geometric anchors `{1, ~V/4, ~V/2, ~V}` plus an
  optional upward ladder. Search default is on only for this workload.
  (`benchmark.py:8`, `benchmark.py:52-77`, `benchmark.py:669-685`,
  `benchmark.py:992`)
- What the client holds constant in code for `pgbench_ro`: one schema
  (`ro_cpu_setup.sql`), session GUCs `jit=off`, `work_mem=64MB`,
  `max_parallel_workers_per_gather=0`, protocol `-M prepared`, `-n` (skip
  vacuum). (`benchmark.py:487-500`, `benchmark.py:594-615`)
- `pgbench_tpcb` does not apply those ALTER DATABASE GUCs.
  (`benchmark.py:822-836`, `benchmark.py:487-500`)
- Engine major version in this image: `FROM postgres:18`. No minor pin.
  (`Dockerfile:3`)
- Code comment (not a measurement in this repo): a separate client VM is
  described as waste because pgbench used <1% CPU and ~10 MB RSS on an
  8-vCPU run, so both roles can share one instance.
  (`benchmark.py:140-147`)

## Workload

- `pgbench_ro` runs `pgbench -D scale=<SC_CPU_SCALE> -f ro_cpu_txn.sql`.
  Jobs = `min(clients, 32)`. (`benchmark.py:42`, `benchmark.py:590-615`)
- Schema row counts in `ro_cpu_setup.sql`: 20,000 products, 50,000
  customers, 250,000 orders, 750,000 line items (3 per order; line items
  reference only the first 5,000 products). SQL comment: footprint
  ~260–320 MB data + indexes. (`ro_cpu_setup.sql:16`,
  `ro_cpu_setup.sql:68-70`, `ro_cpu_setup.sql:92`,
  `ro_cpu_setup.sql:94-106`, `ro_cpu_setup.sql:108`,
  `ro_cpu_setup.sql:140`, `ro_cpu_setup.sql:142-151`)
- Python constant `PGBENCH_RO_CPU_SCHEMA_GIB = 0.17`, comment says keep in
  sync with sc-inspector. Emitted as `schema_gib` for `pgbench_ro`.
  (`benchmark.py:48-49`, `benchmark.py:1094-1097`)
- Transaction is one `SELECT md5(string_agg(...))` over eight CTE blocks
  unioned: `q_idx`, `q_hashjoin`, `q_regex`, `q_fts`, `q_array`,
  `q_stats`, `q_toast`, `q_seqscan`. (`ro_cpu_txn.sql:67`,
  `ro_cpu_txn.sql:120-126`, `ro_cpu_txn.sql:334-341`)
- SQL header (calibration note, not enforced at runtime): at `scale=1`,
  block times sum to ~78 ms, no block above ~30%.
  (`ro_cpu_txn.sql:10-18`)
- `-D scale` multiplies LIMIT/slice widths inside the script; it does not
  resize the tables. (`ro_cpu_txn.sql:3-5`)
- `pgbench_ro` init is `psql -f ro_cpu_setup.sql` then the three ALTER
  DATABASE statements. Scale list is the placeholder `[0]`; `cpu_scale`
  is separate. (`benchmark.py:551-571`, `benchmark.py:718-721`)
- `pgbench_tpcb` init is `pgbench -i -s <scale>` and the run uses
  `-b tpcb-like`. Jobs = clients (no 32 cap). Default scale 65, or
  `SC_SCALEFACTORS` CSV. Plan refuses `max(clients) > scale`.
  (`benchmark.py:523-548`, `benchmark.py:616-617`,
  `benchmark.py:722-725`, `benchmark.py:1035-1039`)
- Each concurrency rung: warmup or settle run (no progress, no latency
  log), then a measurement run. First rung uses `SC_WARMUP_SECONDS` when
  `SC_WARMUP_ONCE` is true (default true); later rungs use
  `SC_SETTLE_SECONDS`. Measurement uses `SC_RUN_SECONDS`, `-P 5`, and a
  latency log. (`benchmark.py:860-895`, `benchmark.py:987-990`)
- Latency log sampling rate 0.01. Percentiles from log field 3 divided by
  1000 (stored as ms): `p50`, `p95`, `p99`, `avg`, `samples`.
  (`benchmark.py:87`, `benchmark.py:416-453`, `benchmark.py:622-628`)
- TPC-B search stops extending the ladder when a non-anchor rung fails to
  beat the previous peak by `SC_PROFILE_IMPROVE_PCT` (default 5). It can
  append the next geometric rung up to the hard cap when the last rung
  still improves. (`benchmark.py:909-930`, `benchmark.py:52-77`)
- Dataset build is skipped when a CDN object exists; otherwise the build
  runs and may be uploaded. RO dump name `pgbench-ro-cpu-v1.sql.zst`.
  TPC-B dump name `pgbench-init-sf<N>.sql.zst`.
  (`db_dataset_cache.py:297-332`, `db_dataset_cache.py:370-385`,
  `benchmark.py:822-846`)
- After a CDN restore of the RO dump, ALTER DATABASE GUCs are applied
  again. (`benchmark.py:847-851`)
- Connection target: if `SC_DB_HOST` is non-empty, that host is used and
  this process does not start PostgreSQL. If empty, this container starts
  `docker-entrypoint.sh postgres` on `127.0.0.1:5432` with pgtune GUCs and
  sets `SC_TOPOLOGY=single_vm` when unset. (`benchmark.py:965-985`,
  `benchmark.py:206-222`)
- Standalone GUCs: `pgtune_leopard.generate_for_host` with site defaults
  dbVersion=18, linux, web, SSD, mid_ram; `synchronous_commit=off` when
  `SC_DURABILITY=async`, else `on`; `max_connections` at least
  `3072 + 50`. `listen_addresses=*`. (`benchmark.py:161-193`,
  `benchmark.py:206-216`, `pgtune_leopard.py:33-39`,
  `benchmark.py:52-76`)
- Remote mode does not apply server GUCs. It only `SHOW`s
  `synchronous_commit` and `max_connections` on the admin database.
  (`benchmark.py:728-767`, `benchmark.py:1012-1013`)
- Client count is capped at `max_connections - 50`. Anchors above that
  cap are dropped; if none remain, the plan is `[1]`.
  (`benchmark.py:770-771`, `benchmark.py:1014-1017`)
- Reported `topology` is `SC_TOPOLOGY` or `multi_vm` when that variable
  was not set. Standalone sets it to `single_vm` only via `setdefault`
  before the summary is built. (`benchmark.py:985`, `benchmark.py:1069`)

## Parameters and defaults

| Name | Default | Meaning | Citation |
|------|---------|---------|----------|
| `SC_WORKLOAD` | `pgbench_ro` | `pgbench_ro` or `pgbench_tpcb` | `benchmark.py:961-963` |
| `SC_DB_HOST` | empty → local server | Remote host when set | `benchmark.py:965`, `benchmark.py:976-983` |
| `SC_DB_PORT` | `5432` | TCP port | `benchmark.py:966` |
| `SC_DB_USER` | `postgres` | Role | `benchmark.py:967` |
| `SC_DB_PASSWORD` | `postgres` | Password (`PGPASSWORD`) | `benchmark.py:134-137`, `benchmark.py:968` |
| `SC_DB_NAME` | `postgres` | Admin database for `CREATE DATABASE` and `SHOW` | `benchmark.py:969`, `benchmark.py:773-789` |
| `SC_PGBENCH_DB` | `pgbench` | Database that is created and benchmarked | `benchmark.py:970` |
| `SC_DB_SSLMODE` | `prefer` | `PGSSLMODE` for CDN restore, `pg_dump`, and the psycopg drop/create path. Not set by `pgbench_run` | `db_dataset_cache.py:135-136`, `db_dataset_cache.py:153-155`, `db_dataset_cache.py:266-268`, `benchmark.py:574-633` |
| `SC_CPU_SCALE` | `1` | `pgbench -D scale=N` for `pgbench_ro` | `benchmark.py:996`, `benchmark.py:615` |
| `SC_SCALEFACTOR` | `65` | `pgbench -i -s` when `SC_SCALEFACTORS` is unset | `benchmark.py:722-725` |
| `SC_SCALEFACTORS` | unset | CSV of TPC-B scales; ignored for `pgbench_ro` | `benchmark.py:111-115`, `benchmark.py:718-725` |
| `SC_PROFILE_VUS` | derived from vCPUs | CSV override of concurrency anchors | `benchmark.py:997-1004` |
| `SC_PROFILE_SEARCH` | true iff `pgbench_tpcb`; forced false for RO | Upward ladder | `benchmark.py:992-994` |
| `SC_PROFILE_IMPROVE_PCT` | `5.0` | TPC-B stop threshold; JSON field is null for RO | `benchmark.py:991`, `benchmark.py:1078` |
| `SC_PROFILE_MAX_CLIENTS` | max anchor | Then min'd with connection cap | `benchmark.py:1005`, `benchmark.py:1015` |
| `SC_PROFILE_HARD_MAX_CLIENTS` | max anchor (RO) or `3072` (TPC-B) | Ladder ceiling; TPC-B also min'd with scale | `benchmark.py:1006-1009`, `benchmark.py:1030-1034` |
| `SC_RUN_SECONDS` | `300` | Measurement `-T` | `benchmark.py:987` |
| `SC_WARMUP_SECONDS` | `120` | First warmup `-T` | `benchmark.py:988` |
| `SC_SETTLE_SECONDS` | `60` | Later-rung warmup when `SC_WARMUP_ONCE` | `benchmark.py:863-864`, `benchmark.py:989` |
| `SC_WARMUP_ONCE` | true (`1/true/yes/on`) | One real warmup, then settle | `benchmark.py:104-108`, `benchmark.py:990` |
| `SC_DB_VCPUS` | `os.cpu_count()` or 2 | Anchor math; standalone pgtune CPU count | `benchmark.py:978`, `benchmark.py:995` |
| `SC_CLIENT_VCPUS` | `os.cpu_count()` or 2 | Recorded only | `benchmark.py:1084` |
| `SC_DB_MEM_GIB` | unset; standalone fills from `/proc/meminfo` | Recorded; 0 becomes JSON null | `benchmark.py:152-158`, `benchmark.py:984`, `benchmark.py:1085` |
| `SC_DURABILITY` | `durable` | Standalone: `async` → `synchronous_commit=off`, else `on`. Also copied into JSON | `benchmark.py:186`, `benchmark.py:977`, `benchmark.py:1070` |
| `SC_TOPOLOGY` | `multi_vm` if never set; standalone `setdefault` `single_vm` | JSON label only | `benchmark.py:985`, `benchmark.py:1069` |
| `SC_CDN_BASE_URL` | `https://cdn.sparecores.net/sc-inspector` | Dump URL prefix | `db_dataset_cache.py:42-44` |
| `SC_CDN_DATASET_POST_B64` | unset | Base64 JSON presigned POST; required to upload | `db_dataset_cache.py:51-64`, `db_dataset_cache.py:291-294` |
| `SC_CDN_UPLOAD` | `1` (enabled only if POST is also set) | `0/false/no` disables upload | `db_dataset_cache.py:291-294` |

- Bool parser accepts `1`, `true`, `yes`, `on`. Empty env uses the
  default. (`benchmark.py:90-108`)
- `pgbench` progress interval on measurement runs is fixed at 5 seconds
  (not an env var). (`benchmark.py:620-621`)
- Init `psql`/`pgbench -i` timeout is 14400 s. A run timeout is
  `seconds + 600`. (`benchmark.py:464`, `benchmark.py:546`,
  `benchmark.py:633`)

## Outputs and schema

- One JSON object on stdout of `benchmark.py`, `indent=2`,
  `sort_keys=True`. (`benchmark.py:1125`)
- Process exit 0 after printing. Local server is stopped after the print
  when standalone. (`benchmark.py:1126-1128`)
- Top-level fields always set: `benchmark` = `pgbench_postgres`,
  `benchmark_image` = `ghcr.io/sparecores/benchmark-pgbench-postgres:main`,
  `workload`, `workload_kind`, `topology`, `durability`,
  `synchronous_commit` (live `SHOW`), `max_connections`,
  `max_connections_client_cap`, timing fields, `improve_pct`,
  `profile_search`, `profile_vus`, `profile_max_clients`,
  `profile_hard_max_clients`, `db_vcpus`, `client_vcpus`, `db_mem_gib`,
  `sizes`, `peak_concurrency`, `score`, `score_unit` = `tpm`,
  `latency_ms`, `latency_avg_ms`, `latency_stddev_ms`.
  (`benchmark.py:1066-1104`)
- `pgbench_ro` adds `cpu_scale`, `peak_cpu_scale`, `schema_gib`.
  `pgbench_tpcb` adds `scalefactors`, `peak_scalefactor`, `scalefactor`.
  (`benchmark.py:1094-1101`)
- Standalone only: `pg_image` (same image string), `storage_gib` = 128,
  `pgtune_share_url` when non-empty, `postgres` repro blob when collection
  succeeds. (`benchmark.py:46-47`, `benchmark.py:1106-1123`)
- `postgres` object: `version`, `server_version`, `server_version_num`,
  `in_recovery`, full `settings` map (pretty `SHOW` values),
  `nondefault_settings`, `extensions`, `role_settings`, and
  `requested_gucs` when pgtune ran. Failure prints to stderr and omits
  the key. (`benchmark.py:364-387`)
- Each `sizes[]` entry: `dataset`, `profile`, `profile_vus`,
  `concurrency_plan`, `profile_max_clients`, `peak_concurrency`, `score`,
  latency fields, `stop_reason`, plus `cpu_scale` or `scalefactor`, and
  `clients_capped_at_scale` (true only for TPC-B).
  (`benchmark.py:940-956`, `benchmark.py:1061-1062`)
- `dataset`: `dataset` (filename), `cdn_url`, `source` = `cdn` or `built`,
  and when built: `uploaded` bool, optional `upload_error`.
  (`db_dataset_cache.py:309-351`)
- Each `profile[]` entry: `concurrency`, `jobs`, `anchor`,
  `warmup_seconds`, pgbench parse fields, `run_seconds`, optional
  `latency_ms`, optional `stop_reason`, `tpm_vs_final_peak_pct`.
  (`benchmark.py:900-939`)
- Pgbench parse fields when the regex matches: `tpm`, `score`,
  `latency_avg_ms`, `latency_stddev_ms`, `tx_processed`, `tx_failed`,
  `initial_connection_ms`. A failed pgbench process is kept only if the
  error text still contains a `tpm`. (`benchmark.py:390-413`,
  `benchmark.py:632-639`)
- `latency_ms.avg` / `p50` / `p95` / `p99` are from the sample log, in
  milliseconds. (`benchmark.py:428`, `benchmark.py:447-453`)

## How to run

- Image tag constant: `ghcr.io/sparecores/benchmark-pgbench-postgres:main`.
  (`benchmark.py:44`)
- Entrypoint: `resource-tracker -- python3 /benchmark/benchmark.py`.
  `TRACKER_QUIET=true`. (`Dockerfile:25-26`)
- Files copied into the image: `benchmark.py`, `db_dataset_cache.py`,
  `pgtune_leopard.py`, `ro_cpu_setup.sql`, `ro_cpu_txn.sql`.
  `profile_v2_breakdown.sql` is not in that COPY. (`Dockerfile:23`)
- Minimal remote run (host required only to avoid standalone): container
  env `SC_DB_HOST`, and password if not `postgres`. Workload defaults to
  `pgbench_ro`. (`benchmark.py:961-968`, `Dockerfile:26`)
- Minimal standalone run: start the image with `SC_DB_HOST` unset. The
  same container inits and serves PostgreSQL, then runs pgbench against
  `127.0.0.1`. (`benchmark.py:972-983`, `benchmark.py:206-222`)
- Packages installed: `ca-certificates`, `curl`, `python3`,
  `python3-psycopg`, `zstd`. (`Dockerfile:14-19`)
- Base image supplies `postgres`, `psql`, `pgbench`, `pg_dump` (invoked
  by name; not installed again in this Dockerfile). (`Dockerfile:3`,
  `benchmark.py:211-212`, `benchmark.py:468`, `benchmark.py:534`,
  `db_dataset_cache.py:157-170`)

## Dependencies and platforms

- `DEPENDS_ON` contents: `resource-tracker` only.
  (`images/benchmark-pgbench-postgres/DEPENDS_ON:1`)
- `BUILD_ARGS`: `RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main-${ARCH}`.
  Dockerfile default before that arg is `ghcr.io/sparecores/resource-tracker:main`.
  (`BUILD_ARGS:1`, `Dockerfile:1-2`, `Dockerfile:22`)
- `PLATFORMS`: `amd64`, `arm64`. (`PLATFORMS:1-2`)
- `CONTEXT`: `images/benchmark-pgbench-postgres`. (`CONTEXT:1`)
- No `ZRAM` file and no `SCCACHE` file in this image folder.
- Sibling `images/benchmark-postgres-server`: `FROM postgres:18`, copies
  resource-tracker, entrypoint `resource-tracker -- nice -n -20
  docker-entrypoint.sh`, `CMD ["postgres"]`. Its `DEPENDS_ON` is also
  only `resource-tracker`. This pgbench image does not reference that
  folder. (`images/benchmark-postgres-server/Dockerfile:4-9`,
  `images/benchmark-postgres-server/DEPENDS_ON:1`)
- CDN key prefix string `sc-inspector`. (`db_dataset_cache.py:21`)
- Comments say several helpers must stay in sync with sc-inspector
  (`benchmark_tiers.py`, `postgres_multi.py`, `pg_repro.py`). Those trees
  are not in this repo. (`benchmark.py:10`, `benchmark.py:48-51`,
  `benchmark.py:164`, `benchmark.py:179`, `benchmark.py:262`,
  `pgtune_leopard.py:2`)

## Conflicts between code and existing docs

- README says the client runs against a separate `postgres:18` server via
  `SC_DB_HOST`, matching a remote application rather than colocating
  client and server, and that IaaS uses a separate client VM.
  (`README.md:3-6`, `README.md:22-24`)
  Code: empty `SC_DB_HOST` starts PostgreSQL in this container and labels
  topology `single_vm`. The module comment says a separate client VM was
  waste. Default JSON topology when a host is set and `SC_TOPOLOGY` is
  unset is still `multi_vm`. (`benchmark.py:140-147`,
  `benchmark.py:972-985`, `benchmark.py:1069`)
- README usage presents `SC_DB_HOST` as part of the run, and the env table
  marks host as having no default (`—`). (`README.md:167-180`)
  Code default is empty string, which is the standalone path, not a
  required remote host. (`benchmark.py:965-983`)
- README links `benchmark-postgres-server` as the server this client runs
  against. (`README.md:4-5`)
  `DEPENDS_ON` does not list that image, and `benchmark.py` never names
  it. (`DEPENDS_ON:1`)
- README: IaaS GUCs are applied by sc-inspector (pgtune web/SSD/PG 18,
  durability, raised `max_connections`); DBaaS tuning is left to the
  vendor; no OS tuning except privileged host-net container flags,
  ulimits, and `nice -n -20` on the server. (`README.md:144-156`)
  In this repo, pgtune + durability + `max_connections` run only inside
  standalone mode of this image. Remote connections are not retuned.
  `nice -n -20` appears on `benchmark-postgres-server`'s entrypoint, not
  on this image. Privileged, host networking, `seccomp`, and ulimits are
  not in either Dockerfile. (`benchmark.py:170-216`,
  `benchmark.py:1012-1013`,
  `images/benchmark-postgres-server/Dockerfile:8`)
- README design constraint: ~100+ ms of server work per transaction at one
  connection. (`README.md:74-75`)
  `ro_cpu_txn.sql` header says the eight blocks total ~78 ms at scale=1.
  (`ro_cpu_txn.sql:16-18`)
  README's own validation table says 84.1 ms average at `-c 1 -D scale=1`.
  (`README.md:343`)
  None of those millisecond figures are computed by `benchmark.py`.
- README schema size ~260–320 MB (`README.md:70-71`, `README.md:201-203`)
  matches the SQL comment (`ro_cpu_setup.sql:16`).
  The same README later says ~170 MB (`README.md:223-224`), and the JSON
  field `schema_gib` is the constant 0.17 (`benchmark.py:48-49`).
  0.17 GiB is not 260–320 MB. The constant and the SQL comment disagree;
  the README states both.
- README: disk and network are excluded so scores are CPU/memory.
  (`README.md:44-56`, `README.md:128-135`)
  Code does not measure or assert disk I/O or RTT. `pgbench_tpcb` is a
  built-in read-write `tpcb-like` script (`benchmark.py:616-617`), so the
  "read-only, no WAL" constraint (`README.md:73`) does not apply to that
  workload. `pgbench_ro` SQL is a single read-only `SELECT`
  (`ro_cpu_txn.sql:67`).
- README: `jit` and parallel query stay off. (`README.md:137-138`,
  `README.md:354-357`)
  Code sets those GUCs only for `pgbench_ro` via ALTER DATABASE
  (`benchmark.py:496-500`). Not applied on the `pgbench_tpcb` path.
- README operational timing (120 / 60 / 300) matches the env defaults.
  (`README.md:162-163`, `benchmark.py:987-989`) Not a conflict.
- README statement that `profile_v2_breakdown.sql` is not copied into the
  image matches the Dockerfile COPY list. (`README.md:285-286`,
  `Dockerfile:23`) Not a conflict.

## Open questions for maintainers

- Which topology should the manual describe: standalone (this container
  starts PostgreSQL when `SC_DB_HOST` is unset), a remote server with the
  client elsewhere, colocated client and server as two processes, or all
  three as supported?
- For production IaaS runs that set `SC_DB_HOST`, where are pgtune,
  `synchronous_commit`, and `max_connections` applied? This image does not
  do it on the remote path.
- Are privileged mode, host networking, `seccomp=unconfined`, `nofile` /
  `memlock`, and `nice -n -20` still used, and in which repo?
- Should published schema size follow `PGBENCH_RO_CPU_SCHEMA_GIB` (0.17) or
  the SQL comment (~260–320 MB)?
- Should published single-connection service time follow the SQL header
  (~78 ms), the README validation row (84.1 ms), or the "~100+ ms"
  constraint?
- Which README history numbers are still approved to publish (21 GUC
  experiments, ~20% gain, ~98% loss at +5 ms, ±0.3%, ~1.25× CPU split,
  benchANT, ~5,000 server types, planned blog post)? None of them appear
  in this image's code.
- Is DBaaS only "point `SC_DB_HOST` at a vendor endpoint and do not change
  GUCs", or does orchestration set extra env (SSL, scale, concurrency)?
- `SC_DB_SSLMODE` is not applied to `pgbench` / `psql -f` in
  `benchmark.py`. Is that intentional for managed endpoints that require
  SSL?
- Does the `resource-tracker` entrypoint leave the JSON document intact on
  stdout?
- `STORAGE_GIB = 128` is hardcoded for standalone JSON. Is that the real
  root volume size, or only a label?
