# Facts: benchmark-pgbench-postgres

Source commit: 15d1379ba43af121fb6ad167722a1adc7415a484
Inputs: images/benchmark-pgbench-postgres, images/benchmark-postgres-server, .agents/context/shared.md, .agents/context/benchmark-pgbench-postgres.md

## What is measured

- `SC_WORKLOAD` accepts `pgbench_ro` and `pgbench_tpcb`, defaulting to `pgbench_ro`. (`benchmark.py:964-966`)
- The headline score is TPM: `pgbench` TPS is rounded after multiplying by 60 and stored as both `tpm` and `score`. (`benchmark.py:393-399`)
- Each size's score is the highest TPM in its concurrency profile; the summary uses the highest-scoring size. (`benchmark.py:933-967`, `benchmark.py:1066-1103`)
- `pgbench_ro` uses fixed concurrency anchors `{1, V/2, V, 2V}` derived from `SC_DB_VCPUS`; adaptive search is disabled for this workload. (`benchmark.py:667-670`, `benchmark.py:997-1006`)
- `pgbench_tpcb` uses geometric anchors with optional ladder expansion; search defaults on for TPC-B and stops when an expanded rung fails the improvement threshold. (`benchmark.py:52-75`, `benchmark.py:675-692`, `benchmark.py:896-923`, `benchmark.py:994-996`)
- Both workloads run with `pgbench -M prepared -n`; the RO path uses the custom SQL and applies `jit=off`, `work_mem='64MB'`, and `max_parallel_workers_per_gather=0`, while TPC-B uses `-b tpcb-like`. (`benchmark.py:487-505`, `benchmark.py:523-621`)
- The image's local database uses `postgres:18`; the Dockerfile does not pin a minor version. Remote server versions are not checked. (`Dockerfile:4-5`, `benchmark.py:964-985`)
- Local and remote modes are supported. Maintainer context says IaaS colocates the client and database on one node, while DBaaS uses a separate client VM. (`benchmark.py:968-985`; context: `.agents/context/benchmark-pgbench-postgres.md`, “Where is the benchmarking client run?”)

## Workload

- `pgbench_ro` runs `pgbench -D scale=<SC_CPU_SCALE> -f ro_cpu_txn.sql`; jobs are capped at 32 per run. (`benchmark.py:42`, `benchmark.py:577-621`)
- The custom transaction is one `SELECT md5(string_agg(...))` using eight CTE blocks: `q_idx`, `q_hashjoin`, `q_regex`, `q_fts`, `q_array`, `q_stats`, `q_toast`, and `q_seqscan`. (`ro_cpu_txn.sql:67`, `ro_cpu_txn.sql:120-126`, `ro_cpu_txn.sql:334-341`)
- `-D scale` changes limit/slice widths in the transaction; it does not resize the schema. (`ro_cpu_txn.sql:3-5`, `ro_cpu_txn.sql:36-61`)
- `ro_cpu_setup.sql` creates 20,000 products, 50,000 customers, 250,000 orders, and 750,000 line items; its comment estimates 260–320 MB of data plus indexes. (`ro_cpu_setup.sql:16`, `ro_cpu_setup.sql:68-70`, `ro_cpu_setup.sql:92`, `ro_cpu_setup.sql:94-106`, `ro_cpu_setup.sql:108`, `ro_cpu_setup.sql:140-151`)
- `PGBENCH_RO_CPU_SCHEMA_GIB=0.17` is emitted as `schema_gib` for the RO run as a fixed constant, not measured from the created schema. (`benchmark.py:49`, `benchmark.py:1097-1100`)
- RO initialization runs the setup SQL and applies the three database settings; TPC-B initialization uses `pgbench -i -s <scale>`. (`benchmark.py:487-505`, `benchmark.py:523-571`)
- Each concurrency rung gets a warmup or settle run followed by a measured run. The first rung uses `SC_WARMUP_SECONDS` when `SC_WARMUP_ONCE` is true; otherwise it uses `SC_SETTLE_SECONDS`. Measured runs use `SC_RUN_SECONDS`, progress reporting, and sampled latency logs. (`benchmark.py:857-919`, `benchmark.py:990-993`, `benchmark.py:577-632`)
- Latency samples are recorded in milliseconds; the parser reports `p50`, `p95`, `p99`, average, and sample count. (`benchmark.py:419-455`)
- Dataset preparation restores a matching CDN dump when available; otherwise it builds the database and may upload a dump. (`db_dataset_cache.py:297-342`)
- With empty `SC_DB_HOST`, the container starts local PostgreSQL at `127.0.0.1:5432`; local settings use pgtune defaults, durability controls `synchronous_commit`, the server process starts at nice `-20`, and `SC_TOPOLOGY` defaults to `single_vm`. A non-empty host is used as-is, without server retuning. (`benchmark.py:161-213`, `benchmark.py:968-985`, `benchmark.py:1015-1018`; context: `.agents/context/benchmark-pgbench-postgres.md`, “Does production apply OS-level tuning?”)

## Parameters and defaults

| Name | Default | Meaning | Citation |
| ------ | --------- | --------- | ---------- |
| `SC_WORKLOAD` | `pgbench_ro` | workload selector | `benchmark.py:964-966` |
| `SC_DB_HOST` | empty string | local server if empty; remote host otherwise | `benchmark.py:968-985` |
| `SC_DB_PORT` | `5432` | database port | `benchmark.py:969-973` |
| `SC_DB_USER` | `postgres` | role name | `benchmark.py:970-972` |
| `SC_DB_PASSWORD` | `postgres` | password via `PGPASSWORD` | `benchmark.py:68-71`, `benchmark.py:970-973` |
| `SC_DB_NAME` | `postgres` | admin database for setup/settings queries | `benchmark.py:972-973` |
| `SC_PGBENCH_DB` | `pgbench` | benchmark database name | `benchmark.py:973` |
| `SC_DB_SSLMODE` | `prefer` | SSL mode for database/dump connections | `db_dataset_cache.py:135-136`, `db_dataset_cache.py:153-154`, `db_dataset_cache.py:266-268` |
| `SC_CPU_SCALE` | `1` | RO transaction work multiplier | `benchmark.py:998-1000`, `benchmark.py:607-614` |
| `SC_SCALEFACTOR` | `65` | TPC-B scale if `SC_SCALEFACTORS` is unset | `benchmark.py:725-728` |
| `SC_SCALEFACTORS` | unset | CSV list of TPC-B scale factors | `benchmark.py:723-728` |
| `SC_PROFILE_VUS` | derived from DB vCPUs | concurrency anchor override | `benchmark.py:1001-1006` |
| `SC_PROFILE_SEARCH` | true for TPC-B; forced false for RO | adaptive ladder extension | `benchmark.py:994-996` |
| `SC_PROFILE_IMPROVE_PCT` | `5.0` | TPC-B ladder improvement threshold | `benchmark.py:31`, `benchmark.py:994-996` |
| `SC_PROFILE_MAX_CLIENTS` | highest anchor | profile concurrency cap | `benchmark.py:1008` |
| `SC_PROFILE_HARD_MAX_CLIENTS` | highest anchor for RO; 3072 for TPC-B | search ceiling | `benchmark.py:1009-1011` |
| `SC_RUN_SECONDS` | `300` | measured duration per rung | `benchmark.py:990-992` |
| `SC_WARMUP_SECONDS` | `120` | initial warmup duration | `benchmark.py:990-992` |
| `SC_SETTLE_SECONDS` | `60` | settle duration between rungs | `benchmark.py:990-992` |
| `SC_WARMUP_ONCE` | true | use one full warmup, then settle | `benchmark.py:62-71`, `benchmark.py:993` |
| `SC_DB_VCPUS` | `os.cpu_count() or 2` | DB anchors and local tuning input | `benchmark.py:980-981`, `benchmark.py:998-1006` |
| `SC_CLIENT_VCPUS` | `os.cpu_count() or 2` | recorded client CPU count | `benchmark.py:1087-1088` |
| `SC_DB_MEM_GIB` | unset; local mode sets detected memory | reported DB memory | `benchmark.py:220-226`, `benchmark.py:984-985`, `benchmark.py:1087-1088` |
| `SC_DURABILITY` | `durable` | local mode sets `synchronous_commit`; `async` turns it off | `benchmark.py:980`, `benchmark.py:186` |
| `SC_TOPOLOGY` | `multi_vm` if unset | reported topology; local mode defaults it to `single_vm` | `benchmark.py:985`, `benchmark.py:1072-1074` |
| `SC_CDN_BASE_URL` | `https://cdn.sparecores.net/sc-inspector` | CDN prefix | `db_dataset_cache.py:42-44` |
| `SC_CDN_DATASET_POST_B64` | unset | presigned upload token | `db_dataset_cache.py:51-64` |
| `SC_CDN_UPLOAD` | `1` | permits upload when a valid token is supplied; false-like values disable it | `db_dataset_cache.py:291-294` |

- Boolean environment parsing accepts `1`, `true`, `yes`, and `on`; empty values use the default. (`benchmark.py:62-71`)
- Progress output is fixed at 5-second intervals. Initialization timeout is 14,400 seconds; each benchmark run has a timeout of duration plus 600 seconds. (`benchmark.py:603-618`, `benchmark.py:338-350`, `benchmark.py:557-568`)

## Outputs and schema

- The process prints one indented, key-sorted JSON object to stdout and returns 0 after a successful run. (`benchmark.py:1125-1134`)
- Common top-level fields are `benchmark`, `workload`, `topology`, `durability`, `synchronous_commit`, `max_connections`, `max_connections_client_cap`, `run_seconds`, `warmup_seconds`, `settle_seconds`, `warmup_once`, `improve_pct`, `profile_search`, `profile_vus`, `profile_max_clients`, `profile_hard_max_clients`, `db_vcpus`, `client_vcpus`, `db_mem_gib`, `sizes`, `peak_concurrency`, `score`, `score_unit`, `latency_ms`, `latency_avg_ms`, and `latency_stddev_ms`. `benchmark_image` and `workload_kind` are added before output. (`benchmark.py:1069-1095`, `benchmark.py:1103-1104`)
- `score_unit` is `tpm` (transactions per minute). RO summaries add `cpu_scale`, `peak_cpu_scale`, and `schema_gib`; TPC-B summaries add `scalefactors`, `peak_scalefactor`, and `scalefactor`. (`benchmark.py:1089-1103`)
- Each `sizes[]` entry has `dataset`, `profile`, `profile_vus`, `concurrency_plan`, `profile_max_clients`, `peak_concurrency`, `score`, `latency_ms`, `latency_avg_ms`, `latency_stddev_ms`, and `stop_reason`. RO entries add `cpu_scale`; TPC-B entries add `scalefactor` and `clients_capped_at_scale`. (`benchmark.py:933-967`, `benchmark.py:1043-1067`)
- Each `profile[]` entry records `concurrency`, `jobs`, `anchor`, `warmup_seconds`, parsed `pgbench` summary fields, and `run_seconds`; it may also contain `latency_ms`, `stop_reason`, and `tpm_vs_final_peak_pct`. Parsed summary fields are `tpm`, `score`, `latency_avg_ms`, `latency_stddev_ms`, `tx_processed`, `tx_failed`, and `initial_connection_ms`. (`benchmark.py:393-417`, `benchmark.py:647-664`, `benchmark.py:875-937`)
- Sampled `latency_ms` fields `p50`, `p95`, `p99`, `avg`, and `samples` come from the 1% latency log sample; latency values are milliseconds and `samples` is a count. (`benchmark.py:87`, `benchmark.py:419-455`, `benchmark.py:647-658`)
- Dataset metadata has `dataset` (dump filename), `cdn_url`, and `source` (`cdn` or `built`); locally built datasets may additionally include `uploaded` and `upload_error`. (`db_dataset_cache.py:301-342`)
- Standalone mode adds `pg_image` and `storage_gib`; it may add `pgtune_share_url` and `postgres` when settings collection succeeds. The `postgres` object contains version, server version fields, recovery state, settings, nondefault settings, extensions, role settings, and requested GUCs. (`benchmark.py:317-387`, `benchmark.py:1105-1124`)

## How to run

- Image tag: `ghcr.io/sparecores/benchmark-pgbench-postgres:main`. (`benchmark.py:40-48`)
- Entrypoint: `resource-tracker -- python3 /benchmark/benchmark.py`; `TRACKER_QUIET=true`. (`Dockerfile:25-26`)
- Remote mode sets `SC_DB_HOST` and may set `SC_DB_PASSWORD`; workload defaults to `pgbench_ro`. (`benchmark.py:964-973`)
- Standalone mode leaves `SC_DB_HOST` empty; the container starts its local server and runs the benchmark against it. (`benchmark.py:968-985`, `benchmark.py:206-225`)
- Installed packages are `ca-certificates`, `curl`, `python3`, `python3-psycopg`, and `zstd`; the `postgres:18` base supplies PostgreSQL binaries. (`Dockerfile:4-23`)
- `profile_v2_breakdown.sql` is a development-only helper and is not copied into the image. (`profile_v2_breakdown.sql:1-5`, `Dockerfile:23`)

## Dependencies and platforms

- `DEPENDS_ON` contains `resource-tracker`. (`DEPENDS_ON:1`)
- `BUILD_ARGS` selects `ghcr.io/sparecores/resource-tracker:main-${ARCH}`; Dockerfile default is `ghcr.io/sparecores/resource-tracker:main`. (`BUILD_ARGS:1`, `Dockerfile:1-5`)
- Platforms are `amd64` and `arm64`; build context is `images/benchmark-pgbench-postgres`. (`PLATFORMS:1-2`, `CONTEXT:1`)
- No `ZRAM` or `SCCACHE` file is present in the image folder. (image folder listing)
- The sibling `benchmark-postgres-server` uses `postgres:18` and `resource-tracker`, but this benchmark does not depend on it. (`images/benchmark-postgres-server/Dockerfile:1-9`, `images/benchmark-postgres-server/DEPENDS_ON:1`, `DEPENDS_ON:1`)
- The dataset helper uses the CDN prefix `sc-inspector`. (`db_dataset_cache.py:21`)

## Conflicts between code and existing docs

- The README says both local and remote targets are PostgreSQL 18, but only the local base image is pinned; the remote path connects to the supplied host without checking its server version. (`README.md:4`, `Dockerfile:4-5`, `benchmark.py:968-985`, `benchmark.py:1015-1016`)
- `docs/limitations.md` reports the dataset as 0.17 GiB, while `ro_cpu_setup.sql` estimates 260–320 MB of data plus indexes. The code emits `schema_gib=0.17` as fixed metadata, not a measurement of physical schema size. (`docs/limitations.md:143-148`, `ro_cpu_setup.sql:16`, `benchmark.py:49`, `benchmark.py:1100-1101`)
- README and Limitations claim client/server placement in the same availability zone over private networking, but local code does not control or verify placement; maintainer context confirms IaaS same-node and DBaaS remote-client modes without confirming zone/VPC placement. (`README.md:57-61`, `docs/limitations.md:217-223`; context: `.agents/context/benchmark-pgbench-postgres.md`, “Where is the benchmarking client run?”)
- README says IaaS runs use pgtune without clarifying local mode; code applies pgtune only when `SC_DB_HOST` is empty and leaves remote servers unchanged. (`README.md:57-61`, `benchmark.py:976-981`)
- The OS-level tuning statement is confirmed by maintainer context; orchestration applies the documented container settings without host `sysctl` or other host OS tweaks. (context: `.agents/context/benchmark-pgbench-postgres.md`, “Does production apply OS-level tuning?”; `benchmark.py:206-221`, sibling `Dockerfile:8-9`)
- `docs/usage.md` says the JSON output is “with stdout (benchmark: pgbench_postgres)”; the code directly prints the JSON object to stdout. (`docs/usage.md:27-34`, `benchmark.py:1128`)
- README gives only a high-level throughput/concurrency description and directs readers to Usage for output details; AGENTS.md requires outputs and interpretation in the one-pass manual. (`README.md:8-9`, `README.md:78-80`, `docs/usage.md:27-34`; `AGENTS.md`, “README: the manual”)

## Open questions for maintainers

- What does the fixed `PGBENCH_RO_CPU_SCHEMA_GIB=0.17` represent relative to the setup SQL estimate of 260–320 MB of data plus indexes?
- Are DBaaS client/server runs always placed in the same availability zone and connected over private VPC addresses, as `docs/limitations.md` says?
- Does the `resource-tracker` runtime preserve the benchmark's JSON object on stdout in production?
