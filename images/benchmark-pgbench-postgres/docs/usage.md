# Usage

## Run Benchmark

This benchmark can be run via Docker:

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

## Key Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `SC_WORKLOAD` | Workload: `pgbench_ro` or `pgbench_tpcb`. | `pgbench_ro` |
| `SC_DB_HOST` | Remote database host; empty starts local PostgreSQL. | Empty. |
| `SC_DB_PORT` | Database port. | `5432` |
| `SC_DB_USER` | Database role. | `postgres` |
| `SC_DB_PASSWORD` | Database password. | `postgres` |
| `SC_DB_NAME` | Admin database used for setup and settings queries. | `postgres` |
| `SC_PGBENCH_DB` | Database used by the benchmark. | `pgbench` |
| `SC_DB_SSLMODE` | SSL mode for database and dataset-dump connections. | `prefer` |
| `SC_CPU_SCALE` | `pgbench_ro` transaction work multiplier. | `1` |
| `SC_SCALEFACTORS` | Comma-separated `pgbench_tpcb` scale factors. | Unset. |
| `SC_SCALEFACTOR` | `pgbench_tpcb` scale factor when `SC_SCALEFACTORS` is unset or empty. | `65` |
| `SC_PROFILE_VUS` | Comma-separated concurrency anchors. | Derived from DB vCPUs. |
| `SC_PROFILE_SEARCH` | Allow adaptive concurrency search; forced off for `pgbench_ro`. | True for `pgbench_tpcb`; false for `pgbench_ro`. |
| `SC_PROFILE_IMPROVE_PCT` | Throughput improvement threshold for TPC-B search. | `5.0` |
| `SC_PROFILE_MAX_CLIENTS` | Maximum client count for the profile. | Highest anchor. |
| `SC_PROFILE_HARD_MAX_CLIENTS` | Hard concurrency ceiling. | Highest anchor for RO; `3072` for TPC-B. |
| `SC_RUN_SECONDS` | Measurement duration per concurrency rung. | `300` |
| `SC_WARMUP_SECONDS` | Initial warmup duration. | `120` |
| `SC_SETTLE_SECONDS` | Settle duration between concurrency rungs. | `60` |
| `SC_WARMUP_ONCE` | Use one full warmup, then settle between rungs. | `true` |
| `SC_DB_VCPUS` | Database vCPU count used for concurrency anchors and local tuning. | `os.cpu_count() or 2` |
| `SC_CLIENT_VCPUS` | Client vCPU count recorded in output. | `os.cpu_count() or 2` |
| `SC_DB_MEM_GIB` | Database memory in GiB recorded in output. | Detected locally; unset for remote databases. |
| `SC_DURABILITY` | Local server durability; `async` disables `synchronous_commit`. | `durable` |
| `SC_TOPOLOGY` | Topology recorded in output. | `single_vm` locally; otherwise `multi_vm`. |
| `SC_CDN_BASE_URL` | CDN (Content Delivery Network) base URL for cached dataset dumps. | `https://cdn.sparecores.net/sc-inspector` |
| `SC_CDN_DATASET_POST_B64` | Optional base64-encoded presigned upload configuration for dataset dumps. | Unset. |
| `SC_CDN_UPLOAD` | Permit dataset upload when a valid upload configuration is supplied. | `1` |

The process prints one indented, key-sorted JSON object to
stdout. It includes a per-concurrency `profile` array and a headline `score`
in TPM (transactions per minute). Profile behavior depends on the workload:

- `pgbench_ro` reports TPM only, not TPS, and uses the fixed concurrency
  profile `{1, V/2, V, 2·V}`. It caps `pgbench` worker jobs at 32 per run,
  independently of the client count.
- `pgbench_tpcb` uses geometric concurrency anchors with optional adaptive
  search.
