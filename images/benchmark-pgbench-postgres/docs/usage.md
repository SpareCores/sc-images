# Usage

## Run Script

To use this benchmark, run the following shell command:

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

## Key Environment Variables

| Variable | Meaning | Default Values |
| - | - | - |
| `SC_WORKLOAD` | `pgbench_ro` (cached CPU-heavy custom script) or `pgbench_tpcb` (built-in `tpcb-like`) | `pgbench_ro` |
| `SC_DB_HOST` / `SC_DB_PORT` / `SC_DB_USER` / `SC_DB_PASSWORD` | connection | — / `5432` / `postgres` / `postgres` |
| `SC_CPU_SCALE` | `pgbench_ro` work multiplier (`-D scale=N`) | `1` |
| `SC_SCALEFACTOR(S)` | `pgbench_tpcb` `-i -s` size(s) | `65` |
| `SC_RUN_SECONDS` / `SC_WARMUP_SECONDS` / `SC_SETTLE_SECONDS` | measurement/warmup timing | `300` / `120` / `60` |

See [`benchmark.py`](/images/benchmark-pgbench-postgres/benchmark.py) docstring for the full list.

The script outputs one JSON document if prompted with `stdout (benchmark: pgbench_postgres)`, with the following information:

- per-concurrency `profile` array
- headline `score` in transactions/minute (TPM)
  - `pgbench_ro` reports TPM only
  - no TPS
  - forces a fixed concurrency profile `{1, V/2, V, 2·V}` instead of `pgbench_tpcb`'s geometric search
