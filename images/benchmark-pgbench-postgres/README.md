# benchmark-pgbench-postgres

`pgbench`-driven PostgreSQL benchmark client for the Spare Cores fleet.
Runs against a separate `postgres:18` server (see
[`benchmark-postgres-server`](../benchmark-postgres-server)) over the network
(`SC_DB_HOST`), matching how a real application talks to a managed/remote
database rather than colocating client and server.

Published as `ghcr.io/sparecores/benchmark-pgbench-postgres:main`.

## Usage

```bash
docker run --rm \
  -e SC_DB_HOST=<postgres-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_WORKLOAD=pgbench_ro \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main
```

Key env vars (see `benchmark.py` docstring for the full list):

| Var | Meaning | Default |
|---|---|---|
| `SC_WORKLOAD` | `pgbench_ro` (cached CPU-heavy custom script) or `pgbench_tpcb` (built-in `tpcb-like`) | `pgbench_ro` |
| `SC_DB_HOST` / `SC_DB_PORT` / `SC_DB_USER` / `SC_DB_PASSWORD` | connection | — / `5432` / `postgres` / `postgres` |
| `SC_CPU_SCALE` | `pgbench_ro` work multiplier (`-D scale=N`) | `1` |
| `SC_SCALEFACTOR(S)` | `pgbench_tpcb` `-i -s` size(s) | `65` |
| `SC_RUN_SECONDS` / `SC_WARMUP_SECONDS` / `SC_SETTLE_SECONDS` | measurement/warmup timing | `300` / `120` / `60` |

Output is one JSON document on stdout (`benchmark: pgbench_postgres`), with a
per-concurrency `profile` array and a headline `score` in **TPM**
(transactions/minute; `pgbench_ro` reports TPM only, no TPS, and forces a
fixed concurrency profile `{1, V/2, V, 2·V}` instead of `pgbench_tpcb`'s
geometric search).

## Workloads

### `pgbench_tpcb`

pgbench's built-in `tpcb-like` script (`-b tpcb-like`) against a standard
`pgbench -i -s N` schema. Standard TPC-B-style OLTP mix (mostly-write,
network- and lock-sensitive); not covered further here.

### `pgbench_ro` — cached CPU-heavy SQL

A fixed, hand-written schema (`ro_cpu_setup.sql`, ~20k products / 50k
customers / 250k orders / 750k line items, ≈260–320 MB) sized to live
entirely in `shared_buffers`, so the transaction's cost is CPU (parse, plan,
execute, join, aggregate, text/JSON/array processing) rather than disk I/O.
One custom transaction (`ro_cpu_txn.sql`, run via `pgbench -D scale=N -f`)
executes 8 read-only query blocks per iteration and returns a single
`md5()` checksum so the planner can't optimize any of it away.

`-D scale=N` linearly scales the row-count knobs inside the transaction
(wider slices, bigger joins) without touching the underlying dataset, so a
single fixed schema can represent a range of CPU intensities.

## Design history: how we got here

### v1: from trivial `pgbench -S` to a cached multi-query script

Plain `pgbench -S` (one `SELECT` by primary key) turned out to be too cheap
per-transaction to say anything meaningful about CPU under network latency —
under `netem`-simulated RTT its TPS collapsed almost entirely from RTT, not
server work (see `sc-db-benchmark-tmp/RESULTS.md`, "Heavy cached RO (table
CPU) vs classic `-S` under netem"). That motivated a **cached, multi-query,
CPU-heavy** custom script (`pg_cpu_benchmark/03_run_benchmark.sql`) sized for
~100–130 ms of server CPU per transaction at `-c 1`, using a small (~170 MB)
schema that fits in `shared_buffers` so I/O is never the bottleneck — so that
RTT stays a small fraction of total latency instead of dominating it.

The original transaction ran four blocks, `q1`–`q4`:

- **q1** — one customer's recent paid/shipped/done orders → join → window
  functions (index scan + nested loop + `WindowAgg`)
- **q2** — a region/plan cohort → their orders → line items, grouped by
  status (broader join + `GroupAggregate`)
- **q3** — a 12,000-row contiguous `order_id` slice, checked against two
  regexes and hashed with `md5()` (regex + JSON + hashing)
- **q4** — top-N buyers of one product (`GROUP BY` + `ORDER BY` + `LIMIT`)

### What was actually wrong with it

Profiling this transaction locally (fresh `postgres:18` in Docker, `jit=off`,
`EXPLAIN (ANALYZE, BUFFERS)` plus `clock_timestamp()` loops per block)
turned up several real problems, not just "regex is slow":

| Finding | Evidence |
|---|---|
| **q3's regex+md5 alone was ~70–82% of the whole transaction.** | Per-block timing: q1 0.09 ms, q2 9.8 ms, q3 49.1 ms, q4 0.10 ms (total 59.6 ms). Inside q3: regex alone was 42.3 ms of q3's 49.1 ms. |
| **The whole script never left btree + nested loop + regex.** | No GIN, BRIN, hash join, merge join, full text search, arrays, TOAST, or ordered-set aggregates anywhere in the schema or query — checked against `/tmp/postgres` (`src/backend/access/`, `src/backend/executor/nodeX.c`, `src/backend/utils/adt/`). |
| **A real data-generation bug**: `status = 1 + (g % 5)` with 50,000 customers (a multiple of 5) means every one of a customer's 5 orders lands on the *same* residue, so all 5 orders always share one status. | `status IN ('paid','shipped','done')` returned 0 rows for ~40% of randomly chosen customers and 5 rows for the rest — q1 was silently a coin flip. |
| **`LIMIT 40` in q1's "recent orders" CTE was a no-op**, since the generated dataset only ever has 5 orders per customer. | Confirmed via the schema's own comment (`~250k orders (≈5 per customer)`). |
| **q2's cohort had no `ORDER BY` before `LIMIT`** | Non-deterministic row set across plans/runs; harmless for average-CPU scoring, bad for reproducible digests/debugging. |
| **Indexes were a plausible-looking mix, not derived from the query's actual hot paths.** | e.g. `attrs->>'tier'` and `email` indexes were never used by the transaction; `profile->>'plan'` (filtered in q2) had no index. |
| **Dataset generation is near-perfectly uniform** (`g % k` modular arithmetic), unlike Zipfian real-world shop traffic. | Acknowledged as a known simplification this redesign does *not* fully solve — see Limitations below. |

### v2: rebalance across Postgres subsystems

Goal: spread CPU across distinct executor/access-method/type subsystems so
no single one dominates, verified empirically rather than assumed. New
schema additions in `ro_cpu_setup.sql`:

| Addition | Purpose |
|---|---|
| `ro_cpu_order.search_doc tsvector` (generated, GIN-indexed) | full text search |
| `ro_cpu_product.tags text[]` (GIN-indexed) | array containment |
| `ro_cpu_product.spec_blob` (>2 KB, low-redundancy text) | TOAST compression/out-of-line fetch |
| `ro_cpu_order_ordered_at_brin` (BRIN) | non-btree range access path |
| Product catalog widened 5k → 20k | genuine cold long tail (only the first 5k ever sell) |
| Status-generation fix | mixes in order-sequence-within-customer so all 5 statuses rotate through every customer instead of collapsing to one |

`ro_cpu_txn.sql` was rewritten into 8 tagged blocks, each targeting code the
old script never touched:

| Block | Targets (Postgres source) | ~ms @ scale=1 |
|---|---|---:|
| `q_idx` | btree index scan, nested loop, window agg (`nodeIndexscan.c`, `nodeWindowAgg.c`) | 0.1 |
| `q_hashjoin` | hash join + hash aggregate over an unfiltered order/item/product join (`nodeHash.c`, `nodeHashjoin.c`) | 24 |
| `q_regex` | regex + md5, shrunk from 12,000 to ~3,600 rows (`utils/adt/regexp.c`) | 15 |
| `q_fts` | full text search: `tsvector`/`tsquery`/`ts_rank` via GIN (`utils/adt/tsvector_op.c`, `access/gin`) | 12 |
| `q_array` | array containment (`@>`) via GIN, bounded join (`utils/adt/arrayfuncs.c`, `access/gin`) | 8 |
| `q_stats` | ordered-set + statistical aggregates: `percentile_cont`, `stddev_samp`, `corr` (`nodeAgg.c`, `numeric.c`) | 13 |
| `q_toast` | out-of-line TOAST fetch + decompression (`access/heap/heaptoast.c`) | 4 |
| `q_seqscan` | plain seq scan + aggregate, no predicate/index (`nodeSeqscan.c`) | 2 |

**Result**: max single-block share dropped from **~82% (q3/regex) to
~30–34% (q_hashjoin)**, with every other block landing in a much narrower
2–15 ms band, verified with the same Docker profiling method used to find
the original problem (see `profile_v2_breakdown.sql`, kept in this folder
for future recalibration — not copied into the image).

### Calibration gotchas found along the way (worth knowing before touching this again)

- **A block can silently duplicate another block's cost.** `q_array`'s
  first version joined its GIN-matched products straight to the full
  750k-row `order_item` table with no bound; Postgres picked the exact same
  "seq-scan `order_item` + hash join" plan as `q_hashjoin`, so `q_array`
  wasn't testing the array/GIN path at all — it was just paying
  `q_hashjoin`'s cost a second time. Fixed by bounding `q_array`'s join to
  an indexed `order_id` range slice (like `q_regex`/`q_stats` already do).
  **Lesson: always `EXPLAIN` new blocks, don't assume the intended index
  gets used just because it exists.**
- **`q_hashjoin` has a hard floor (~20–24 ms) that doesn't shrink further.**
  Genuine Hash Join semantics require Postgres to fully scan the smaller of
  the two join inputs' *probe* side; since `order_item` (750k rows) has no
  narrowing predicate here, shrinking the time window from 22,400 s to
  1,500 s only cut runtime from ~51 ms to ~24 ms (not proportionally),
  because the mandatory full-table scan dominates regardless of window
  width. This is accepted and documented in the script rather than
  fought — it's a legitimate, deliberate "large scan + hash join" test case.
- **A bonus Merge Join appeared unprompted.** At the calibrated window
  width, `EXPLAIN` showed Postgres choosing a Merge Join (`nodeMergejoin.c`)
  for `q_hashjoin`'s outer product join (on top of the Hash Join for the
  order/item join) — real coverage of a third join strategy that wasn't
  deliberately engineered, just verified after the fact.
- **BRIN vs. btree "skip scan" isn't pinned.** The time-window predicate on
  `ordered_at` sometimes uses the new BRIN index and sometimes a `(status,
  ordered_at)` btree "skip scan" (a Postgres 17+ feature), depending on
  window width — since this is one monolithic SQL statement (deliberately
  kept as a single network round trip; see below), there's no way to force
  one path for this block without an `enable_*` GUC that would also affect
  every other block. Documented as "either, verified via `EXPLAIN`" rather
  than a false promise of one specific plan.
- **PL/pgSQL's plan cache is a trap for parameterized calibration
  harnesses.** The `clock_timestamp()` loop harness in
  `profile_v2_breakdown.sql` uses PL/pgSQL variables for `LIMIT`/width
  params; PL/pgSQL switches to a **genericized cached plan** after 5 calls,
  which mis-estimated a variable `LIMIT` badly enough to revert `q_array`
  to the exact bad "full scan" plan it was redesigned to avoid — measured
  at **271 ms** instead of ~8 ms. Fixed with `SET plan_cache_mode =
  force_custom_plan` in the harness. This is purely a harness artifact:
  real `pgbench` (simple query protocol) substitutes `:variables` as
  literal text before every execution, so `ro_cpu_txn.sql` itself was never
  affected — confirmed by a live `pgbench` run (84 ms avg latency, 0
  failures across 179 transactions) before and after the fix.
- `round(double precision, integer)` doesn't exist in Postgres (only
  `round(numeric, integer)`); `percentile_cont`/`stddev_samp`/`corr` needed
  explicit `::numeric` casts before rounding for the `q_stats` digest.

### Validation

Real `pgbench` runs against the redesigned schema/script (local Docker
`postgres:18`, `jit=off`, `work_mem=64MB`, `max_parallel_workers_per_gather=0`):

| Run | Result |
|---|---|
| `-c 1 -T 15 -D scale=1` | 179 txns, **0 failed**, 84.1 ms avg latency |
| `-c 4 -j 4 -T 20 -D scale=1` | 899 txns, **0 failed**, 89.3 ms avg latency |
| `-c 1 -T 15 -D scale=4` | 80 txns, **0 failed**, 187.8 ms avg latency (sub-linear vs. scale=1, since `q_hashjoin`'s floor cost doesn't scale with `-D scale`) |

### Limitations / deliberately out of scope

- **Distribution is still uniform, not Zipfian.** The product catalog now
  has a genuine cold long tail (20k products, only 5k ever sell), but
  customer/order/order-item generation is still `g % k` modular arithmetic,
  not a realistic power-law. A real "few whales, many one-off customers"
  shape would be a bigger, separate change to the data generator.
- **`jit` and `max_parallel_workers_per_gather` stay off**, matching the
  original design's rationale: this benchmark measures raw engine/CPU
  behavior, not LLVM JIT jitter or Gather scalability — those are treated
  as a separate testing axis, not something this rebalance should touch.
- **A single monolithic statement** (one `SELECT` with 8 CTEs, one
  `UNION ALL`) is deliberate: it keeps one `pgbench` transaction == one
  network round trip, which is what made the cached-RO redesign resilient
  to `netem`-simulated RTT in the first place (see history above). The
  tradeoff is that per-block planner GUCs (e.g. forcing Merge Join
  specifically) aren't possible without affecting every block.
- Weights were calibrated on one local Docker Postgres 18 instance; re-run
  `profile_v2_breakdown.sql` after any schema/query change, or on
  significantly different hardware, to confirm no block has drifted back
  into dominance.

### Re-calibrating

```bash
docker run -d --name ro-cpu-cal -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench \
  -v "$PWD:/sql:ro" postgres:18 -c shared_buffers=1GB -c jit=off
docker exec -e PGPASSWORD=bench ro-cpu-cal psql -U postgres -d bench -f /sql/ro_cpu_setup.sql
docker exec -e PGPASSWORD=bench ro-cpu-cal psql -U postgres -d bench -f /sql/profile_v2_breakdown.sql
# adjust widths in ro_cpu_txn.sql, repeat until no block dominates, then:
docker exec -e PGPASSWORD=bench ro-cpu-cal pgbench -h localhost -U postgres -d bench \
  -n -c 1 -T 20 -D scale=1 -f /sql/ro_cpu_txn.sql
```
