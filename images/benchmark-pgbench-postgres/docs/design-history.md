# `pgbench_ro`

## How we got here - tools and approaches

We benchmarked the benchmarks before trusting one. In rough order:

- **sysbench, HammerDB TPROC-C, and BenchBase (Wikipedia read-only and YCSB datasets), alongside pgbench:**

  On regular block storage vs. `tmpfs`, with baseline vs. host-tuned PostgreSQL configs.
  - `tmpfs` lifted write-heavy OLTP results substantially (~10–25% and more), which proved the point that those suites were measuring storage and WAL behavior more than the server. `tmpfs` is not an option on DBaaS anyway, so it only covers IaaS.
  - Their warehouse/scale-factor sizing also couldn't cover a fleet spanning 1 vCPU to thousands of vCPUs
- **A systematic PostgreSQL config (GUC) sweep:**

  21 experiments on a 32-vCPU host, where the winning combination (modest `work_mem`, `io_uring`, small WAL buffers, parallel gather off, right-sized `shared_buffers`) gained ~20% throughput over the baseline.
  - Tuning clearly matters, which is why production runs delegate it:
    - pgtune-style host tuning on IaaS
    - the vendor's own tuning on DBaaS
- **Latency and pipelining experiments:**

  Measurements for `pgbench -S` under induced network delay, then under a custom-built sliding-window `--pipeline-depth` mode in a `pgbench` fork.
  - Pipelining does rescue RTT-bound scripts (~10× queries/s at +5 ms delay), but for a CPU-bound transaction it adds nothing at low concurrency and actively collapses throughput at high concurrency.
  - **Conclusion:** make the transaction heavy instead of pipelining a light one.
    - serial mode, a fixed `{1, V/2, V, 2·V}` concurrency profile, and a TPM score
- **Outcome**:
  - See `pgbench_ro` workload in the [description](./workloads.md#pgbench_ro)
  - `pgbench_tpcb` is kept as a secondary classic-OLTP (TPC-B-like) reference

A [blog post](https://sparecores.com/articles) with the detailed findings is planned.

## v1: From trivial `pgbench -S` to a cached multi-query script

Plain `pgbench -S` (one `SELECT` by primary key) turned out to be too cheap per-transaction to say anything meaningful about CPU under network latency. Under `netem`-simulated RTT its TPS collapsed almost entirely from RTT, and not from server work (~98% loss at a single connection with +5 ms one-way delay, while a CPU-heavy transaction under the same delay barely moved).
This motivated a **cached, multi-query, CPU-heavy** custom script, sized for 100–130 ms of server CPU time per transaction at `-c 1`, using a small (~170 MB) schema that fits in the `shared_buffers` so I/O never becomes a bottleneck, so that RTT stays a small fraction of total latency instead of dominating it.

The original transaction ran four blocks, `q1`–`q4`:

- **q1:** One customer's recent paid/shipped/done orders → join → window functions (index scan + nested loop + `WindowAgg`)
- **q2:** A region/plan cohort → their orders → line items, grouped by status (broader join + `GroupAggregate`)
- **q3:** A 12,000-row, contiguous `order_id` slice, checked against two regexes and hashed with `md5()` (regex + JSON + hashing)
- **q4:** The top-N buyers of one product (`GROUP BY` + `ORDER BY` + `LIMIT`)

### What was actually wrong with it

Profiling this transaction locally (fresh `postgres:18` in Docker, `jit=off`, `EXPLAIN (ANALYZE, BUFFERS)` plus `clock_timestamp()` loops per block) turned up several real problems, not just "regex is slow":

<!-- markdownlint-disable MD033 -->
| Finding | Evidence |
| - | - |
| q3's regex+md5 alone was **~70–82%** of the whole transaction. | Per-block timing: q1: 0.09 ms, q2: 9.8 ms, q3: 49.1 ms, q4: 0.10 ms (total 59.6 ms).<br>Inside q3: regex alone was 42.3 ms of q3's 49.1 ms. |
| The whole script **never left** btree + nested loop + regex. | No GIN, BRIN, hash join, merge join, full text search, arrays, TOAST, or ordered-set aggregates anywhere in the schema or query.<br>Checked against `/tmp/postgres` (`src/backend/access/`, `src/backend/executor/nodeX.c`, `src/backend/utils/adt/`). |
| A real **data-generation bug**: `status = 1 + (g % 5)` with 50,000 customers (a multiple of 5) means every one of a customer's 5 orders lands on the *same* residue, so all 5 orders always share one status. | `status IN ('paid','shipped','done')` returned 0 rows for ~40% of randomly chosen customers and 5 rows for the rest.<br>q1 was silently a coin flip. |
| `LIMIT 40` in q1's "recent orders" CTE was a **no-op**, since the generated dataset only ever has 5 orders per customer. | Confirmed via the schema's own comment (`~250k orders (≈5 per customer)`). |
| q2's cohort had **no `ORDER BY` before `LIMIT`** | Non-deterministic row set across plans/runs; harmless for average-CPU scoring, bad for reproducible digests and debugging. |
| Indexes were a plausible-looking mix, but were **not derived from the query's actual hot paths**. | e.g. `attrs->>'tier'` and `email` indexes were never used by the transaction; `profile->>'plan'` (filtered in q2) had no index. |
| Dataset generation is **near-perfectly uniform** (`g % k` modular arithmetic), unlike Zipfian real-world shop traffic. | Acknowledged as a known simplification this redesign does *not* fully solve. See [Limitations](./limitations.md) for details. |

## v2: Rebalance across PostgreSQL subsystems

**Goal:** spread CPU time across distinct executor/access-method/type subsystems so no single one dominates, verified empirically rather than assumed. New schema additions in `ro_cpu_setup.sql`:

| Addition | Purpose |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `ro_cpu_order.search_doc tsvector` (generated, GIN-indexed) | full text search |
| `ro_cpu_product.tags text[]` (GIN-indexed) | array containment |
| `ro_cpu_product.spec_blob` (>2 KB, low-redundancy text) | TOAST compression/out-of-line fetch |
| `ro_cpu_order_ordered_at_brin` (BRIN) | non-btree range access path |
| Product catalog widened 5k → 20k | genuine cold long tail (only the first 5k ever sell) |
| Status-generation fix | mixes in order-sequence-within-customer so all 5 statuses rotate through every customer instead of collapsing to one |

`ro_cpu_txn.sql` was rewritten into 8 tagged blocks, each targeting code the old script never touched:

| Block | Targets (PostgreSQL source) | ~ms @ scale=1 |
| ------------ | --------------------------------------------------------------------------------------------------------- | ------------: |
| `q_idx` | btree index scan, nested loop, window agg (`nodeIndexscan.c`, `nodeWindowAgg.c`) | 0.1 |
| `q_hashjoin` | hash join + hash aggregate over an unfiltered order/item/product join (`nodeHash.c`, `nodeHashjoin.c`) | 24 |
| `q_regex` | regex + md5, shrunk from 12,000 to ~3,600 rows (`utils/adt/regexp.c`) | 15 |
| `q_fts` | full text search: `tsvector`/`tsquery`/`ts_rank` via GIN (`utils/adt/tsvector_op.c`, `access/gin`) | 12 |
| `q_array` | array containment (`@>`) via GIN, bounded join (`utils/adt/arrayfuncs.c`, `access/gin`) | 8 |
| `q_stats` | ordered-set + statistical aggregates: `percentile_cont`, `stddev_samp`, `corr` (`nodeAgg.c`, `numeric.c`) | 13 |
| `q_toast` | out-of-line TOAST fetch + decompression (`access/heap/heaptoast.c`) | 4 |
| `q_seqscan` | plain seq scan + aggregate, no predicate/index (`nodeSeqscan.c`) | 2 |

**Result**:

- Max single-block share dropped from **~82% (q3/regex)** to **~30–34% (q_hashjoin)**
- Every other block landed in a much narrower 2–15 ms band
- Verified with the same Docker profiling method used to find the original problem
  - (see `profile_v2_breakdown.sql`, kept in this folder for future recalibration, not copied into the image)

### Calibration gotchas found along the way (worth knowing before touching this again)

- **A block can silently duplicate another block's cost.**
  - `q_array`'s first version joined its GIN-matched products straight to the full 750k-row `order_item` table with no bound
  - PostgreSQL picked the exact same "seq-scan `order_item` + hash join" plan as `q_hashjoin`, so `q_array` wasn't testing the array/GIN path at all. It was just paying `q_hashjoin`'s cost a second time.
  - Fixed by bounding `q_array`'s join to an indexed `order_id` range slice (like `q_regex`/`q_stats` already do).
  - **Lesson:** always `EXPLAIN` new blocks, don't assume the intended index gets used just because it exists.
- **`q_hashjoin` has a hard floor (~20–24 ms) that doesn't shrink further.**
  - Genuine Hash Join semantics require PostgreSQL to fully scan the smaller of the two join inputs' *probe* side; since `order_item` (750k rows) has no narrowing predicate here, shrinking the time window from 22,400 s to 1,500 s only cut runtime from ~51 ms to ~24 ms (not proportionally), because the mandatory full-table scan dominates regardless of window width. This is accepted and documented in the script rather than fought — it's a legitimate, deliberate "large scan + hash join" test case.
- **A bonus Merge Join appeared unprompted.**
  - At the calibrated window width, `EXPLAIN` showed PostgreSQL choosing a Merge Join (`nodeMergejoin.c`) for `q_hashjoin`'s outer product join (on top of the Hash Join for the order/item join). This is real coverage of a third join strategy that wasn't deliberately engineered, just verified after the fact.
- **BRIN vs. btree "skip scan" isn't pinned.**
  - The time-window predicate on `ordered_at` sometimes uses the new BRIN index and sometimes a `(status, ordered_at)` btree "skip scan" (a PostgreSQL 17+ feature), depending on window width.
  - Since this is one monolithic SQL statement (deliberately kept as a single network round trip; see below), there's no way to force one path for this block without an `enable_*` GUC, which would also affect every other block.
  - Documented as "either, verified via `EXPLAIN`" rather than a false promise of one specific plan.
- **PL/pgSQL's plan cache is a trap for parameterized calibration harnesses.**
  - The `clock_timestamp()` loop harness in `profile_v2_breakdown.sql` uses PL/pgSQL variables for `LIMIT`/width params
- PL/pgSQL switches to a **genericized cached plan** after 5 calls, which mis-estimated a variable `LIMIT` badly enough to revert `q_array` to the exact bad "full scan" plan it was redesigned to avoid
  - Measured at **271 ms** instead of ~8 ms
    - Fixed with `SET plan_cache_mode = force_custom_plan` in the harness.
  - **This is purely a harness artifact:** real `pgbench` (simple query protocol) substitutes `:variables` as literal text before every execution, so `ro_cpu_txn.sql` itself was never affected
    - Confirmed by a live `pgbench` run (84 ms avg latency, 0 failures across 179 transactions) before and after the fix
- `round(double precision, integer)` doesn't exist in PostgreSQL (only `round(numeric, integer)`)
- `percentile_cont`/`stddev_samp`/`corr` needed explicit `::numeric` casts before rounding for the `q_stats` digest.

## Validation

Real `pgbench` runs against the redesigned schema/script (local Docker `postgres:18`, `jit=off`, `work_mem=64MB`, `max_parallel_workers_per_gather=0`):

| Run | Result |
| --- | --- |
| `-c 1 -T 15 -D scale=1` | 179 txns, **0 failed**, 84.1 ms avg latency |
| `-c 4 -j 4 -T 20 -D scale=1` | 899 txns, **0 failed**, 89.3 ms avg latency |
| `-c 1 -T 15 -D scale=4` | 80 txns, **0 failed**, 187.8 ms avg latency (sub-linear vs. scale=1, since `q_hashjoin`'s floor cost doesn't scale with `-D scale`) |

## Recalibration

```bash
docker run -d --name ro-cpu-cal -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench \
  -v "$PWD:/sql:ro" postgres:18 -c shared_buffers=1GB -c jit=off
docker exec -e PGPASSWORD=bench ro-cpu-cal psql -U postgres -d bench -f /sql/ro_cpu_setup.sql
docker exec -e PGPASSWORD=bench ro-cpu-cal psql -U postgres -d bench -f /sql/profile_v2_breakdown.sql
# adjust widths in ro_cpu_txn.sql, repeat until no block dominates, then:
docker exec -e PGPASSWORD=bench ro-cpu-cal pgbench -h localhost -U postgres -d bench \
  -n -c 1 -T 20 -D scale=1 -f /sql/ro_cpu_txn.sql
```
