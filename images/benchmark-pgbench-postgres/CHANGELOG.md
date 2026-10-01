# Changelog

## Current RO workload: v2 subsystem rebalance

The v2 redesign spreads transaction work across PostgreSQL executor, access
method, and data type subsystems so no single block dominates. The schema adds
full-text search, arrays, TOAST data, BRIN, and related access paths. The
transaction in `ro_cpu_txn.sql` uses eight tagged blocks:

| Block | Workload area | Approx. ms at scale 1 |
| --- | --- | ---: |
| `q_idx` | B-tree index scan, nested loop, window aggregate | 0.1 |
| `q_hashjoin` | Hash join and hash aggregate | 24 |
| `q_regex` | Regex and `md5()` | 15 |
| `q_fts` | Full-text search using `tsvector`, `tsquery`, `ts_rank`, and GIN | 12 |
| `q_array` | Array containment using GIN and a bounded join | 8 |
| `q_stats` | Ordered-set and statistical aggregates | 13 |
| `q_toast` | Out-of-line TOAST fetch and decompression | 4 |
| `q_seqscan` | Sequential scan and aggregate | 2 |

The maximum single-block share fell from about 82% in the old regex block to
about 30–34% in `q_hashjoin`; other blocks measured in a narrower 2–15 ms band.
The query plans and block timings were checked using the same local Docker
profiling method used to investigate v1.

### Calibration findings

- The first `q_array` version joined GIN-matched products to all 750,000
  `order_item` rows. PostgreSQL chose the same sequential scan and hash join as
  `q_hashjoin`, duplicating its cost rather than isolating the array/GIN path.
  Bounding the join to an indexed `order_id` slice corrected this. New blocks
  should be checked with `EXPLAIN`; the presence of an index does not guarantee
  that the planner uses it.
- `q_hashjoin` has a floor of about 20–24 ms. The `order_item` probe side must
  be scanned even when the time window narrows, so reducing the window from
  22,400 seconds to 1,500 seconds only reduced the measured time from about 51
  ms to 24 ms. This is retained as a deliberate large-scan/hash-join case.
- PostgreSQL selected a Merge Join for the outer product join in `q_hashjoin` in
  addition to its Hash Join. This was observed in `EXPLAIN`, not forced.
- The time-window predicate may use BRIN or a B-tree skip scan over `(status,
  ordered_at)`, depending on window width. The single-statement design prevents
  per-block planner GUCs without affecting the other blocks, so the
  documentation records either observed path rather than promising one.
- The PL/pgSQL timed-loop harness in `profile_v2_breakdown.sql` switches to a
  generic cached plan after five calls. A variable `LIMIT` was then estimated
  poorly, causing `q_array` to use the unbounded full-table plan and take 271 ms
  instead of about 8 ms. `SET plan_cache_mode = force_custom_plan` fixes this
  harness artifact. It does not affect the real `pgbench` script: `pgbench`
  substitutes its variables as literals before planning each execution. A direct
  `pgbench` run confirmed the production script was unaffected.
- PostgreSQL has no `round(double precision, integer)` overload; the relevant
  values must be cast to `numeric` before rounding. `percentile_cont`,
  `stddev_samp`, and `corr` likewise needed explicit numeric casts for the
  `q_stats` digest.

### Validation

Real `pgbench` runs against the redesigned schema and script used local Docker
`postgres:18`, `jit=off`, `work_mem=64MB`, and
`max_parallel_workers_per_gather=0`:

| Run | Result |
| --- | --- |
| `-c 1 -T 15 -D scale=1` | 179 transactions, 0 failed, 84.1 ms average latency |
| `-c 4 -j 4 -T 20 -D scale=1` | 899 transactions, 0 failed, 89.3 ms average latency |
| `-c 1 -T 15 -D scale=4` | 80 transactions, 0 failed, 187.8 ms average latency; growth is sub-linear because the `q_hashjoin` floor does not scale with `-D scale` |

### Recalibration procedure

Run the setup and profiling scripts against a fresh PostgreSQL 18 database, then
run the transaction directly after adjusting its block widths:

```bash
docker run -d --name ro-cpu-cal -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench \
  -v "$PWD:/sql:ro" postgres:18 -c shared_buffers=1GB -c jit=off
docker exec -e PGPASSWORD=bench ro-cpu-cal psql -U postgres -d bench -f /sql/ro_cpu_setup.sql
docker exec -e PGPASSWORD=bench ro-cpu-cal psql -U postgres -d bench -f /sql/profile_v2_breakdown.sql
# Adjust widths in ro_cpu_txn.sql, repeat until no block dominates, then:
docker exec -e PGPASSWORD=bench ro-cpu-cal pgbench -h localhost -U postgres -d bench \
  -n -c 1 -T 20 -D scale=1 -f /sql/ro_cpu_txn.sql
```

Re-run `profile_v2_breakdown.sql` after any
schema/query change, or on significantly different hardware, to confirm no
block has drifted back into dominance.

## Initial custom workload: v1

Plain `pgbench -S` (one primary-key `SELECT`) was too cheap per transaction to
measure CPU behavior under network latency. With `netem` ([Linux network
emulator](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html))
simulating RTT (round-trip time), its throughput fell by about 98% at one
connection with an additional 5 ms of one-way delay, while the CPU-heavy
transaction barely changed.

The first custom script used a cached, multi-query transaction sized for 100–130
ms of server CPU time at `-c 1`, with a roughly 170 MB schema intended to fit in
`shared_buffers`. It had four blocks:

- **q1:** One customer's recent paid/shipped/done orders, joined and processed
  with window functions (index scan, nested loop, and `WindowAgg`).
- **q2:** A region/plan cohort joined to orders and line items, grouped by
  status (broader join and `GroupAggregate`).
- **q3:** A contiguous 12,000-row `order_id` slice checked against two regular
  expressions and hashed with `md5()` (regex, JSON, and hashing).
- **q4:** The top buyers of one product (`GROUP BY`, `ORDER BY`, and `LIMIT`).

Profiling a fresh `postgres:18` container with `jit=off`, `EXPLAIN (ANALYZE,
BUFFERS)`, and per-block `clock_timestamp()` loops identified these issues:

| Finding | Evidence |
| --- | --- |
| q3's regex and `md5()` work took about 70–82% of the transaction. | Timings were q1 0.09 ms, q2 9.8 ms, q3 49.1 ms, and q4 0.10 ms (59.6 ms total); regex alone took 42.3 ms in q3. |
| The script only exercised B-tree, nested-loop, and regex work. | It did not use GIN, BRIN, hash or merge joins, full-text search, arrays, TOAST, or ordered-set aggregates. The local PostgreSQL source was checked under `src/backend/access/`, `src/backend/executor/nodeX.c`, and `src/backend/utils/adt/`. |
| A data-generation bug made all five orders for a customer share one status. | With 50,000 customers, a multiple of five, `status = 1 + (g % 5)` assigned each customer's orders the same residue. `status IN ('paid','shipped','done')` returned zero rows for about 40% of sampled customers and five for the rest, making q1 inconsistent. |
| q1's `LIMIT 40` did not limit results. | The generated dataset contained only five orders per customer. |
| q2 limited a cohort without `ORDER BY`. | The selected rows could vary across plans and runs, making digests and debugging less reproducible. |
| Several indexes did not match the query's actual hot paths. | The `attrs->>'tier'` and `email` indexes were unused, while q2 filtered on `profile->>'plan'` without an index. |
| Generated data was nearly uniform rather than Zipfian. | The `g % k` arithmetic remained a known simplification in v2; see `docs/limitations.md`. |

## Experiments that informed the design

### Dataset comparisons

`sysbench`, HammerDB TPROC-C, BenchBase (Wikipedia read-only and YCSB datasets),
and `pgbench` were compared using regular block storage and `tmpfs`, with
baseline and host-tuned PostgreSQL configurations.

- `tmpfs` improved write-heavy OLTP results by about 10–25% or more, indicating
  that those suites were measuring storage and WAL behavior as well as server
  performance.
- `tmpfs` was not available for DBaaS comparisons. Warehouse and scale-factor
  sizing also did not cover the tested range from 1 vCPU to thousands of vCPUs.

### PostgreSQL configuration sweep

Twenty-one experiments on a 32-vCPU host found a winning configuration with
about 20% more throughput than the baseline. The combination used modest
`work_mem` and `io_uring`, small WAL buffers, parallel gather disabled, and
right-sized `shared_buffers`. The results motivated per-host tuning for IaaS and
provider-managed tuning for DBaaS.

### Latency and pipelining

The experiments measured `pgbench -S` under induced network delay, then used a
custom sliding-window `--pipeline-depth` mode in a `pgbench` fork. Pipelining
improved RTT-bound scripts by about 10x at +5 ms delay, but added nothing to a
CPU-bound transaction at low concurrency and reduced throughput at high
concurrency. The resulting design made the transaction heavier rather than
pipelining a light transaction: serial mode, fixed `{1, V/2, V, 2·V}`
concurrency, and a TPM score.

The outcome is a custom, cached, CPU-heavy `pgbench_ro` workload; the built-in
`pgbench_tpcb` workload remains available as a conventional TPC-B-like
reference.
