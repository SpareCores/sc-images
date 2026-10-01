# Workloads

## `pgbench_tpcb`

This workload is `pgbench`'s built-in `tpcb-like` script (`-b tpcb-like`) with a
standard `pgbench -i -s N` schema. It is a standard TPC-B (Transaction
Processing Performance Council Benchmark B)-style OLTP (Online Transaction
Processing) mix (mostly-write, network- and lock-sensitive).

See the [official
documentation](https://www.postgresql.org/docs/current/pgbench.html) for
details.

## `pgbench_ro`

`pgbench_ro` is a custom read-only workload with one transaction composed of
eight query blocks. They exercise joins, aggregation, indexes, full-text search,
arrays, and other PostgreSQL subsystems to emphasize CPU work rather than disk
I/O. The setup creates the following test data with an overall estimate of
~260–320 MB for the data plus indexes:

- 20k products
- 50k customers
- 250k orders
- 750k order items

Run this workload via Docker. The harness invokes `pgbench` with
`-D scale=N` and `-f ro_cpu_txn.sql` internally.

- `-D scale=N` linearly scales the row-count knobs inside the transaction (wider
  slices, bigger joins) without touching the underlying dataset, so a single
  fixed schema can represent a range of CPU intensities.
- It uses fixed concurrency points instead of a geometric search to work with
  the uniformity of the test database.

The transaction is intentionally a single SQL script that runs several blocks,
each touching a different PostgreSQL subsystem:

- `q_idx`: btree index scan + nested loop + window agg
- `q_hashjoin`: hash join + hash aggregate over a time slice
- `q_regex`: regex + `md5()`
- `q_fts`: full-text search via `tsvector`/GIN
- `q_array`: array containment with GIN
- `q_stats`: ordered-set/statistical aggregates
- `q_toast`: TOAST ([The Oversized-Attribute Storage
  Technique](https://www.postgresql.org/docs/current/storage-toast.html))
  fetch/decompression
- `q_seqscan`: plain sequential scan + aggregate

At the end of each transaction, the script combines the eight block outputs
with `UNION ALL` and hashes them into one `md5(string_agg(...))` checksum. The
checksum makes the result depend on the block outputs and prevents the planner
from optimizing away their work; it is not the benchmark score. The harness then
converts `pgbench` throughput to TPM (transactions per minute), the published
headline score.
