# Workloads

## `pgbench_tpcb`

This workload is pgbench's built-in `tpcb-like` script (`-b tpcb-like`) with a standard `pgbench -i -s N` schema.

Standard TPC-B-style OLTP mix (mostly-write, network- and lock-sensitive). See the [official documentation](https://www.postgresql.org/docs/current/pgbench.html) for details.

## `pgbench_ro`

This schema uses a cached CPU-heavy SQL workload.

A custom, read-only PostgreSQL benchmark sized to fit in `shared_buffers`, so this benchmark is dominated by CPU work (parse, plan, execute, join, aggregate, text/JSON/array processing) rather than disk I/O. It creates the following test data, taking up roughly 260–320 MB in memory:

- 20k products
- 50k customers
- 250k orders
- 750k order items

This benchmark can be run via the `pgbench -D scale=N -f ro_cpu_txn.sql` command.

- `-D scale=N` linearly scales the row-count knobs inside the transaction (wider slices, bigger joins) without touching the underlying dataset, so a single fixed schema can represent a range of CPU intensities.
- It uses fixed concurrency points instead of a geometric search to work with the uniformity of the test database.

The transaction is intentionally a single SQL script that runs several blocks, each touching a different PostgreSQL subsystem:

- `q_idx`: btree index scan + nested loop + window agg
- `q_hashjoin`: hash join + hash aggregate over a time slice
- `q_regex`: regex + `md5()`
- `q_fts`: full-text search via `tsvector`/GIN
- `q_array`: array containment with GIN
- `q_stats`: ordered-set/statistical aggregates
- `q_toast`: TOAST ([The Oversized-Attribute Storage Technique](https://www.postgresql.org/docs/current/storage-toast.html)) fetch/decompression
- `q_seqscan`: plain sequential scan + aggregate

At the end of the transaction, the script does the following:

- builds a single `md5(string_agg(...))` checksum
- unites all eight block outputs (`UNION ALL`)
- returns one final headline score for easy comparison

This prevents the planner from optimizing away the work, and makes it a realistic, read-only CPU benchmark, rather than a trivial constant-time query.
