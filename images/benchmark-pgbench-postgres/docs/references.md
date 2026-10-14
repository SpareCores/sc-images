# References

Terms specific to this image, alphabetized. Common terms (e.g. AWS, CV, I/O,
SQL) are defined in the [glossary](https://docs.sparecores.com/glossary) of the
Spare Cores docs, which also lists the terms each image uses.

- **BRIN** (Block Range Index): A small PostgreSQL index summarizing ranges of
  table blocks, for naturally ordered data such as timestamps.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/brin.html)
- **CTE** (Common Table Expression): A named subquery of a SQL statement,
  written with `WITH`.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/queries-with.html)
- **GIN** (Generalized Inverted Index): A PostgreSQL index for values with
  many elements, e.g. arrays, JSONB or full-text search.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/gin.html)
- **GUC** (Grand Unified Configuration): The configuration settings of
  PostgreSQL, e.g. `shared_buffers`.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/config-setting.html)
- **JIT** (Just-in-Time Compilation): PostgreSQL compiling parts of a query
  to machine code at run time, worth it for long analytical queries.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/jit.html)
- **`netem`** (Network Emulator): A Linux traffic control tool to add delay or
  packet loss to a network interface.
  [SRT cookbook](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html)
- **Parallel Query**: PostgreSQL running parts of one query on several
  worker processes.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/parallel-query.html)
- **`pgbench`**: The benchmark tool shipped with PostgreSQL, which this image
  runs. [PostgreSQL docs](https://www.postgresql.org/docs/current/pgbench.html)
- **`pgtune`**: Suggests PostgreSQL settings for the hardware and workload.
  [pgtune](https://pgtune.leopard.in.ua/)
- **PL/pgSQL** (PostgreSQL Procedural Language): The procedural language of
  PostgreSQL functions.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/plpgsql.html)
- **SSL** (Secure Sockets Layer): The encryption of client connections, TLS
  nowadays. [PostgreSQL docs](https://www.postgresql.org/docs/current/ssl-tcp.html)
- **TOAST** (The Oversized-Attribute Storage Technique): How PostgreSQL stores
  large values, compressed or out of line.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/storage-toast.html)
- **TPC-B** (Transaction Processing Performance Council Benchmark B): A
  banking-style benchmark of short update transactions, the default script of
  `pgbench`. [TPC](https://www.tpc.org/tpcb/)
- **TPM** (Transactions Per Minute): The throughput this image reports.
  [Wikipedia](https://en.wikipedia.org/wiki/Transactions_per_second)
- **TPROC-C** (HammerDB's transaction processing workload): An OLTP workload
  derived from TPC-C. [HammerDB docs](https://www.hammerdb.com/docs/)
- **WAL** (Write-Ahead Logging): The log PostgreSQL writes every change to
  before applying it, for crash recovery and replication.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/wal-intro.html)
- **YCSB** (Yahoo! Cloud Serving Benchmark): A benchmark of key-value and
  cloud database workloads. [Wikipedia](https://en.wikipedia.org/wiki/YCSB)
- **Zipfian**: A skewed distribution where a few items are accessed much more
  often than the rest. [Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
