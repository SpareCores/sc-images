# References

Terms specific to this image, alphabetized. Common terms (e.g. AWS, CV, I/O,
SQL) are defined in the [glossary](https://docs.sparecores.com/glossary) of the
Spare Cores docs, which also lists the terms each image uses.

- **BenchBase**: A framework that benchmarks SQL databases, the successor to
  OLTPBench. It runs workloads such as YCSB and a Wikipedia workload against
  any relational database with a JDBC (Java Database Connectivity) driver, and
  collects per-transaction-type latency and throughput logs.
  [GitHub](https://github.com/cmu-db/benchbase)
- **BRIN** (Block Range Index): A PostgreSQL index type that stores only a
  summary, such as the minimum and maximum value, of each range of consecutive
  table blocks. It is tiny compared to a B-tree, and works well on large tables
  whose values follow their physical order, such as timestamps of appended
  rows. [PostgreSQL docs](https://www.postgresql.org/docs/current/brin.html)
- **CTE** (Common Table Expression): A named temporary result set, defined with
  `WITH` at the start of a SQL statement, that the rest of the statement can
  query like a table. CTEs make complex queries easier to read, and recursive
  CTEs can walk hierarchies such as trees.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/queries-with.html)
- **concurrency points**: The client counts at which a database benchmark
  measures throughput and latency, each with that number of connections open
  for a fixed time. Measuring several concurrency points, from one client up
  to a count derived from the server's size, shows how throughput and latency
  change with load.
- **GIN** (Generalized Inverted Index): A PostgreSQL index type that maps each
  element of a composite value, such as an array item, a key of a JSON
  document, or a word of a text, to the rows that contain it. It speeds up
  containment and full-text searches, at the cost of slower writes than a
  B-tree.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/gin.html)
- **GUC** (Grand Unified Configuration): The configuration system of
  PostgreSQL, and by extension its individual settings, such as
  `shared_buffers` or `work_mem`. A setting can be set in `postgresql.conf`, on
  the command line, per database or role, or per session with `SET`.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/config-setting.html)
- **HammerDB**: An open-source database benchmark engine. It runs standardized
  workloads, such as TPROC-C, with the same test definitions on SQL Server,
  Db2, Oracle, PostgreSQL, MariaDB, and MySQL, so that results are comparable
  across database systems. Runs can be scripted and repeated automatically.
  [HammerDB](https://www.hammerdb.com/)
- **JIT** (Just-in-Time Compilation): A PostgreSQL feature that compiles parts
  of a query, such as its expressions, to machine code with LLVM at run time.
  It speeds up long, CPU-bound analytical queries, but on short queries the
  compile time can outweigh the gain.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/jit.html)
- **`netem`** (Network Emulator): A Linux traffic control queueing discipline
  that adds delay, jitter, packet loss, duplication or reordering to the
  packets leaving a network interface. It is used to test how software behaves
  on slow or unreliable networks.
  [SRT cookbook](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html)
- **Parallel Query**: A PostgreSQL feature that splits one query across several
  worker processes, for example to scan, join or aggregate a large table on
  several CPU cores at once. `max_parallel_workers_per_gather` limits how many
  workers one part of a query can use.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/parallel-query.html)
- **`pgbench`**: The benchmark tool shipped with PostgreSQL. It runs a SQL
  script repeatedly over many concurrent client sessions, and reports the
  throughput in transactions per second and the latency. Its built-in script is
  modeled on TPC-B, and custom scripts can define any other workload.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/pgbench.html)
- **`pgtune`**: A web tool that suggests PostgreSQL settings, such as
  `shared_buffers` and `work_mem`, from the memory, CPU count, storage type and
  workload of a server. Its suggestions are a starting point for tuning rather
  than an optimum. [pgtune](https://pgtune.leopard.in.ua/)
- **PL/pgSQL** (PostgreSQL Procedural Language): The built-in procedural
  language of PostgreSQL, which adds variables, loops, conditionals and error
  handling to SQL. It is used to write functions, procedures and triggers that
  run inside the database server.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/plpgsql.html)
- **SSL** (Secure Sockets Layer): A protocol that encrypts network connections
  and verifies the identity of the server, and optionally of the client, with
  certificates. TLS has replaced it, but SSL is still the common name for both,
  including in PostgreSQL's settings.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/ssl-tcp.html)
- **`sysbench`**: A scriptable, multi-threaded benchmark tool built on LuaJIT.
  Its tests cover CPU, memory, file I/O, threads, and mutexes, and its Lua
  scripts run transactional database workloads against MySQL and PostgreSQL.
  It reports rates and latency percentiles.
  [GitHub](https://github.com/akopytov/sysbench)
- **TOAST** (The Oversized-Attribute Storage Technique): The mechanism with
  which PostgreSQL stores values too large for a regular 8 kB page, by
  compressing them, moving them out of line into a separate table, or both. It
  applies automatically to large values, such as long texts or JSON documents.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/storage-toast.html)
- **TPC-B** (Transaction Processing Performance Council Benchmark B): A
  database benchmark of short, update-heavy banking transactions, each
  updating an account, a teller and a branch balance and logging the change.
  The TPC retired it in 1995, but the default script of `pgbench` is still
  modeled on it. [TPC](https://www.tpc.org/tpcb/)
- **TPM** (Transactions Per Minute): A throughput measure of transaction
  processing systems: the number of transactions they complete per minute.
  TPC-C reports a variant of it, tpmC, which counts only new-order
  transactions, while tools such as `pgbench` report transactions per second
  (TPS), which is 1/60 of TPM.
  [Wikipedia](https://en.wikipedia.org/wiki/Transactions_per_second)
- **TPROC-C** (HammerDB's transaction processing workload): The OLTP workload
  of the HammerDB benchmark tool, derived from the TPC-C specification of an
  order-entry system with warehouses, orders and payments. It is not an
  official TPC-C implementation, so its results are not comparable with
  published TPC-C results. [HammerDB docs](https://www.hammerdb.com/docs/)
- **WAL** (Write-Ahead Logging): The log of changes that PostgreSQL writes to
  durable storage before it changes the data files. After a crash, replaying
  the WAL restores a consistent state, and shipping it to other servers drives
  replication and point-in-time recovery. On write-heavy workloads, the speed
  of WAL writes often limits throughput.
  [PostgreSQL docs](https://www.postgresql.org/docs/current/wal-intro.html)
- **YCSB** (Yahoo! Cloud Serving Benchmark): An open-source benchmark
  framework, released by Yahoo! in 2010, for key-value and NoSQL databases. It
  runs standard workloads that mix reads, updates, inserts and scans in
  different proportions, and is widely used to compare cloud data stores.
  [Wikipedia](https://en.wikipedia.org/wiki/YCSB)
- **Zipfian**: A skewed probability distribution, named after Zipf's law, in
  which an item's frequency is inversely proportional to its rank, so a few
  items are accessed far more often than the rest. Benchmarks use it to model
  real-world hot spots, such as popular products or users.
  [Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
