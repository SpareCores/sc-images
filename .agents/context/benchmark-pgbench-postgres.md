# Context: benchmark-pgbench-postgres

## Why does this benchmark exist?

Spare Cores monitors and publishes empirical performance data for over 5,000
cloud server types in the [Navigator project](https://sparecores.com/servers). A
proper Relational Database Management System (RDBMS) benchmark was missing.
Earlier stand-ins were PassMark database operations (doesn't scale to 32+
vCPUs), Redis (not a relational database), raw CPU speed, and memory bandwidth
(useful proxies, but not a database workload).

## Where is the benchmarking client run?

The client is agnostic if the database server is local or remote, so it can run
on the database server or connect to it remotely, and the related difference in
latency should not materially change the benchmark results, as (1) the workload
is heavy and running for ~100ms, so 1-5ms latency is negligible, and (2) the
client load is minimal (<1% CPU and ~10 MB RSS on an 8-vCPU run), so should not
affect the database server performance.

As the client is containerized, it needs permissions and e.g. Docker to be able
to run, which is not doable on managed databases, so we always start a separate
benchmarking client VM that connects remotely to the managed database server.

IaaS does not use a separate client VM: the client and the database ran on the
same node to save on infrastructure costs.

## Does production apply OS-level tuning?

Production runs do not apply `sysctl` or other host OS tweaks.

Privileged mode, host networking, `seccomp=unconfined`, and ulimits (high
`nofile` and an unlimited `memlock` unlocking huge pages and `io_uring` for the
server) are container settings applied by `sc-inspector` orchestration. The
image itself starts PostgreSQL with `nice -n -20`; privileged mode is what lets
that higher priority take effect.

## What is the primary advantage versus other database benchmarks?

Our primary goal was to have a benchmarking methodology that scales across
instance sizes: from small instances (e.g. 1 vCPU and 2 GB of RAM) to large
nodes with hundreds of vCPUs.

Other database benchmarks usually focus on storage, network throughput, a single
database operation, or one production workload -- while we focus on CPU and
memory speed of the instance, as disk and network are usually configured
alongside the instance type.

## What is the size of the benchmarking database?

The database schema is static and the data is generated on the fly via the
`ro_cpu_setup.sql` script.

Historically we experimented with a `PGBENCH_RO_CPU_SCHEMA_GIB` setting to
dynamically tune the dataset size, but it's fixed now for all cloud server
types; the image still emits the old constant as `schema_gib`.
`pg_database_size` reports 303 MiB after a fresh import into PostgreSQL 18.

## Are the DBaaS server and its benchmarking client always placed in the same availability zone and connected over private VPC addresses?

It's always private VPC, but actual placement depends on the vendor. E.g. for
AWS, it's always the same AZ, but others might be different zones of the same
region. Note that since the benchmark is less sensitive to latency, the client
doesn't necessarily need to be in the same AZ.

## Do IaaS and DBaaS runs use the same hardware?

Yes, in general, it's the same hardware: the cloud provider provisions the
managed database on a given cloud server type (referenced as the underlying
`server_id` in the Spare Cores Navigator data), so essentially the same silicon.

## Is the PostgreSQL version held constant across runs?

Only the major version is fixed, the minor version is allowed to vary:

- Many DBaaS providers we benchmark do not allow pinning the minor version and
  apply minor upgrades automatically.
- The local PostgreSQL version is also kept at 18 for consistency by building on
  the `postgres:18` Docker image.

## Why is disk performance excluded?

Database throughput usually depends on the disk first (IOPS, then bandwidth). In
the cloud, that disk is almost always network-attached block storage that the
user provisions independently of the server type, so it says little about the
server itself. Provisioning volumes fast enough never to be the bottleneck on
all ~5,000 server types would also be prohibitively expensive. We therefore
eliminate disk from the measurement and score the engine's CPU and memory
performance.

## Why is network performance excluded, and how is RTT handled?

With a remote client, bandwidth and especially latency between client and server
can dominate short workloads. Initially, we tried to minimize RTT through server
and client placement in the same AZ or at least same region, but random latency
glitches still distorted the results, so we designed the workload so that the
remaining RTT is a rounding error. This was achieved by using heavy read
operations that run for ~100ms instead of the default read-only `pgbench`.

## Why is the DBaaS engine not tuned?

The managed service's tuning is part of what is being measured, so the
vendor-managed configuration is left untouched by design. The harness also
cannot assume superuser access or control over GUCs on DBaaS. IaaS servers are
tuned per host with `pgtune` because a configuration sweep showed that tuning
matters (about 20% more throughput on a 32-vCPU host).

## Why are JIT and parallel query disabled for `pgbench_ro`?

The benchmark measures raw engine and CPU behavior. JIT variance and Gather
scalability are treated as a separate testing axis.

## Why is the `pgbench_ro` transaction a single statement?

One `SELECT` whose eight query blocks are CTEs combined with `UNION ALL` keeps
one `pgbench` transaction equal to one network round trip, which is what makes
the workload resilient to RTT. The trade-off is that per-block planner GUCs (for
 example, forcing a Merge Join for one block) cannot be set without affecting
every block.

## Why does `pgbench_ro` use a fixed concurrency profile?

It came out of the latency and pipelining experiments. Pipelining helped
RTT-bound scripts but added nothing to a CPU-bound transaction at low
concurrency and reduced throughput at high concurrency. We chose to make the
transaction heavy instead of pipelining a light one: serial mode, a fixed `{1,
V/2, V, 2·V}` concurrency profile (`V` stands for the number of vCPUs), and a
TPM score.

## Why is `pgbench_tpcb` kept?

It's currently unused due to being disk-limited, but we kept it to potentially
revisit later. The main and our only PostgreSQL benchmark in production is
`pgbench_ro`.

## Is the uniform data distribution intentional?

It is a known simplification, not a goal. The product catalog has a deliberate
cold long tail (20,000 products, of which only the first 5,000 ever sell), but
customer, order, and order-item generation still uses `g % k` modular arithmetic
rather than a realistic power-law ("few whales, many one-off customers")
distribution.

## Who helped shape the design?

We consulted benchANT (https://benchant.com) while iterating on the tools,
configurations, and design constraints -- and we highly appreciate their help!

## Which servers is `pgbench_ro` run on in production?

Only servers with at least 2 GiB of RAM, as reported by the vendor. On smaller
 nodes the dataset does not fit entirely in `shared_buffers`, so there might be
some disk overhead. This 2 GiB minimum is also the smallest instance size the
design targets.


## Does a measured `pgbench_ro` run write WAL?

"No WAL" refers to the steady state, i.e. the actual measured benchmark.

## Where does the `m9g.24xlarge` timing example come from?

It is an actual production run, recorded at
https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m9g.24xlarge/pgbench_postgres_ro_durable/stdout

## Which open edge cases are deliberately left out of the docs?

- Planner statistics and hint bits after a CDN restore, and the generic-plan
  risk under `-M prepared`: probably fine, to be double-checked later.
- An upper bound for `SC_CPU_SCALE`, and truncated CDN downloads: guards may be
  added to the code later.

None of these need to be covered in the docs.

## What is `storage_gib`?

A fixed label for the benchmarking environment (the 128 GiB `sc-inspector`
root volume), not a measurement and not a benchmark metric. The image reports
128 in every standalone run, including a plain `docker run` outside
production.

## Should the docs cover leftovers from earlier experiments?

No. The benchmark grew out of running and sizing standard benchmark suites
before writing our own, so the code may still contain leftovers from that
experimentation. Document what is in use, not those leftovers.

Some experiment artifacts are kept at
https://github.com/SpareCores/sc-db-benchmark-tmp, but not all of them.

## Where were the `pgbench_ro` block weights calibrated?

First on a local machine, then on the cloud test instance (which changed between
cloud providers because of credit outages). This detail does not matter for the
docs.

## Does the `m9g.24xlarge` example's extra time need a breakdown?

No. Which steps account for the time beyond the measurement windows is not
relevant for the docs.

## Should the docs explain the `schema_gib` output field?

Yes, in one line. It is a leftover constant (`0.17`), but every `pgbench_ro`
run still prints it, so the README warns that it is not the dataset size.

## Does the `MemTotal` rounding need a code fix?

No. A server sold with 2 GiB of RAM usually reports a little less in
`MemTotal`, so the harness tunes it as 1 GiB and `shared_buffers` (256 MB) ends
up below the 303 MiB database. This is accepted as is.
