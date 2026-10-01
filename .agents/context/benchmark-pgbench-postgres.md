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

Not at the moment. Production runs do not apply `sysctl` or
other host OS tweaks; privileged mode, host networking,
`seccomp=unconfined`, ulimits, and PostgreSQL process priority are container
settings applied by `sc-inspector` orchestration.

## What is the primary advantage versus other database benchmarks?

Our primary goal was to have a benchmarking methodology that scales across
instance sizes: from small instances (e.g. 1 vCPU and 1 GB of RAM) to large
nodes with hundreds of vCPUs.

Other database benchmarks usually focus on storage, network throughput, a single
database operation, or one production workload -- while we focus on CPU and
memory speed of the instance, as disk and network are usually configured
alongside the instance type.

## What does the fixed `PGBENCH_RO_CPU_SCHEMA_GIB=0.17` represent relative to the setup SQL estimate of 260–320 MB of data plus indexes?

This variable used to be dynamically tuned for the early versions of the
benchmark, depending on the instance memory size, but in the current version
it's static. It's a leftover from a system we used to create the
schema (this was the ingested dataset size, of course the on disk dataset was
different), now unused.

## Are DBaaS client/server runs always placed in the same availability zone and connected over private VPC addresses?

AZ-private VPC: always private, but AZ depends on the vendor. Strictly same AZ
for AWS, others might be same region. The difference it makes is less important
since our benchmark is not latency-bound.

## Does the `resource-tracker` runtime preserve the benchmark's JSON object on stdout in production?

We run all benchmarks wrapped into resource-tracker. It sends the metrics to our
sentinel API and we save the JSON output to that S3 bucket simultaneously. We
collect these data for our own purposes and don't (currently) publish them. We
have future plans on using these metrics on sparecores.com.
