# Limitations

## Disclaimer: what this benchmark does *NOT* deliver

- **A Production workload:**
  - Transactions are synthetic proxies; deliberately balanced mixes of PostgreSQL subsystems. See [Workloads](./workloads.md) for details.
  - The benchmark does *not* predict the throughput of any specific application.
  - It instead gives a general sense of the relative RDBMS performance expected of a server type.
- **Disk I/O speed:**
  - Block storage is provisioned independently of the instance type in most clouds, so it is not a property of the server being ranked. The scores do *not* measure storage performance. See [Deliberate Exclusions](#exclusions) for details.
- **Network performance:**
  - The design actively minimizes Round-Trip Time (RTT) sensitivity, so the scores say nothing about a server's network throughput or latency (we publish separate benchmarks for this purpose).
- **Other limitations:**
  - Data distribution in the database test tables is uniform rather than [Zipfian](https://en.wikipedia.org/wiki/Zipf%27s_law). See [Workloads](./workloads.md) for details.
  - PostgreSQL's [Just-in-Time Compilation](https://www.postgresql.org/docs/current/jit.html) (JIT) and [Parallel Query](https://www.postgresql.org/docs/current/parallel-query.html) features are disabled to measure raw engine and CPU behavior.
  - The dataset is [small by design](#deliberately-out-of-scope), so large instances are exercised through concurrency rather than data volume.

## Exclusions

Most published database benchmarks compare *database engines*, engine versions, or config tuning on fixed hardware. Our approach is the inverse: keep the engine constant and vary the hardware across thousands of server types. This inversion is the design's primary drive.

- ***External Limitations:***
  - **Minor Engine Versions:** Many DBaaS providers we benchmark do not allow pinning the minor engine version (minor upgrades are applied automatically), so only the *major* PostgreSQL version is fixed across IaaS and DBaaS runs.
- ***Deliberate Exclusions:***
  - **Disk Speed:** Database throughput usually hinges on the underlying disk (Input/Output operations per second (IOPS) first, bandwidth second). In the cloud, that disk is almost always network-attached block storage, provisioned independently of the server type and entirely up to the user. As such, it says little about the server itself. Deploying volumes with high-enough IOPS to never bottleneck across a ~5,000-server fleet would also be prohibitively expensive. Because of this, we exclude disk speed from the measurement and score the database engine's CPU and memory performance.
  - **RTT:** With a remote client, both bandwidth and especially latency between client and server can dominate chatty workloads. To avoid this, we minimize RTT via placement, then design the workload so the remaining RTT is a rounding error.
- ***Design Considerations:***
  - **Massive workload size range:** From the smallest cloud server instances with a singe vCPU and mere megabytes of memory to industrial-scale machines with thousands of vCPUs and terabytes of memory, the same workload must produce meaningful, comparable numbers.
    - Available warehouse and scale-factor sizing schemes can't quite fulfill this purpose. Because of this, we chose a fixed-size workload that could run on all servers of the fleet, with larger servers being taxed by concurrency rather than workload size.
  - **Engine config:** DBaaS forbids config control. The vendor tunes the managed engine, so the harness cannot assume superuser access or [Grand Unified Configuration](https://www.postgresql.org/docs/current/config-setting.html) (GUC) control.

### Deliberately out of scope

- **Distribution is still uniform, not Zipfian.** The product catalog has over 20k products, but the customer/order/order-item generation is still `g % k` modular arithmetic, not a realistic power-law. A real "few whales, many one-off customers" shape would be a bigger, separate change to the data generator.
- **`jit` and `max_parallel_workers_per_gather` stay off**, matching the original design's rationale: this benchmark measures raw engine/CPU behavior, *not* LLVM JIT jitter or Gather scalability. Those are treated as a separate testing axis.
- **A single monolithic statement** (one `SELECT` with 8 CTEs, one `UNION ALL`) is deliberate: it keeps one `pgbench` transaction equal to one network round trip, which makes the cached-RO redesign resilient to `netem`-simulated RTT (see [[Design History]] for details). The tradeoff is that per-block planner GUCs (e.g. forcing Merge Join specifically) aren't possible without affecting every block.
- **Pre-calibrated weights.** Weights are calibrated on one local Docker Postgres 18 instance. Re-run `profile_v2_breakdown.sql` after any schema/query change, or on significantly different hardware, to confirm no block has drifted back into dominance.

## Design Constraints

With the help of the [benchANT](https://benchant.com) team, over many iterations with different tools and configs, we identified the following core principles:

1. **Memory-fit, small dataset:** ~260–320 MB, stored comfortably in `shared_buffers`, even on the smallest instances. After warmup, the disk is not read again.
2. **Read-only workload:** No [Write-Ahead Logging](https://www.postgresql.org/docs/current/wal-intro.html) (WAL), no checkpoints, and no disk-write paths.
3. **CPU-heavy transactions:** 100+ ms of server work per transaction per connection. This minimizes network round-trip time to ~0.2–4% of the total service time and prevents it from dominating. This is [a necessary constraint](#a-necessary-constraint), evidenced by our lab measurements.

### A Necessary Constraint

The default lightweight `pgbench` read-only workload (`-S`, one primary-key `SELECT` per transaction) lost ~98% of its single-connection throughput when we injected just +5 ms of one-way delay, and even same-zone random latency glitches visibly distorted results. It measured the network, not the server.

The CPU-heavy, cached, read-only script stayed within ±0.3% at high concurrency under the same injected delay. It also ranks CPUs honestly: two same-size 32-vCPU servers of different CPU generations tied under `-S` at high concurrency, while the heavy script separates them by ~1.25× at a single connection.

## Operational Details

The benchmark can be used for the following production runs:

- **IaaS server tuning**: Postgres GUCs are generated per host by [pgtune](https://pgtune.leopard.in.ua/).
  - Form defaults:
    - web application
    - SSD
    - Postgres 18 (with the host's RAM and CPU count)
    - `sc-inspector` orchestration
    - `synchronous_commit` is set by the durable/async task variant
    - `max_connections` is raised to cover the concurrency profile
- **No DBaaS tuning**: The vendor-managed config is left untouched by design, as the tuning of the managed service is part of what is being measured.
- **No OS-level tuning**:
  - No `sysctl` or other host tweaks.
  - Container-level only. The server runs privileged with the following:
    - host networking
    - `seccomp=unconfined`
    - high `nofile` and unlimited `memlock` ulimits (unlocking huge pages and `io_uring`)
    - postgres process at `nice -n -20`
- **Topology**: Client and server VMs are deployed in the same availability zone of the same region, talking over private VPC addresses to minimize RTT.

  **Note**: This alone is insufficient. Occasional latency glitches still distort lightweight-workload results even when deployed in the same zone. Because of this the workload itself must be RTT-tolerant.
- **Timing**: See [Key Environment Variables](./usage.md#key-environment-variables) for details.
  - 120 s warmup (once)
  - 60 s settle between concurrency rungs
  - 300 s measurement per rung
