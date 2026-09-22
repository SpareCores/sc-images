Supplementary information for benchmark-pgebench-postgres

# 1
**Limitations / Deliberately out of scope:**
- **Distribution is still uniform, not Zipfian.** The product catalog has over 20k products, but the customer/order/order-item generation is still `g % k` modular arithmetic, not a realistic power-law. A real "few whales, many one-off customers" shape would be a bigger, separate change to the data generator.
- **`jit` and `max_parallel_workers_per_gather` stay off**, matching the original design's rationale: this benchmark measures raw engine/CPU behavior, *not* LLVM JIT jitter or Gather scalability. Those are treated as a separate testing axis.
- **A single monolithic statement** (one `SELECT` with 8 CTEs, one `UNION ALL`) is deliberate: it keeps one `pgbench` transaction equal to one network round trip, which makes the cached-RO redesign resilient to `netem`-simulated RTT (see [[Design History]] for details). The tradeoff is that per-block planner GUCs (e.g. forcing Merge Join specifically) aren't possible without affecting every block.
- **Pre-calibrated weights.** Weights are calibrated on one local Docker Postgres 18 instance. Re-run `profile_v2_breakdown.sql` after any schema/query change, or on significantly different hardware, to confirm no block has drifted back into dominance.
# 2
This is a necessary constraint, evidenced by our lab measurements.

The default lightweight `pgbench` read-only workload (`-S`, one primary-key `SELECT` per transaction) lost ~98% of its single-connection throughput when we injected just +5 ms of one-way delay, and even same-zone random latency glitches visibly distorted results. It measured the network, not the server.

The CPU-heavy, cached, read-only script stayed within ±0.3% at high concurrency under the same injected delay. It also ranks CPUs honestly: two same-size 32-vCPU servers of different CPU generations tied under `-S` at high concurrency, while the heavy script separates them by ~1.25× at a single connection.
# 3
That alone proved insufficient. Occasional latency glitches still distorted lightweight-workload results even same-zone, which is why the workload itself must be RTT-tolerant.