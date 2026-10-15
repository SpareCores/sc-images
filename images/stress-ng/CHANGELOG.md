# Changelog

## All Stressors Run on All CPUs

Every stressor of the stressor suite now runs with one worker and then with one
worker per CPU. Before, only a subset of 23 stressors also ran with all CPUs:
10 compute stressors (`cpu`, `matrix`, `cache`, `stream`, `memcpy`, `vecmath`,
`zlib`, `hash`, `crypt`, `malloc`) and 13 hypervisor stressors. The subset
lists are still defined in `run_stressng_benchmarks.py`, but no longer used.

## Stressor Suite and `stress-ng` 0.20.01

The image moved from `stress-ng` 0.17.08 to 0.20.01, and added
`run_stressng_benchmarks.py`, which runs 85 curated stressors in seven
categories and prints bogo ops/s as JSON.

## CPU Scaling Loop with `div16`

The CPU scaling loop runs the `cpu` stressor with the `div16` method. After
running every `--cpu-method` on selected x86_64 and Arm servers, with and
without SMT, `div16` was the method whose scaling matched expectations. Other
methods showed either lower or higher scaling than expected:

- On a 2 vCPU server with SMT, `div16` gave about 1x the single-worker
  throughput with two workers; other methods gave up to 1.8x.
- On SMT servers with many cores, other methods gave either very low scaling
  (about 50x the single-worker throughput on a server with 64 cores) or very high
  scaling (about 130x on a server with 64 cores and 128 threads).
