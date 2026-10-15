# stress-ng

This benchmark measures the throughput of CPU, memory, and kernel-heavy
workloads with `stress-ng`. It runs 85 curated stressors in seven categories,
each for 5 s (15 s for `stream`), first with one worker and then with one
worker per available CPU. A second mode runs the `cpu` stressor at increasing
worker counts to trace how throughput scales across cores.

This benchmark reports bogo ops/s (bogus operations per second), the
`stress-ng` throughput metric, for each stressor and worker count, as one JSON
document. The scaling mode prints one `<workers>,<bogo ops/s>` line per worker
count.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for over 5,000 cloud server types, such as
raw CPU speed, memory bandwidth, and compression throughput. This benchmark
covers a wide range of CPU and memory work in one run, from integer and
floating-point math to sorting, hashing, and memory bandwidth.

It also includes a `hypervisor` category: stressors such as context switches,
page faults, system calls, and timers, whose throughput on a VM is dominated by
hypervisor overhead. Comparing them across providers shows differences in
virtualization quality.

The Navigator's Inspector uses the scaling mode to determine a server type's
SCore, a relative performance score for comparing the speed of server types.
See [CPU Scaling Loop](./docs/workloads.md#cpu-scaling-loop).

The [`stress-ng-longrun`](../stress-ng-longrun) image pins an older
`stress-ng` version for an experimental 24-hour run on a few server types. It
shows sustained CPU throughput over time, throttling on shared-core server
types, and noisy-neighbor variance by time of day.

## Limitations

These results are synthetic `stress-ng` throughput figures, not application
performance. The following are the main limitations:

- **Not comparable across stressors:** Each stressor counts its own kind of
  operation, so compare bogo ops/s only for the same stressor and the same
  `stress-ng` version. See
  [Bogo Ops Units](./docs/limitations.md#bogo-ops-units).
- **No single score:** The stressor suite reports one value per stressor and
  worker count, and does not combine them. See
  [A Single Score](./docs/limitations.md#a-single-score).
- **Short, single runs:** Each stressor runs once for 5 s (15 s for `stream`),
  with no warm-up and no repetitions. See
  [Run Length](./docs/limitations.md#run-length).
- **Logical CPUs, not physical cores:** The multi-worker run uses one worker
  per logical CPU available to the container, even though the output field is
  named `physical_cpus`. See
  [Worker Count](./docs/limitations.md#worker-count).
- **Missing results are `null`:** A stressor that fails, times out, or is not
  supported on the CPU architecture is recorded as `null`, and the run still
  succeeds. See [Missing Results](./docs/limitations.md#missing-results).

## Usage

The image has no default command. Append the stressor suite's command after
the image name:

```bash
docker run --rm ghcr.io/sparecores/stress-ng:main \
  -c 'python3 /usr/local/bin/run_stressng_benchmarks.py' > stress-ng.json
```

Production also runs the command under `nice -n -20`, which needs
`--privileged`. See [Production Setup](./docs/usage.md#production-setup) for
the production commands, including the CPU scaling loop.

The stressor suite took 7 to 66 minutes in production, 15 minutes at the
median. See [Run Duration](./docs/usage.md#run-duration).

### Settings

The following settings shorten a run, for example for a quick check; see the
[full environment-variable reference](./docs/usage.md#environment-variables)
for details:

- `STRESSNG_QUICK=1` runs each stressor for 2 s instead of 5 s (`stream`
  still runs for 15 s).
- `STRESSNG_LIMIT=<N>` runs only the first N stressors in alphabetical order.

### Results

The stressor suite prints one JSON document to stdout when the run ends.
Interpret its main fields as follows:

- `stressors.<name>` holds one entry per worker count: `"1"`, and the CPU
  count for the multi-worker run. Each entry is
  `{"bogo_ops_per_sec": <value>}`, the total across all workers, or `null` if
  the run failed.
- `categories` lists the stressors in each of the seven categories.
- `meta.physical_cpus` is the worker count of the multi-worker run, and
  `meta.stressor_count` the number of stressors run.

See [Output](./docs/usage.md#output) for every field.

## Workloads

This image runs the following workloads:

- [Stressor suite](./docs/workloads.md#stressor-suite):
  [`run_stressng_benchmarks.py`](./run_stressng_benchmarks.py) runs 85
  stressors in seven categories, each with one worker and then one worker per
  CPU.
- [CPU scaling loop](./docs/workloads.md#cpu-scaling-loop): the `cpu` stressor
  with the `div16` method runs for 10 s at each worker count from 1 up to the
  CPU count, with [`count.awk`](./count.awk) picking the counts.

For more information on stressor categories and run order, see
[Workloads](./docs/workloads.md).

## Design History

The CPU scaling loop uses the `div16` method of the `cpu` stressor. The Spare
Cores team ran every `--cpu-method` on selected x86_64 and Arm servers, with
and without SMT (Simultaneous Multithreading). Other methods scaled more or
less than expected: on a 2 vCPU server with SMT, some reached up to 1.8 times
the single-worker throughput with two workers, while `div16` stayed at about
the single-worker throughput.

The stressor suite came with an upgrade from `stress-ng` 0.17.08 to 0.20.01.
At first, only a 23-stressor subset also ran with all CPUs; now every stressor
does.

See the [changelog](./CHANGELOG.md) for the changes to this benchmark.

## References

See the full list of [references](./docs/references.md) used in this
documentation.
