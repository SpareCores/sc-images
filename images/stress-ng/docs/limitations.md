# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### Bogo Ops Units

Bogo ops/s (bogus operations per second) count how many times a stressor
completes its own unit of work per second. Each stressor defines that unit
differently, so results are *not* comparable across stressors: compare a
stressor only with itself, on the same `stress-ng` version. The image pins
`stress-ng` 0.20.01 through the tag of its base image.

### A Single Score

The stressor suite reports one value per stressor and worker count, and does
*not* combine them into a category or overall score. It also does not
attribute the results of the `hypervisor` stressors to a specific cause, such
as the hypervisor or the CPU.

### Application Performance

The stressors are synthetic loops. Results show how fast a server runs each
loop, *not* how fast a given application runs. This benchmark does not
measure disk or network performance.

## Scope Decisions

### Run Length

Each stressor runs once, for 5 s (15 s for `stream`), with no warm-up and no
repetitions. A full run took about 15 minutes at the median in production. A
single short run does not show variance between runs or effects that build up
over time, such as CPU-credit throttling on shared-core server types. The
[`stress-ng-longrun`](../../stress-ng-longrun) image covers sustained load in
an experimental 24-hour run.

### Worker Count

The multi-worker run uses one worker per logical CPU available to the
container, not one per physical core. The script would use the physical core
count only if the `psutil` Python package were installed, and the image does
not install it. On a server with SMT (Simultaneous Multithreading), the
worker count therefore includes both threads of each core, even though the
output field is named `physical_cpus`.

## Design Constraints

### Missing Results

A stressor run that fails, takes 30 s longer than its duration, or prints no
result is recorded as `null`, and the script still exits with code 0. The
following stressors are often `null` in production:

- `ipsec-mb` has been `null` in every successful production run so far.
- `tsc`, `usersyscall`, and `x86syscall` are x86-only. On Arm servers,
  `stress-ng` still lists them, so they run and return `null`.
- `l1cache`, `vm`, `zlib`, and `cpu` were `null` in a few production runs.

Because these stressors are still listed, `meta.stressor_count` counts them.
Check each entry for `null` instead of relying on the count.
