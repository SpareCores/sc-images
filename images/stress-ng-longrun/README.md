# stress-ng Long Run

This benchmark measures how the sustained CPU throughput of a server type
changes over 24 hours. Once a minute, it runs the `cpu` stressor of
`stress-ng` with the stress-ng div16 method and one worker per CPU, for 5
seconds per minute in the first hour, rising to 55 seconds per minute from the
11th hour on.

This benchmark reports stress-ng bogo ops/s (bogus operations per second) in
real time for each one-minute sample, 1,440 samples in a full run, along with
the time and duration of each sample.

The [stress-ng manual](../stress-ng/README.md) describes `stress-ng`, its
stressors, and their results, and this page covers only what this image does
differently.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) measures CPU
speed with the [`stress-ng`](../stress-ng/README.md) image, which runs each
stress-ng stressor for 15 seconds or less. A run that short does not show
effects that build up over time. This benchmark loads the CPU once a minute
for 24 hours, so that its results can expose the following effects:

- changes in sustained CPU throughput over time
- CPU-credit throttling on shared-core server types
- noisy-neighbor variance by time of day

The load rises during the run to test whether a higher sustained load triggers
throttling or another penalty. These runs are experimental: a run of 24 hours
costs far more than a standard run of a few minutes, so production runs this
benchmark on only eight server types.

## Limitations

Two limitations of the `stress-ng` image also apply to this benchmark:
[bogo ops/s compare only for the same stressor](../stress-ng/docs/limitations.md#cross-stressor-comparison),
and the
[worker count is one per logical CPU, not per physical core](../stress-ng/docs/limitations.md#worker-count).
This benchmark also has the following limitations:

- **[Older `stress-ng` Version](#design-history):** This image pins
  `stress-ng` 0.17.08, not the 0.20.01 of the `stress-ng` image, and bogo ops/s
  compare only within one `stress-ng` version.
- **[CPU Only](#workload):** Every sample runs the same `cpu` stressor method,
  so this benchmark measures CPU throughput, not disk or network performance.
- **[No Cause Attribution](#workload):** Within one run, both the load per
  minute and the time of day change, and no recorded field shows the cause of
  a change, such as a CPU credit balance. A drop in throughput shows up in the
  samples, but this benchmark does not attribute it to a specific cause, such
  as CPU credits or a neighboring server.
- **[Eight Server Types](#production-setup):** Production runs this benchmark
  on only eight server types, and its completed runs are from 2024. The image
  does not enforce this list.
- **[Loop Not in the Image](#usage):** The image contains `stress-ng` and
  `resource-tracker` only. The loop of 24 hours is a command that the caller
  passes, and production, not the image, collects the samples.

## Usage

The image has no default command: its entrypoint runs `sh` under
`resource-tracker`, so pass the command to run with `-c`. The following command
runs one sample of 5 seconds, with one worker per CPU, and saves its
stress-ng YAML report from stderr:

```bash
docker run --rm ghcr.io/sparecores/stress-ng-longrun:main \
  -c 'stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t 5 -Y /dev/stderr' 2> sample.yaml
```

The following command runs the full loop of 24 hours as production does, and
saves stdout and stderr:

```bash
docker run --rm --privileged --network host ghcr.io/sparecores/stress-ng-longrun:main \
  -c "nice -n -20 sh -c 'for i in \$(seq 1 1440); do SPM=\$((\$((\$i / 60 + 1)) * 5)); SPM=\$(( \$SPM > 55 ? 55 : \$SPM )); stress-ng --metrics --cpu \$(nproc) --cpu-method div16 -t \$SPM -Y /dev/stderr; sleep \$((60 - \$(date +%-S) )); done'" > longrun.log 2> longrun.yaml
```

In `bash` and other `sh`-compatible shells, the backslashes keep the host shell
from expanding the `$` signs, so that the shell in the container evaluates them.
The
`nice -n -20` command needs `--privileged`. Without `--privileged`, remove
`nice -n -20` from the command, and the loop runs at normal priority.

### Production Setup

The Navigator's Inspector runs the loop above with `--privileged` and
`--network host`, and with the loop under `nice -n -20`. It runs no other
benchmark in parallel with this one, and stops it after 26 hours. Full
production runs took about 24 hours.

Production runs this benchmark on only the following server types:

- AWS (Amazon Web Services): `t4g.medium`, `c7g.large`
- GCP (Google Cloud Platform): `e2-medium`, `c2d-highcpu-2`
- Hetzner Cloud: `cx21`, `cx22`, `cax11`, `ccx13`

The completed production runs are from 2024. The image itself does not limit
the server types it runs on.

### Results

Each sample prints its stress-ng metrc line and other `info` lines to stdout,
for example:

```text
stress-ng: metrc: [9] cpu               15177      5.00      9.91      0.00      3035.31        1531.58        99.09          1192
```

Each sample also writes one stress-ng YAML report to stderr. Its `system-info`
section describes the run and the system, such as the `stress-ng` version and
the time at the end of the sample. Its `metrics` section holds the following
main fields of the `cpu` stressor:

- `bogo-ops-per-second-real-time`: the headline result, the bogo ops/s in real
  time, the total across all workers
- `wall-clock-time`: the duration of the sample in seconds
- `bogo-ops`, `bogo-ops-per-second-usr-sys-time`, `user-time`, and
  `system-time`: the counts and times behind the result

The Navigator's Inspector reads the YAML reports and writes one `parsed.csv`
line per sample, with no header: the time at the end of the sample,
`wall-clock-time`, and `bogo-ops-per-second-real-time`. The first line of a
production run on `c7g.large` is the following:

```text
2024-07-09T12:07:32,5.000142,3035.313593
```

See the full
[`parsed.csv`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/stressnglongrun/parsed.csv),
[`stdout`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/stressnglongrun/stdout),
and
[`stderr`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/stressnglongrun/stderr)
of that run.

## Workload

The loop runs 1,440 samples, one per minute. Each sample runs the `cpu`
stressor with the div16 method (`--cpu-method div16`) and one worker per
logical CPU available to the container (`nproc`), then waits until the start of
the next minute. The load of each sample rises with its number as follows:

| Samples | Load per minute |
| ------- | --------------- |
| 1-59 | 5s |
| 60-119 | 10s |
| Each next 60 samples | 5s more |
| 540-599 | 50s |
| 600-1,440 | 55s |

The load totals 62,750 seconds, about 17.4 hours of the 24. There is no
warm-up: the first sample, of 5 seconds, counts like the others.

The load and the time of day change together during a run, so the samples of
one run do not tell their effects apart.

## Design History

This image pins `stress-ng` 0.17.08 through the tag of its base image, the
version that the `stress-ng` image used before it moved to 0.20.01 (see its
[changelog](../stress-ng/CHANGELOG.md)). The completed production runs from
2024 used this version, and we keep it so that new runs stay comparable with
them. We do not plan to update it.

## References

The terms on this page are defined in the following places:

- The [references](../stress-ng/docs/references.md) of the stress-ng manual
  define the terms of `stress-ng`, such as stress-ng bogo ops/s.
- The [references](./docs/references.md) of this image define the
  stress-ng YAML report.
- The [glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs
  defines common terms, such as CPU-credit throttling and shared-core.
