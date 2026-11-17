# stress-ng Long Run

This benchmark measures how the sustained CPU throughput of a server type
changes over 24 hours. Once a minute, it runs the `cpu` stressor of
`stress-ng` with the stress-ng div16 method and one worker per CPU. The load
lasts 5 seconds per minute in the first hour and rises to 55 seconds per minute
from about the 11th hour on.

For each one-minute sample, this benchmark reports stress-ng bogo ops/s
(bogus operations per second) measured over wall-clock (real) time, along with
the sample's end time and load time. A full run has 1,440 samples.

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
  `stress-ng` 0.17.08, while the `stress-ng` image uses 0.20.01, and bogo ops/s
  compare only within one `stress-ng` version.
- **[CPU Only](#workload):** Every sample runs the same `cpu` stressor with
  the div16 method, so this benchmark measures CPU throughput, not
  [disk](../benchmark-storage/README.md) or network performance.
- **[No Cause Attribution](#workload):** Within a run, sample load times and
  time of day both vary, and no recorded field shows the cause of a change,
  such as a CPU credit balance. The samples show throughput drops, but this
  benchmark does not attribute them to a specific cause, such as CPU credits
  or a neighboring server.
- **[Eight Server Types](#production-setup):** Production runs this benchmark
  on only eight server types, and its completed runs are from 2024.
- **[Loop Not in the Image](#usage):** The image contains `stress-ng` and
  `resource-tracker` only. The loop of 24 hours is a command that the caller
  passes, and production, not the image, collects the samples.

## Usage

The image has no default command: its [entrypoint](./Dockerfile) runs `sh`
under `resource-tracker`, so pass the command to run with `-c`. To run one
sample with 5 seconds of load and one worker per CPU and save its stress-ng
YAML report from stderr:

```bash
docker run --rm ghcr.io/sparecores/stress-ng-longrun:main \
  -c 'stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t 5 -Y /dev/stderr' 2> sample.yaml
```

To run the full 24-hour loop as production does and save stdout and stderr:

```bash
docker run --rm --privileged --network host ghcr.io/sparecores/stress-ng-longrun:main \
  -c "nice -n -20 sh -c 'for i in \$(seq 1 1440); do SPM=\$((\$((\$i / 60 + 1)) * 5)); SPM=\$(( \$SPM > 55 ? 55 : \$SPM )); stress-ng --metrics --cpu \$(nproc) --cpu-method div16 -t \$SPM -Y /dev/stderr; sleep \$((60 - \$(date +%-S) )); done'" > longrun.log 2> longrun.yaml
```

In `bash` and other `sh`-compatible shells, the backslashes keep the host shell
from expanding the `$` signs; the shell in the container evaluates them.
The `nice -n -20` command needs `--privileged`. Without `--privileged`, remove
`nice -n -20` from the command, and the loop runs at normal priority. The loop
runs no network workload, so it does not need `--network host`, which
production sets with its other Docker options.

### Production Setup

The Navigator's
[Inspector](https://github.com/SpareCores/sc-inspector) runs the loop above as
its `stressnglongrun` task, defined in
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py).
It uses `--privileged` and `--network host` and runs the loop under
`nice -n -20`. The Inspector runs no other benchmark in parallel with this
one and stops a run that lasts longer than 26 hours. Full production runs took
about 24 hours. Production retries failed runs, so a server type's latest
result can be a failed or partial run. For example, the latest `cax11`
attempts (2026-09-29 and 2026-10-23) ended early.

Production runs this benchmark on only the following server types:

- AWS (Amazon Web Services): `t4g.medium`, `c7g.large`
- GCP (Google Cloud Platform): `e2-medium`, `c2d-highcpu-2`
- Hetzner Cloud: `cx21`, `cx22`, `cax11`, `ccx13`

The completed production runs are from 2024. The image itself does not limit
the server types it runs on.

### Results

Each sample prints a stress-ng metrc table to stdout, along with other `info`
lines. The bogo ops/s (real time) column holds the headline result, as in the
first sample of a production run on `c7g.large`:

```text
stress-ng: metrc: [9] stressor       bogo ops real time  usr time  sys time   bogo ops/s     bogo ops/s CPU used per       RSS Max
stress-ng: metrc: [9]                           (secs)    (secs)    (secs)   (real time) (usr+sys time) instance (%)          (KB)
stress-ng: metrc: [9] cpu               15177      5.00      9.91      0.00      3035.31        1531.58        99.09          1192
```

Each sample also writes one stress-ng YAML report to stderr. The report's
`system-info` section describes the run and system, including the
`stress-ng` version and the time at the end of the sample. Its `metrics`
section holds the following main fields of the `cpu` stressor:

- `bogo-ops-per-second-real-time`: the headline result: the bogo ops/s over
  wall-clock (real) time, summed across all workers
- `wall-clock-time`: the load time of the sample, the duration of the
  stress-ng run in seconds
- `bogo-ops-per-second-usr-sys-time`: the bogo ops/s over the user and system
  CPU time of all workers
- `bogo-ops`, `user-time`, and `system-time`: the count and times behind the
  result

The image does not extract a time series from the YAML reports; it leaves
that to the caller. The Navigator's Inspector extracts the time series with its
[`parse.stressnglongrun`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/parse.py)
parser. The parser writes one line per sample to
[`parsed.csv`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/stressnglongrun/parsed.csv),
which has no header. Each line holds the sample's end time, `wall-clock-time`,
and `bogo-ops-per-second-real-time`. For example, the first line of a
production run on `c7g.large` is:

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

The loop of the `stressnglongrun` task in
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
runs 1,440 samples, one per minute. Each sample runs the `cpu`
stressor with the div16 method (`--cpu-method div16`) and one worker per
logical CPU available to the container (`nproc`), then waits until the start of
the next minute. The load time for each sample depends on its sample number:

| Samples | Load time |
| ------- | --------- |
| 1-59 | 5s |
| 60-119 | 10s |
| 120-539 | 15s-45s, 5s more every 60 samples |
| 540-599 | 50s |
| 600-1,440 | 55s |

The load times total 62,750 seconds, about 17.4 hours of the 24-hour run.
There is no warm-up: the first sample, with 5 seconds of load, counts like
the others.

The load time and the time of day change together during a run, so its samples
do not show the separate effects of load time and time of day.

## Design History

This image pins `stress-ng` 0.17.08 through the tag of its base image,
`ghcr.io/colinianking/stress-ng`, in its [`Dockerfile`](./Dockerfile). The
`stress-ng` image used 0.17.08 until its update to 0.20.01 (see the
`stress-ng` [changelog](../stress-ng/CHANGELOG.md#stressor-suite-and-stress-ng-02001)).
The completed production runs from 2024 used this version, and we keep it so
that new runs stay comparable with them. We do not plan to update it.

## References

The terms on this page are defined in the following places:

- The [references](../stress-ng/docs/references.md) of the
  [stress-ng manual](../stress-ng/README.md) define the terms of `stress-ng`,
  such as stress-ng bogo ops/s.
- The [references](./docs/references.md) of this image define the
  stress-ng YAML report and stress-ng bogo ops/s (real time).
- The [glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs
  defines common terms, such as CPU-credit throttling and shared-core.
