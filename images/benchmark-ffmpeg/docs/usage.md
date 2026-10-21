# Usage

## CPU Run

Run the image via Docker without options to measure the audio and CPU video
scenarios:

```bash
docker run --rm ghcr.io/sparecores/benchmark-ffmpeg:main > results.json
```

The four GPU scenarios are skipped with `skip_reason` `no_cuda_gpu_detected`.
No environment variables are required, and the image needs no network access at
run time: the test media is downloaded and checksum-verified when the image is
built.

`--version` prints the versions of the harness
([`benchmark.py`](../benchmark.py)) and FFmpeg, and exits:

```bash
docker run --rm ghcr.io/sparecores/benchmark-ffmpeg:main --version
```

## GPU Run

Pass the GPUs to the container to add the NVIDIA video scenarios:

```bash
docker run --rm --gpus all ghcr.io/sparecores/benchmark-ffmpeg:main > results.json
```

Before you run, check the following prerequisites:

- The host has an NVIDIA GPU, the NVIDIA driver, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
  The image ships no CUDA
  (Compute Unified Device Architecture)
  toolkit or driver libraries;
  NVENC (NVIDIA Video Encoder)
  and NVDEC (NVIDIA Video Decoder)
  are loaded from the host driver at run time.
- `nvidia-smi` works inside the container. The harness uses it to find the
  GPUs; without it, the GPU scenarios are skipped.

The image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, so the
NVIDIA Container Toolkit also mounts the driver's video libraries; you do not
need to request them yourself. Without the `video` capability, the GPU
scenarios fail to load `libnvcuvid.so.1` and are skipped.

Each GPU scenario first runs a one-worker probe. If the GPU cannot open an
NVENC or NVDEC session for it, the scenario is skipped with the error text as
its `skip_reason`.

## Container Resource Limits

This benchmark sizes its worker counts to the resources the container is
allowed to use, read from the cgroup v2
(control groups v2) files
and the CPU affinity. The following
[Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/)
all lower the worker counts:

- `--cpus` (cgroup `cpu.max`)
- `--cpuset-cpus` (CPU affinity)
- `--memory` (`memory.max`): each audio worker is budgeted 64 MiB and each
  video worker 320 MiB of available memory.
- `--pids-limit` (`pids.max`): each worker is budgeted
  `FFMPEG_BENCH_PID_TASKS_PER_WORKER` tasks (default four), because the PID
  (Process Identifier) limit counts threads, not only processes.

The output records the limits it found in `host`, including `cgroup_cpu_max`,
`cgroup_cpuset_effective`, and the `cgroup_cpu_stat_before` and
`cgroup_cpu_stat_after` snapshots. The `_gb` memory fields are in GiB.

The harness copies the test media into a work directory on `/dev/shm` when it
has at least 512 MiB free, and into the default temporary directory
otherwise. Docker's default `/dev/shm` is 64 MiB, so a plain `docker run`
uses `/tmp`.

## Worker Counts

Each scenario is measured at several worker counts (parallel `ffmpeg`
processes), set in [`benchmark.py`](../benchmark.py) as follows:

- **CPU scenarios:** 1, P/2 (rounded down), and P, where P is the number of
  physical cores the container can use. Logical CPUs that are SMT (Simultaneous
  Multithreading) siblings of the same core count once. If throughput at P is
  at least 1.08 times the throughput at P/2, one more point at 2·P is added,
  capped by the logical CPU count and the limits above.
- **GPU scenarios:** 1 and G, where G is the number of GPUs. The search then
  keeps doubling the worker count (2·G, 4·G, ...) as long as throughput does
  not drop, up to eight sessions per GPU by default. Any drop stops the search;
  the 1.08 ratio does not apply to GPUs.

A worker count with a failed worker stops the search for that scenario. On
servers with more than one NUMA
(Non-Uniform Memory Access) node,
workers are spread across the nodes in proportion to their CPUs and bound to
them with [`numactl`](https://man7.org/linux/man-pages/man8/numactl.8.html), if
binding works in the container. Otherwise, normal Linux scheduling applies.

For example, a server with 48 physical cores and 96 logical CPUs, with no
container limits, measures the CPU scenarios at 1, 24, and 48 workers, and at
96 workers if 48 workers beat 24 by at least 8%.

The following settings change the worker counts:

- `FFMPEG_BENCH_WORKERS` replaces the automatic counts with an explicit,
  comma-separated list, capped at the maximum worker count, and disables the
  search.
- `FFMPEG_BENCH_OVERSUBSCRIPTION` above 1 adds one point at that multiple of P
  (or of G, for GPU scenarios).
- `FFMPEG_BENCH_MAX_WORKERS` caps the worker count.

## Run Duration

Each scenario runs the following steps:

1. A one-worker probe with 1 s of video or 5 s of audio checks that
   the scenario can run and estimates its speed. A scenario whose probe fails is
   skipped.
2. At each worker count, a pilot run sizes the media for each worker so that
   one repetition takes about `FFMPEG_BENCH_TARGET_SECONDS` (default 5 s)
   of wall time. Faster servers therefore process more media in the same time.
   A successful pilot is not reported; a pilot that still fails after up to four
   attempts is reported as a failed repetition and ends the scenario.
3. By default, three repetitions are measured. While the CV (Coefficient of
   Variation) of their rates is above 10%, more are added, up to five. A
   repetition with a failed worker ends the worker count.

Each repetition, including pilots, times out after
`FFMPEG_BENCH_REPETITION_TIMEOUT_SECONDS` (default 30 s), and workers that
run past it count as failed. The whole run stops after
`FFMPEG_BENCH_TIMEOUT_SECONDS` (default 7,200 s); scenarios that have not
started by then are reported as skipped with `overall_timeout`.

Total run time therefore depends on the number of scenarios that run and the
worker counts each one measures. A production run on a 64 vCPU AWS
(Amazon Web Services) `m9g.16xlarge` without GPUs
measured every CPU and audio scenario at 1, 32, and 64 workers and took about
10 minutes
([recorded output](https://github.com/SpareCores/sc-inspector-data/tree/main/data/aws/m9g.16xlarge/0/ffmpeg)).
Across 372 production runs of harness version 3.2.1, run time ranged from about
3 minutes on servers with 2 vCPUs to about 17 minutes on GPU
servers, with a median of about 9.5 minutes.

### Repetition Length

Each repetition targets 5 s of wall time by default. Because the pilot
scales the media to that target, larger servers process more media in the same
time instead of running longer. The default comes from a study that compared
2, 5, and 10 s targets on one developer workstation, a 24-core Intel Core
i9-14900K desktop CPU without an NVIDIA GPU, so GPU scenarios were not part of
it. Only the study's summary was kept; see
[Why 5 Seconds](../CHANGELOG.md#why-5-seconds).

### Shorter Test Runs

For a quick check that the image works, reduce the worker counts, repetitions,
and media lengths:

```bash
docker run --rm \
  -e FFMPEG_BENCH_WORKERS=1,2 \
  -e FFMPEG_BENCH_REPETITIONS=1 \
  -e FFMPEG_BENCH_TARGET_SECONDS=1 \
  -e FFMPEG_BENCH_MIN_MEDIA_SECONDS=1 \
  -e FFMPEG_BENCH_MAX_MEDIA_SECONDS=5 \
  -e FFMPEG_BENCH_AUDIO_CALIBRATION_SECONDS=1 \
  ghcr.io/sparecores/benchmark-ffmpeg:main
```

The results of such a run are not comparable with default runs.

## Production Setup

The Inspector of Spare Cores Navigator
([`sc-inspector`](https://github.com/SpareCores/sc-inspector)) runs this image
on every server type except those on the
[`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
blocklist, with the following settings. A plain `docker run` differs from it as
noted below:

- The container runs privileged and with host networking. This benchmark does
  not need either, and a plain `docker run` uses neither.
- No `--shm-size` is set, so `/dev/shm` keeps Docker's 64 MiB default and
  the test media is copied to `/tmp`, as in a plain run.
- GPUs are requested for the container, equivalent to `--gpus all`. If that
  fails, the run is retried without GPUs.
- The task timeout is 2 hours, the same as the harness default.
- Production skips servers with less than 1 GiB of memory. The image does
  not enforce this minimum.

We investigate failed production runs, manually or semi-automatically, and
then run them again. Until then, a server type's latest result in
`sc-inspector-data` can be a failed or partial run.

## Environment Variables

The following variables configure this benchmark. See
[Results](../README.md#results) for the output fields.

| Variable | Meaning | Default |
| --- | --- | --- |
| `FFMPEG_BENCH_TARGET_SECONDS` | Target wall time per repetition, in seconds. | `5` |
| `FFMPEG_BENCH_MIN_MEDIA_SECONDS` | Minimum media length per worker, in seconds. | `0.5` |
| `FFMPEG_BENCH_MAX_MEDIA_SECONDS` | Maximum media length per worker, in seconds. | `1800` |
| `FFMPEG_BENCH_VIDEO_CALIBRATION_SECONDS` | Media length of the one-worker video probe, in seconds. | `1` |
| `FFMPEG_BENCH_AUDIO_CALIBRATION_SECONDS` | Media length of the one-worker audio probe, in seconds. | `5` |
| `FFMPEG_BENCH_REPETITIONS` | Measured repetitions per worker count (at least 1). | `3` |
| `FFMPEG_BENCH_MAX_REPETITIONS` | Maximum repetitions per worker count when the rates vary. | `5` |
| `FFMPEG_BENCH_CV_THRESHOLD` | Add repetitions while the coefficient of variation of the rates is above this value. | `0.10` |
| `FFMPEG_BENCH_OVERSUBSCRIPTION` | Above 1, adds a worker count at this multiple of the physical cores (CPU) or GPUs (GPU). | `1` |
| `FFMPEG_BENCH_SCALE_CONTINUE_RATIO` | CPU only: throughput gain from P/2 to P workers required to try 2·P workers. | `1.08` |
| `FFMPEG_BENCH_WORKERS` | Comma-separated worker counts; replaces the automatic counts and disables the search. | Automatic |
| `FFMPEG_BENCH_MAX_WORKERS` | Hard cap on the worker count. | Automatic |
| `FFMPEG_BENCH_GPU_ENCODE_SESSIONS_PER_GPU` | Highest NVENC worker count per GPU the search can reach. | `8` |
| `FFMPEG_BENCH_GPU_DECODE_SESSIONS_PER_GPU` | Highest NVDEC worker count per GPU the search can reach. | `8` |
| `FFMPEG_BENCH_PID_TASKS_PER_WORKER` | Tasks budgeted per worker against the container's PID limit (at least 2). | `4` |
| `FFMPEG_BENCH_REPETITION_TIMEOUT_SECONDS` | Timeout for each repetition or pilot, in seconds. | 6 × target, at least `30` |
| `FFMPEG_BENCH_TIMEOUT_SECONDS` | Deadline for the whole run, in seconds. | `7200` |
| `FFMPEG_BENCH_AUDIO_SOURCE` | Path of an alternate FLAC file. It is not checksum-verified. | Bundled sample |
| `FFMPEG_BENCH_VIDEO_SOURCE` | Path of an alternate H.264 MP4 file. It must match `FFMPEG_BENCH_VIDEO_SHA256`. | Bundled clip |
| `FFMPEG_BENCH_VIDEO_HEVC_SOURCE` | Path of an alternate H.265 MP4 file. It must match `FFMPEG_BENCH_VIDEO_HEVC_SHA256`. | Bundled clip |
| `FFMPEG_BENCH_VIDEO_SHA256` | Expected SHA-256 (Secure Hash Algorithm 256) hash of the H.264 file; empty skips the check. | Pinned hash |
| `FFMPEG_BENCH_VIDEO_HEVC_SHA256` | Expected SHA-256 of the H.265 file; empty skips the check. | Pinned hash |
| `FFMPEG_BENCH_CDN_BASE_URL` | Base URL recorded as the test media source in the output; nothing is downloaded. | `https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg` |
| `FFMPEG`, `FFPROBE` | Alternate `ffmpeg` and `ffprobe` binaries. | `ffmpeg` and `ffprobe` on `PATH` |
