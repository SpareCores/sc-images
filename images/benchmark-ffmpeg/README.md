# benchmark-ffmpeg

This benchmark measures how much audio and video a server can transcode with
FFmpeg when many jobs are run at once. It starts independent, single-threaded
`ffmpeg` processes from a shared start signal: audio encoding runs on the CPU,
and H.264 and H.265 video encoding and decoding run on the CPU and, when
available, on NVIDIA GPUs.

This benchmark reports the aggregate throughput of each scenario at several
worker counts: frames per second for video, and audio seconds processed per
second for audio. It emits every measured repetition as raw data, without a
headline score.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for more than 5,000 cloud server types.
Media businesses often run video and audio transcoding and similar tasks in the
cloud, because their compute needs change over time. The Navigator previously
covered this kind of work only through proxies from the Geekbench suite: photo
editing, ray tracing, and document rendering.

This benchmark measures transcoding directly. It runs many single-threaded jobs
in parallel and times the whole group, so the result shows how much media a
server type processes at once, rather than how fast a single job finishes.

## Limitations

These results compare transcoding throughput across server types. Note the
following limitations:

- **[Output quality](./docs/limitations.md#output-quality):** This benchmark
  measures speed, not output quality, and computes no quality metric. GPU
  encode settings are calibrated to approximately match the CPU encode quality,
  but the match is not exact. See also
  [CPU and GPU Encode Quality](./docs/limitations.md#cpu-and-gpu-encode-quality).
- **[Storage](./docs/limitations.md#storage-and-output):** Encoded output is
  discarded through FFmpeg's null muxer, so file, disk, and object-storage
  writes are not measured.
- **[Fixed media and settings](./docs/limitations.md#a-specific-media-pipeline):**
  Every job uses the same short video clip or music sample with fixed codec
  settings. Other resolutions, settings, and codecs are not covered, for
  example AV1 (AOMedia Video 1).
- **[GPUs](./docs/limitations.md#gpu-coverage):** Only NVIDIA GPUs are used, for
  video only; audio always runs on the CPU. A GPU scenario that cannot start on
  a given GPU is skipped.

## Usage

Run the image via Docker. With no options, it runs the CPU and audio scenarios
and skips the GPU scenarios:

```bash
docker run --rm ghcr.io/sparecores/benchmark-ffmpeg:main > results.json
```

The image contains its test media, so it needs no network access at run time.

To include the NVIDIA GPU scenarios, pass the GPUs to the container. This needs
the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
on the host:

```bash
docker run --rm --gpus all ghcr.io/sparecores/benchmark-ffmpeg:main > results.json
```

See [GPU Run](./docs/usage.md#gpu-run) for the GPU requirements.

Production runs the container privileged and with host networking; a plain
`docker run` needs neither. See
[Production Setup](./docs/usage.md#production-setup) before you compare your
results with production runs.

A production run on an AWS (Amazon Web Services)
`m9g.16xlarge` took about 10 minutes. Run time grows with each worker count
measured, and with GPU scenarios. See
[Run Duration](./docs/usage.md#run-duration) for the steps and timeouts.

### Settings

The following settings are the most common; see the
[full environment-variable reference](./docs/usage.md#environment-variables)
for all options and defaults:

- `FFMPEG_BENCH_TARGET_SECONDS` sets the wall time of each measured repetition
  (default 5 s).
- `FFMPEG_BENCH_REPETITIONS` and `FFMPEG_BENCH_MAX_REPETITIONS` set how many
  repetitions run at each worker count (default three, up to five).
- `FFMPEG_BENCH_WORKERS` replaces the automatic worker counts with an explicit,
  comma-separated list.
- `FFMPEG_BENCH_MAX_WORKERS` caps the number of parallel workers.
- `FFMPEG_BENCH_TIMEOUT_SECONDS` sets the deadline for the whole run (default
  7,200 s).

### Results

The process prints one compact JSON line to stdout when the run ends; logs go to
stderr. The main fields of the JSON line are the following:

- `scenarios[]` has one entry per scenario, with its `name`, `backend` (`cpu`
  or `gpu`), `operation` (`encode` or `decode`), and `codec`. A scenario that
  could not run has `skipped: true` and a `skip_reason`.
- Each `scaling_steps[]` entry is one worker count: `workers` parallel jobs,
  each processing `media_duration_sec` seconds of media.
- Each `repetitions[]` entry is one measurement at that worker count. Video
  repetitions report `aggregate_fps` (frames per second across all workers) and
  `processed_frames`; audio repetitions report `audio_seconds_per_sec` and
  `processed_audio_seconds`. `wall_time_sec` is the time from the shared start
  to the last worker's exit.
- `successful_workers`, `failed_workers`, and `timed_out_workers` describe each
  repetition. A failed worker contributes no media, but its time still counts.
- There is no score, peak, or median: aggregate the repetitions yourself, for
  example by taking the median rate at each worker count.
- `host`, `ffmpeg`, `sources`, and `methodology` record the server, the FFmpeg
  build, the test media, and the measurement settings.

If `benchmark.py` itself fails, the process exits with code 1 and prints a JSON
object with an `error` field; if `ffmpeg` is not on the `PATH`, it exits with
code 1 and prints no JSON. Exit code 0 does not mean that every scenario ran:
check `skipped`, `skip_reason`, and `failed_workers` in the output.

## Workloads

All scenarios run by default, in the following groups:

- [Audio encoding](./docs/workloads.md#audio-encoding): Vorbis, Opus, AAC
  (Advanced Audio Coding), MP3,
  and FLAC (Free Lossless Audio Codec) encodes of one
  music sample, on the CPU.
- [CPU video](./docs/workloads.md#cpu-video): H.264 and H.265 encoding and
  decoding of one film clip, with one codec thread per job.
- [NVIDIA video](./docs/workloads.md#nvidia-video): the same H.264 and H.265
  encoding and decoding on NVIDIA GPUs, with NVENC
  (NVIDIA Video Encoder)
  and NVDEC
  (NVIDIA Video Decoder).

See [How Measurement Works](./docs/workloads.md#how-measurement-works) for the
worker counts, timing, and repetitions.

## Design History

We built this benchmark around aggregate capacity. Instead of adding up the
speed that each `ffmpeg` process reports for itself, the harness
([`benchmark.py`](./benchmark.py)) times the whole group with one clock, from a
shared start to the last worker's exit, so process startup and teardown count
too. Worker counts follow physical cores rather than logical CPUs, because two
single-threaded workers on the hyperthreads of one core compete for it and can
lower throughput, especially for decoding.

GPU encodes were later calibrated against the CPU encodes: NVENC has no direct
equivalent of the CPU encoders' quality setting, so we compared several NVENC
quality levels on the test clip and picked the closest match. In version 3.3.0,
the image also started requesting the NVIDIA driver's video libraries itself,
which many earlier GPU runs could not load.

The remaining limits are that this benchmark checks no output quality at run
time, that CPU and GPU encode quality match only approximately, and that GPU
results depend on what each GPU and driver support. See the
[changelog](./CHANGELOG.md) for the calibration data and the measurement-length
study.

## References

See the full list of [references](./docs/references.md) used in this
documentation. To change, rebuild, or test this image, see the
[contributor guide](./docs/contributors.md).
