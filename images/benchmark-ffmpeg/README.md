# benchmark-ffmpeg

Aggregate FFmpeg transcoding-capacity benchmark for x86_64 and arm64. It
measures Ogg Vorbis and FLAC audio transcoding alongside the existing
H.264/H.265 CPU and NVIDIA video scenarios.

Published as `ghcr.io/sparecores/benchmark-ffmpeg:main`.

## Usage

```bash
docker run --rm ghcr.io/sparecores/benchmark-ffmpeg:main > results.json
docker run --rm --gpus all ghcr.io/sparecores/benchmark-ffmpeg:main > results.json
```

The image needs no runtime network access. `--version` prints the harness and
FFmpeg versions.

For a quick development run:

```bash
docker run --rm \
  -e FFMPEG_BENCH_WORKERS=1,2 \
  -e FFMPEG_BENCH_REPETITIONS=1 \
  -e FFMPEG_BENCH_TARGET_SECONDS=1 \
  -e FFMPEG_BENCH_MIN_MEDIA_SECONDS=1 \
  -e FFMPEG_BENCH_MAX_MEDIA_SECONDS=5 \
  -e FFMPEG_BENCH_VIDEO_CALIBRATION_SECONDS=1 \
  -e FFMPEG_BENCH_AUDIO_CALIBRATION_SECONDS=1 \
  ghcr.io/sparecores/benchmark-ffmpeg:main
```

## Fixtures

The image downloads pinned audio and video fixtures from
[sc-cdn](https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/) at build
time, verifies SHA-256, and needs no runtime network access. Workers loop the
local files with `-stream_loop -1` and `-t` so fast machines still measure a
long enough interval without storing an expanded copy.

See also [`SOURCE.md`](SOURCE.md) for regeneration and upload steps.

| File | CDN (image build) | SHA-256 |
|---|---|---|
| `source.flac` | https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/source.flac | `4445399abe62c9d7c546711a853fccfab8ab274226d2e80aa0e5ad948589e516` |
| `source.mp4` | https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/source.mp4 | `a49fe8b82c96bafcc344d374facb716f481e3fa9c6753d56f6d1e0ed509a14e7` |

### Audio — original source

| | |
|---|---|
| Work | “Entre dos Aguas” sample, performed by Michael Laucke |
| Format | FLAC, ~85 s, ~8.3 MB |
| License | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| Commons page | https://commons.wikimedia.org/wiki/File:Entre-dos-aguas-laucke-version-sample.flac |
| Direct download | https://commons.wikimedia.org/wiki/Special:Redirect/file/Entre-dos-aguas-laucke-version-sample.flac |

### Video — original source

| | |
|---|---|
| Work | [Tears of Steel](https://mango.blender.org/) (Blender Foundation open movie) |
| Upstream release | [HD 1080p MOV](https://mango.blender.org/download/) (~557 MB, 1920×800, 24 fps, H.264) |
| Mirror used by `prepare-video-fixture.sh` | http://ftp.halifax.rwth-aachen.de/blender/demo/movies/ToS/tears_of_steel_1080p.mov (zip-wrapped MOV) |
| Alternate upstream | https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip |
| Bundled excerpt | 30 s cut at 60–90 s with `-c:v copy` → `source.mp4` (~20 MB) |
| License | [CC BY 3.0](https://mango.blender.org/sharing/) |

To refresh fixtures: `./prepare-video-fixture.sh` (video excerpt from upstream),
copy or download the FLAC, then `./upload-fixtures.sh` (AWS profile `sc`, bucket
`sc-cdn-cae3awai`, prefix `sc-inspector/benchmark-ffmpeg/`).

## Audio profiles

Every audio job decodes the same lossless music source and explicitly produces
stereo audio. The 96–320 kbps and lossless profiles use 44.1 kHz. The 24 kbps
profile uses 16 kHz because `libvorbis` rejects an exact managed 24 kbps stereo
profile at 44.1 or 22.05 kHz. Encoded packets go to FFmpeg's null muxer so the
score measures codec capacity rather than storage.

| Scenario | Encoder | Output profile |
|---|---|---|
| `ogg_vorbis_24k` | `libvorbis` | Ogg Vorbis, 24 kbps, 16 kHz |
| `ogg_vorbis_96k` | `libvorbis` | Ogg Vorbis, 96 kbps |
| `ogg_vorbis_160k` | `libvorbis` | Ogg Vorbis, 160 kbps |
| `ogg_vorbis_320k` | `libvorbis` | Ogg Vorbis, 320 kbps |
| `flac_lossless` | `flac` | FLAC, compression level 5 |

Every audio job uses the pinned CC0 FLAC fixture described under
[Fixtures](#fixtures).

## Video profiles

Every video job loops the pinned Tears of Steel H.264 excerpt (1920×800, 24 fps,
30 seconds) described under [Fixtures](#fixtures).

| Scenario | Backend | Operation | Codec |
|---|---|---|---|
| `cpu_h264_encode` | CPU | encode | `libx264 -crf 18` |
| `cpu_h265_encode` | CPU | encode | `libx265 -crf 18` |
| `cpu_h264_decode` | CPU | decode | H.264 |
| `gpu_h264_encode` | NVIDIA | encode | `h264_nvenc` |
| `gpu_h264_decode` | NVIDIA | decode | `h264_cuvid` |

GPU workers are distributed round-robin over all GPUs reported by
`nvidia-smi`. The search starts at one session per GPU and doubles until
aggregate throughput drops; eight sessions per GPU is only a ceiling. A runtime
probe still decides whether each scenario is actually usable.

Vorbis and FLAC are CPU-only. FFmpeg exposes no NVENC/NVDEC, VAAPI, QSV, AMF,
or other hardware encoder for these audio codecs. GPUs therefore do not change
the audio scores. NVIDIA's documented hardware codecs are H.264, HEVC,
and AV1; see the [NVIDIA FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html).

## Correct aggregate timing

Audio encoding does not scale one stream across many cores: neither
`libvorbis` nor FFmpeg's FLAC encoder advertises frame or slice threading.
The benchmark therefore runs independent FFmpeg processes with one codec
thread each, matching a batch transcoding service.

For each scaling step it:

1. Pre-spawns every worker behind one inherited pipe barrier.
2. Starts `time.monotonic_ns()` and releases every worker with the same EOF.
3. Records each exit and the last successful/failed completion.
4. Reports the original count and its rate: processed frames and aggregate FPS
   for video, or processed audio seconds and audio-seconds-per-second for audio.
5. Repeats the step and emits every raw repetition. Variability is calculated
   internally only to decide whether another repetition is necessary.
6. Enforces the whole-benchmark time budget with a monotonic deadline that is
   passed into every repetition.

It never sums per-process `speed=` values, which would overstate capacity when
workers start or finish at different times. Startup and teardown are included.
FFmpeg documents both its
[benchmark options and machine-readable progress output](https://ffmpeg.org/ffmpeg.html);
the harness uses its own group clock because FFmpeg only knows about one
process.

There are no converted capacity units, peaks, recommendations, summary
statistics, or duplicated realtime factors in the output. Consumers receive
the source counts, wall-clock time, rates, worker outcomes, and finish spread
for every repetition and can aggregate those measurements as needed.

## Scaling on large instances

The worker ladder is built from **physical cores** (SMT siblings collapsed via
`thread_siblings_list`), not `nproc`. On this class of instance that is 48
cores rather than 96 hyperthreads. CPU scenarios measure **1, P/2, and P**. If
aggregate throughput at P is still at least 8% above P/2, one SMT probe at
`min(2P, logical)` is added. Memory-bound work (H.264 decode) typically peaks
at P/2 and skips the HT point; compute-bound encode may keep the extra sample.

GPU scenarios measure **1 and G** (one session per GPU), then **double** (2G,
4G, …) until throughput falls or a session ceiling is hit. That finds the
NVENC/NVDEC sweet spot without assuming 8 sessions per GPU. Failed worker
counts stop the search immediately.

Independent FFmpeg processes are already single-threaded.
`FFMPEG_BENCH_OVERSUBSCRIPTION=2` still forces a 2·P CPU point.
`FFMPEG_BENCH_WORKERS` disables adaptive search and uses an explicit list.

It always attempts the physical-core (CPU) or one-per-GPU (GPU) anchor unless
an earlier worker count cannot run.

Available capacity is the minimum of scheduler affinity and cgroup v2
`cpu.max`; RAM, `pids.max`, and an optional explicit cap limit process count.
The benchmark applies a conservative per-worker PID budget because the cgroup
PID controller counts threads, not just top-level FFmpeg processes. The JSON
records cpuset/quota information and raw `cpu.stat` snapshots.
On usable multi-node systems, `numactl` stages one hot source copy per NUMA
node and binds each node's worker pool and memory proportionally to the CPUs on
that node. Restricted containers fall back to normal Linux scheduling.

A discarded pilot at every worker count sizes media to the requested wall
time. If a pilot times out, the harness shortens the media and retries up to
four times. A failed measured repetition stops that step immediately instead
of repeating the same timeout. Because calibration targets wall time rather
than a fixed amount of media, larger machines process more frames or audio in
the same measurement interval.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `FFMPEG_BENCH_TARGET_SECONDS` | `5` | Approximate measured time per repetition |
| `FFMPEG_BENCH_MIN_MEDIA_SECONDS` | `0.5` | Minimum media encoded per worker |
| `FFMPEG_BENCH_MAX_MEDIA_SECONDS` | `1800` | Maximum media encoded per worker |
| `FFMPEG_BENCH_VIDEO_CALIBRATION_SECONDS` | `1` | Media used by discarded video calibration |
| `FFMPEG_BENCH_AUDIO_CALIBRATION_SECONDS` | `5` | Media used by discarded audio calibration |
| `FFMPEG_BENCH_REPETITIONS` | `3` | Measured repetitions per worker count |
| `FFMPEG_BENCH_MAX_REPETITIONS` | `5` | Adaptive repetition cap |
| `FFMPEG_BENCH_CV_THRESHOLD` | `0.10` | Internal threshold for adding repetitions |
| `FFMPEG_BENCH_OVERSUBSCRIPTION` | `1` | Set to `2` to add a 2·V worker count |
| `FFMPEG_BENCH_PID_TASKS_PER_WORKER` | `4` | Conservative cgroup PID budget per worker |
| `FFMPEG_BENCH_MAX_WORKERS` | automatic | Hard worker cap |
| `FFMPEG_BENCH_GPU_ENCODE_SESSIONS_PER_GPU` | `8` | NVENC search ceiling per GPU |
| `FFMPEG_BENCH_GPU_DECODE_SESSIONS_PER_GPU` | `8` | NVDEC search ceiling per GPU |
| `FFMPEG_BENCH_SCALE_CONTINUE_RATIO` | `1.08` | Min gain vs previous point to keep doubling / try SMT |
| `FFMPEG_BENCH_WORKERS` | automatic | Explicit comma-separated ladder |
| `FFMPEG_BENCH_TIMEOUT_SECONDS` | `7200` | Whole benchmark deadline |
| `FFMPEG_BENCH_REPETITION_TIMEOUT_SECONDS` | `max(30, target×6)` | Pilot/repetition deadline |
| `FFMPEG_BENCH_AUDIO_SOURCE` | bundled fixture | Alternate local FLAC |
| `FFMPEG_BENCH_VIDEO_SOURCE` | bundled fixture | Alternate local MP4 |
| `FFMPEG` / `FFPROBE` | PATH binaries | Alternate FFmpeg build |

### Why five seconds

A localhost study ran every available CPU codec for five repetitions at 2, 5,
and 10-second targets. At five seconds, the median coefficient of variation was
2.2%; seven of eight codecs were within 10% after the first three repetitions,
and only H.264 decode requested extra runs. At two seconds only six of eight
were within 10%. Ten-second runs were less stable because ambient host load
drift dominated, demonstrating that a longer sample does not fix an
uncontrolled machine. NVIDIA scenarios were discovered but skipped because the
study host had no CUDA GPU.

Five seconds is therefore the shortest reliable default observed on the test
host. Large machines do not automatically need longer runs: the per-worker
count pilot scales media volume to keep wall time constant, so faster machines
process more source data in those five seconds. Increase the duration only for
especially noisy or externally shared hosts.

The null muxer intentionally excludes object-store and filesystem output; use
a separate storage benchmark when production write throughput matters. FFmpeg
describes the [null muxer as intended for testing and benchmarking](https://ffmpeg.org/ffmpeg-formats.html#null).

## Output

One compact JSON document is written to stdout
(`benchmark=ffmpeg_transcoding`, `version=3.0.0`); logs go to stderr.
Version 3 intentionally removes all derived rollups. Each repetition contains
`wall_time_sec`, worker outcomes, and either:

- `processed_frames` plus `aggregate_fps`, or
- `processed_audio_seconds` plus `audio_seconds_per_sec`.

## Local development

```bash
python3 images/benchmark-ffmpeg/test_benchmark.py
python3 -m py_compile images/benchmark-ffmpeg/benchmark.py
images/benchmark-ffmpeg/prepare-video-fixture.sh   # optional: refresh source.mp4
images/benchmark-ffmpeg/upload-fixtures.sh         # upload to sc-cdn (AWS profile sc)
docker build -t benchmark-ffmpeg:local images/benchmark-ffmpeg
```

The image builds FFmpeg 9.0.1 from source (see `build-ffmpeg.sh`, `BUILD_ARGS`).
Expect several minutes per architecture on a cold build; CI enables zram for the
compile stage. Override `FFMPEG_VERSION` or `NV_CODEC_HEADERS_TAG` via
`docker build --build-arg` when testing newer upstream releases.
