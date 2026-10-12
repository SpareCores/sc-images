# Changelog

## NVENC Quality Matched to CPU Encodes and GPU Driver Libraries Mounted

Harness version 3.3.0 changes the GPU encode scenarios and the image's GPU
setup:

- `gpu_h264_encode` and `gpu_h265_encode` now run NVENC with `-preset p4 -rc
  vbr -cq 22 -b:v 0` instead of `-preset p4` with the default rate control.
  `-cq 22` was the closest match to the CPU encodes at `-crf 18`; see
  [NVENC CQ Sweep](#nvenc-cq-sweep). The match is approximate, not exact.
- The image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, so the
  NVIDIA Container Toolkit mounts the driver's NVENC and NVDEC libraries.
  Before this change, many production GPU runs skipped all GPU scenarios with
  `Cannot load libnvcuvid.so.1`.

## FFmpeg Built Without `--enable-nonfree`

`build-ffmpeg.sh` no longer passes `--enable-nonfree`. The NVIDIA options
(`--enable-cuvid`, `--enable-nvenc`, and `--enable-ffnvcodec`) do not need it
in FFmpeg 9 and later, and it would make the binary non-redistributable.

## Audio Profiles: Opus, AAC, and MP3 Instead of Extra Vorbis Bitrates

Moved from the previous README:

> Multiple Vorbis bitrates were dropped: on real hosts 96/160/320 kbps scored
> within \~±15%, so they did not separate instances. The suite now covers
> distinct lossy codecs professionals actually choose, plus FLAC.

## Output Version 3: Raw Repetitions Only

Moved from the previous README:

> Version 3 intentionally removes all derived rollups.

> There are no converted capacity units, peaks, recommendations, summary
> statistics, or duplicated realtime factors in the output. Consumers receive
> the source counts, wall-clock time, rates, worker outcomes, and finish spread
> for every repetition and can aggregate those measurements as needed.

## Experiments That Informed the Design

The following notes are moved from the previous README, with only punctuation
adapted to the style manual.

### NVENC CQ Sweep

NVENC has no CRF. Preset `p4` is FFmpeg's medium default. A CQ sweep on the
pinned Tears of Steel H.264 fixture (`source.mp4`, software decode into both
encoders) was run on UpCloud `GPU-32xCPU-160GB-2xRTXPRO6000` (2× RTX PRO 6000
Blackwell) to pick the closest constant-quality match to CPU `-crf 18`. Scales
still differ by encoder family -- approximate parity, not bit-exact.

**H.264 / H.265 encode vs source (CQ 18 vs CRF 18):**

| Encode | Bytes | Bitrate (bps) | SSIM All vs source | PSNR avg |
| --- | ---: | ---: | ---: | ---: |
| `libx264 -crf 18 -preset medium` | 15165835 | 4036087 | 0.993770 | 48.962137 |
| `h264_nvenc -preset p4 -rc vbr -cq 18 -b:v 0` | 28290834 | 7531112 | 0.995426 | 51.395851 |
| `libx265 -crf 18 -preset medium` | 11696839 | 3111461 | 0.993383 | 48.895972 |
| `hevc_nvenc -preset p4 -rc vbr -cq 18 -b:v 0` | 25329289 | 6742451 | 0.995170 | 51.142542 |

GPU vs. CPU pairwise SSIM All: H.264 0.993345, H.265 0.993799.

**H.264 NVENC CQ sweep vs `libx264 -crf 18` (same fixture):**

| Setting | Bytes | Bitrate (bps) | SSIM All vs source | SSIM All vs CPU |
| --- | ---: | ---: | ---: | ---: |
| CRF 18 (CPU) | 15165835 | 4036087 | 0.993770 | n/a |
| CQ 15 | 40578290 | 10803186 | 0.996467 | 0.993435 |
| CQ 16 | 36600522 | 9743939 | 0.996173 | 0.993424 |
| CQ 17 | 32627231 | 8685869 | 0.995861 | 0.993427 |
| CQ 18 | 28290834 | 7531112 | 0.995426 | 0.993345 |
| CQ 19 | 24934170 | 6637265 | 0.995019 | 0.993288 |
| CQ 20 | 21833713 | 5811627 | 0.994608 | 0.993289 |
| CQ 21 | 19143406 | 5095219 | 0.994195 | 0.993186 |
| CQ 22 | 16652394 | 4431879 | 0.993700 | 0.993010 |
| CQ 23 | 14553619 | 3872988 | 0.993181 | 0.992821 |
| CQ 24 | 12590211 | 3350150 | 0.992607 | 0.992560 |

`-cq 22` is closest to CRF 18 on SSIM-to-source (0.993700 vs 0.993770) and
bitrate (4431879 vs 4036087). `-cq 18` is higher quality at 1.87× the CPU
bitrate. The GPU encode scenarios therefore use `-cq 22`.

### Why 5 Seconds

A localhost study ran every available CPU codec for five repetitions at 2, 5,
and 10&nbsp;second targets. At 5&nbsp;seconds, the median coefficient of
variation was 2.2%; seven of eight codecs were within 10% after the first three
repetitions, and only H.264 decode requested extra runs. At 2&nbsp;seconds only
six of eight were within 10%. Runs of 10&nbsp;seconds were less stable because
ambient host load drift dominated, demonstrating that a longer sample does not
fix an uncontrolled machine. NVIDIA scenarios were discovered but skipped
because the study host had no CUDA GPU.

A target of 5&nbsp;seconds is therefore the shortest reliable default observed
on the test host. Large machines do not automatically need longer runs: the
per-worker count pilot scales media volume to keep wall time constant, so
faster machines process more source data in those 5&nbsp;seconds. Increase the
duration only for especially noisy or externally shared hosts.

### Worker Scaling on Large Instances

The worker ladder is built from **physical cores** (SMT siblings collapsed via
`thread_siblings_list`), not `nproc`. On this class of instance that is 48
cores rather than 96 hyperthreads. CPU scenarios measure **1, P/2, and P**. If
aggregate throughput at P is still at least 8% above P/2, one SMT probe at
`min(2P, logical)` is added. Memory-bound work (H.264 decode) typically peaks
at P/2 and skips the HT point; compute-bound encode may keep the extra sample.

GPU scenarios measure **1 and G** (one session per GPU), then **double** (2G,
4G, ...) until throughput falls or a session ceiling is hit. That finds the
NVENC/NVDEC sweet spot without assuming 8 sessions per GPU. Failed worker
counts stop the search immediately.

### Aggregate Timing Rationale

Audio encoding does not scale one stream across many cores: neither
`libvorbis`, `libopus`, `libmp3lame`, native AAC, nor FFmpeg's FLAC encoder
advertises frame or slice threading useful for this workload.
The benchmark therefore runs independent FFmpeg processes with one codec
thread each, matching a batch transcoding service.

It never sums per-process `speed=` values, which would overstate capacity when
workers start or finish at different times. Startup and teardown are included.
FFmpeg documents both its
[benchmark options](https://ffmpeg.org/ffmpeg.html#Advanced-options) and
[machine-readable progress output](https://ffmpeg.org/ffmpeg.html#Main-options);
the harness uses its own group clock because FFmpeg only knows about one
process.
