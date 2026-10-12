# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### Output Quality

This benchmark measures throughput, not the quality of the encoded output. It
computes no quality metric, such as PSNR ([Peak Signal-to-Noise
Ratio](https://ffmpeg.org/ffmpeg-filters.html#psnr)), SSIM
([Structural Similarity Index
Measure](https://ffmpeg.org/ffmpeg-filters.html#ssim)), or VMAF ([Video
Multimethod Assessment Fusion](https://github.com/Netflix/vmaf)). The encode
settings are fixed per scenario, so the quality target stays the same across
server types; see [CPU and GPU Encode Quality](#cpu-and-gpu-encode-quality)
for how CPU and GPU settings relate.

### Storage and Output

Every job sends its encoded output to FFmpeg's [null
muxer](https://ffmpeg.org/ffmpeg-formats.html#null) (`-f null -`), which
discards it. This benchmark therefore does *not* measure disk, file system, or
object-storage writes, and a transcoding service that stores its output can be
limited by storage instead.

### A Specific Media Pipeline

All jobs use the same pinned test media and fixed codec settings: a film clip
of 30&nbsp;s at 1920x800 pixels and 24&nbsp;fps, and a music sample of about
85&nbsp;s. See [Test Media](./workloads.md#test-media) for details.

The results do *not* predict the speed of other resolutions, frame rates,
content, codec settings, or codecs, such as AV1 ([AOMedia Video
1](https://aomedia.org/specifications/av1/)). They give a relative sense of how
much of this fixed work each server type completes.

### Single-Job Speed

This benchmark measures the combined throughput of many parallel jobs, each
limited to one codec thread. The one-worker step shows the speed of a single
such job, not of one `ffmpeg` process that spreads a single stream across many
threads. See [One Codec Thread per Job](#one-codec-thread-per-job).

## Scope Decisions

### One Codec Thread per Job

Each worker is an independent `ffmpeg` process limited to one codec thread
([`benchmark.py`](../benchmark.py)). `libx265` additionally runs with one frame
thread and no thread pool. Throughput on larger servers comes from running more
workers in parallel, not from more threads per job.

### Physical Cores, Not Logical CPUs

The CPU worker counts are based on physical cores rather than logical CPUs.
Two single-threaded workers on the two hyperthreads of one core compete for
that core, so counting logical CPUs over-subscribes servers with SMT
(Simultaneous Multithreading) and can lower throughput. When throughput is
still rising at the physical core count, one extra measurement at twice that
count shows whether SMT helps; see [Worker Counts](./usage.md#worker-counts).

### Audio on the CPU Only

FFmpeg's NVIDIA encoders and decoders handle video only, and there is no NVIDIA
hardware encoder for Vorbis, Opus, AAC (Advanced Audio Coding), MP3, or FLAC
(Free Lossless Audio Codec). The audio scenarios therefore always run on the
CPU, and a GPU does not change the audio results. The output records this as
`audio_gpu_acceleration: false`. See the [NVIDIA FFmpeg
guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)
for the supported video codecs.

### NVIDIA GPUs Only

The FFmpeg build in this image enables only NVIDIA's video acceleration: NVENC
([NVIDIA Video
Encoder](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html))
and NVDEC (NVIDIA Video Decoder). It enables no other GPU backend, so other vendors' GPUs are
not used. Within NVIDIA, the build targets datacenter GPUs; Jetson devices are
not supported.

### Raw Results Only

The output contains every measured repetition, but no headline score, peak, or
summary statistic. You decide how to aggregate the repetitions; see
[Results](../README.md#results).

## Design Constraints

### CPU and GPU Encode Quality

NVENC has no CRF ([Constant Rate
Factor](https://trac.ffmpeg.org/wiki/Encode/H.264#crf)), the quality setting the
CPU encodes use. The scenarios use the following settings:

- CPU encodes: `libx264` and `libx265` with `-crf 18` and the encoders' default
  preset (medium).
- GPU encodes: `h264_nvenc` and `hevc_nvenc` with `-preset p4 -rc vbr -cq 22
  -b:v 0`, a constant-quality mode.

A sweep of NVENC quality levels on the test clip found that `-cq 22` matched
`libx264 -crf 18` most closely, at an SSIM to the source of about 0.9937 and a
bitrate of about 4&nbsp;Mbps. The match is approximate, not exact, so compare
CPU and GPU encode rates with that in mind. See the [NVENC CQ (Constant
Quality) sweep](../CHANGELOG.md#nvenc-cq-sweep) for the data.

### GPU Coverage

Before each scenario, a one-worker probe checks that it can run. A scenario
whose probe fails is skipped with the error text as its `skip_reason`. This
happens, for example, when a GPU cannot open an NVENC or NVDEC session. A
missing GPU result therefore means the scenario could not run in that
environment.

GPU scenarios also need the NVIDIA driver's video libraries inside the
container. Since version 3.3.0, the image requests them itself; earlier
production runs on many GPU servers skipped all GPU scenarios because these
libraries were missing. Production results for GPUs are therefore incomplete
until those servers are measured again.

### Computed Media Counts

`processed_frames` and `processed_audio_seconds` are computed from the media
length each worker was asked to process, not read from FFmpeg's output:
successful workers × media length (× the clip's frame rate, for video). A failed
or timed-out worker adds no media, but the group's wall time still includes it,
so failures lower the rate.
