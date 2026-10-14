# Workloads

## Audio Encoding

Every audio job decodes the bundled FLAC
(Free Lossless Audio Codec) music sample, resamples
it when needed, and encodes stereo output with one codec thread, using Vorbis,
Opus, AAC (Advanced Audio Coding),
MP3, or FLAC. The audio scenarios run on the CPU only; see
[Audio on the CPU Only](./limitations.md#audio-on-the-cpu-only).

| Scenario | Encoder | Settings | Sample rate |
| --- | --- | --- | --- |
| `ogg_vorbis_160k` | `libvorbis` | 160 kbps | 44.1 kHz |
| `opus_128k` | `libopus` | 128 kbps | 48 kHz |
| `aac_128k` | `aac` (FFmpeg's native AAC encoder) | 128 kbps | 44.1 kHz |
| `mp3_192k` | `libmp3lame` | 192 kbps | 44.1 kHz |
| `flac_lossless` | `flac` (FFmpeg's native FLAC encoder) | Compression level 5 | 44.1 kHz |

`opus_128k` uses 48 kHz because `libopus` accepts only 8, 12, 16, 24, or
48 kHz, not the sample's 44.1 kHz.

Audio results are reported as `audio_seconds_per_sec`: seconds of audio
processed per second of wall time, across all workers.

## CPU Video

The CPU video scenarios use FFmpeg's software codecs, with one codec thread per
job for both decoding and encoding:

| Scenario | Operation | Input | Decoder | Encoder and settings |
| --- | --- | --- | --- | --- |
| `cpu_h264_encode` | Encode | H.264 clip | `h264` | `libx264 -crf 18` |
| `cpu_h265_encode` | Encode | H.264 clip | `h264` | `libx265 -crf 18`, one frame thread, no thread pool |
| `cpu_h264_decode` | Decode | H.264 clip | `h264` | None |
| `cpu_h265_decode` | Decode | H.265 clip | `hevc` | None |

The encode scenarios also decode their H.264 input on the CPU, so they measure
a full transcode. CRF
(Constant Rate Factor) 18 is a
quality target; no `-preset` is set, so `libx264` and `libx265` use their
default preset (medium).

Video results are reported as `aggregate_fps`: frames processed per second of
wall time, across all workers.

## NVIDIA Video

The NVIDIA video scenarios decode on the GPU and, for encodes, keep the frames
in GPU memory for NVENC
(NVIDIA Video Encoder):

| Scenario | Operation | Input | Decoder | Encoder and settings |
| --- | --- | --- | --- | --- |
| `gpu_h264_encode` | Encode | H.264 clip | `h264_cuvid` | `h264_nvenc -preset p4 -rc vbr -cq 22 -b:v 0` |
| `gpu_h264_decode` | Decode | H.264 clip | `h264_cuvid` | None |
| `gpu_h265_encode` | Encode | H.264 clip | `h264_cuvid` | `hevc_nvenc -preset p4 -rc vbr -cq 22 -b:v 0` |
| `gpu_h265_decode` | Decode | H.265 clip | `hevc_cuvid` | None |

Each job runs with `-hwaccel cuda` on one GPU. The `-cq 22` constant-quality
setting was chosen to approximately match the CPU encodes at CRF 18; see
[CPU and GPU Encode Quality](./limitations.md#cpu-and-gpu-encode-quality).

## Test Media

The image build downloads the test media from the Spare Cores CDN (Content
Delivery Network) and checks each file's SHA-256
(Secure Hash Algorithm 256)
checksum. At run time, the H.264 and H.265 clips are checked again; the audio
file is not. The files are the following:

- **H.264 clip** (`source.mp4`): an excerpt of 30 s from the Blender
  Foundation's open movie [Tears of Steel](https://mango.blender.org/), cut at
  60 s from the 1080p release with stream copy. It is 1920x800 pixels at
  24 fps. License: CC BY 3.0
  (Creative Commons Attribution 3.0),
  as stated on the
  [Tears of Steel sharing page](https://mango.blender.org/sharing/).
- **H.265 clip** (`source-hevc.mp4`): the same excerpt, transcoded with
  `libx265 -crf 18 -preset medium`, so the H.265 decode scenarios get the same
  frame size, frame rate, and duration.
- **Music sample** (`source.flac`): the "Entre dos Aguas" sample performed by
  Michael Laucke, about 85 s at 44.1 kHz. License: CC0 1.0
  (Creative Commons Zero 1.0).

Each job loops its input (`-stream_loop -1`) and stops after the media length
set for that worker count (`-t`), so the length of a job does not depend on the
length of the file. The harness ([`benchmark.py`](../benchmark.py)) probes each
file's resolution, frame rate, and duration at run time and reports them in
`sources`.

## How Measurement Works

This benchmark runs its 13 scenarios in a fixed order, as defined in
[`benchmark.py`](../benchmark.py): the eight video scenarios first, then the
five audio scenarios. Each repetition measures one group of parallel workers in
the following steps:

1. The harness starts every worker as a separate process, held back by a shared
   pipe.
2. It reads a monotonic clock and closes the pipe, which releases all workers
   at once.
3. It records each worker's exit time and exit code. The wall time is the time
   from the release to the last exit, so process startup and teardown are
   included.
4. It computes the rate: successful workers × media length per worker, divided
   by the wall time. For video, the media length is converted to frames with
   the clip's frame rate.

The harness never adds up the speed that each `ffmpeg` process reports for
itself. Workers that time out are stopped, recorded with exit code 124, and
count as failed. Encoded output goes to the null muxer and is discarded; see
[Storage and Output](./limitations.md#storage-and-output).

See [Worker Counts](./usage.md#worker-counts) for how many workers each group
has, and [Run Duration](./usage.md#run-duration) for the repetitions and
timeouts.
