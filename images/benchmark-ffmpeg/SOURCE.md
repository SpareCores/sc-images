# Benchmark fixtures

Fixtures are served from [sc-cdn](https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/)
(S3 bucket `sc-cdn-cae3awai`, prefix `sc-inspector/benchmark-ffmpeg/`). The image
build downloads and verifies them; runtime needs no network access.

| File | CDN URL | SHA-256 |
|---|---|---|
| `source.flac` | https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/source.flac | `4445399abe62c9d7c546711a853fccfab8ab274226d2e80aa0e5ad948589e516` |
| `source.mp4` | https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/source.mp4 | `a49fe8b82c96bafcc344d374facb716f481e3fa9c6753d56f6d1e0ed509a14e7` |
| `source-hevc.mp4` | https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg/source-hevc.mp4 | `79a70a6aa81e745d650621768dee5cd3fc1da7d2c15d39871125737064f6cde7` |

## Audio

“Entre dos Aguas” sample, performed by Michael Laucke (FLAC, ~85 s, CC0 1.0).
Original: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Entre-dos-aguas-laucke-version-sample.flac).

## Video

30-second H.264 excerpt from the Blender Foundation open movie **Tears of Steel**
([1080p release](https://mango.blender.org/download/)): 1920×800, 24 fps, cut at
60–90 s with stream copy → `source.mp4`. Upstream mirror:
`ftp.halifax.rwth-aachen.de/.../tears_of_steel_1080p.mov` (zip-wrapped MOV).
License: [CC BY 3.0](https://mango.blender.org/sharing/).

`source-hevc.mp4` is a local `libx265 -crf 18 -preset medium` transcode of that
same excerpt so H.265 decode scenarios have a real HEVC bitstream without a
second upstream download. See `README.md` (Fixtures) and
`prepare-hevc-fixture.sh`.

Regenerate locally with `./prepare-video-fixture.sh`, `./prepare-hevc-fixture.sh`,
add `source.flac`, then `./upload-fixtures.sh` (AWS profile `sc`).

## FFmpeg

The image compiles FFmpeg from upstream release tarballs instead of using the
distro package. Build metadata is pinned in `BUILD_ARGS`:

- `FFMPEG_VERSION` — currently `9.0.1` from [ffmpeg.org/releases](https://ffmpeg.org/releases/)
- `NV_CODEC_HEADERS_TAG` — currently `n13.0.19.1` for amd64 NVENC/NVDEC only

The build script `build-ffmpeg.sh` enables only what the benchmark needs:

- CPU: `libx264`, `libx265`, `libvorbis`, `libopus`, `libmp3lame`, native AAC,
  native FLAC
- NVIDIA (amd64 + arm64): `ffnvcodec`, `cuvid`, `nvenc` (dynamically loaded
  against the host driver at runtime; no CUDA toolkit in the runtime image) —
  covers `h264_nvenc` / `hevc_nvenc` / `h264_cuvid` / `hevc_cuvid`. Targets
  datacenter GPUs (T4/L4/… and arm64 T4G on AWS g5g). Jetson/`nvmpi` is not
  supported. NVENC/CUVID have no audio codecs.
