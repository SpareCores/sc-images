# Audio benchmark fixture

`source.flac` is downloaded while building the image from Wikimedia Commons:

- Work: “Entre dos Aguas” sample, performed by Michael Laucke
- Format: FLAC, approximately 85 seconds and 8.31 MB
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Source and license record: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Entre-dos-aguas-laucke-version-sample.flac)
- SHA-256: `4445399abe62c9d7c546711a853fccfab8ab274226d2e80aa0e5ad948589e516`

The image build verifies this checksum. The benchmark loops the local fixture
in memory to obtain a sufficiently long measured interval without storing an
expanded copy.

## FFmpeg

The image compiles FFmpeg from upstream release tarballs instead of using the
distro package. Build metadata is pinned in `BUILD_ARGS`:

- `FFMPEG_VERSION` — currently `9.0.1` from [ffmpeg.org/releases](https://ffmpeg.org/releases/)
- `NV_CODEC_HEADERS_TAG` — currently `n13.0.19.1` for amd64 NVENC/NVDEC only

The build script `build-ffmpeg.sh` enables only what the benchmark needs:

- CPU: `libx264`, `libx265`, `libvorbis`, native FLAC
- amd64 NVIDIA: `ffnvcodec`, `cuvid`, `nvenc` (dynamically loaded against the
  host driver at runtime; no CUDA toolkit in the runtime image)

arm64 images omit NVIDIA support because this benchmark's GPU scenarios target
NVIDIA data-center GPUs on x86_64 hosts.
