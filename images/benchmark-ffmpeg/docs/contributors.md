# Contributor Guide

## Local Development

Run the unit tests and a syntax check from the `sc-images` root:

```bash
python3 images/benchmark-ffmpeg/test_benchmark.py
python3 -m py_compile images/benchmark-ffmpeg/benchmark.py
```

The tests in [`test_benchmark.py`](../test_benchmark.py) are not copied into
the image, and CI does not run them, so run them before you open a pull
request.

Build the image locally from the image folder:

```bash
docker build -t benchmark-ffmpeg:local images/benchmark-ffmpeg
```

## FFmpeg Build

The image compiles FFmpeg from the upstream release tarball rather than using
the Ubuntu package. [`build-ffmpeg.sh`](../build-ffmpeg.sh) enables only what
this benchmark needs:

- CPU codecs: `libx264`, `libx265`, `libvorbis`, `libopus`, `libmp3lame`, and
  FFmpeg's native AAC
  ([Advanced Audio Coding](https://ffmpeg.org/ffmpeg-codecs.html#aac)) and FLAC
  ([Free Lossless Audio Codec](https://xiph.org/flac/)) encoders.
- NVIDIA video: `--enable-cuvid --enable-nvenc --enable-ffnvcodec`, built
  against `nv-codec-headers`. The NVIDIA libraries are loaded from the host
  driver at run time.

The build supports `amd64` and `arm64` only, and both include the NVIDIA
support. It does not use `--enable-nonfree`: the NVIDIA options do not need it
in FFmpeg 9 and later, and it would make the binary non-redistributable.

The versions are pinned in [`BUILD_ARGS`](../BUILD_ARGS) and can be overridden
with `docker build --build-arg` when you test a newer upstream release:

- `FFMPEG_VERSION`: `9.0.1`
- `NV_CODEC_HEADERS_TAG`: `n13.0.19.1`

The build downloads from `ffmpeg.org`, `git.videolan.org`, and the Spare Cores
CDN (Content Delivery Network). CI enables compressed swap
([`zram`](https://docs.kernel.org/admin-guide/blockdev/zram.html)) on the
builder for the FFmpeg compile, set by the [`ZRAM`](../ZRAM) file.

## Image Metadata

CI publishes two files from this folder as image labels on the GitHub Container
Registry:

- [`DESCRIPTION`](../DESCRIPTION): the one-sentence summary, shown as the image
  description. Keep it consistent with the opening of the
  [README](../README.md).
- [`CATEGORY`](../CATEGORY): `benchmark`.

## Test Media

The image bundles three test files, served from the Spare Cores CDN and
described in [`SOURCE.md`](../SOURCE.md). The `Dockerfile` copies `SOURCE.md`
into the image at `/opt/benchmark-ffmpeg/SOURCE.md`, so keep the file.

To regenerate the files, run the following scripts from this folder:

1. [`prepare-video-fixture.sh`](../prepare-video-fixture.sh) downloads the
   Tears of Steel 1080p release and cuts the H.264 excerpt of 30&nbsp;s,
   starting at 60&nbsp;s, with stream copy, into `source.mp4`.
2. [`prepare-hevc-fixture.sh`](../prepare-hevc-fixture.sh) transcodes
   `source.mp4` into `source-hevc.mp4` with `libx265 -crf 18 -preset medium`.
3. Add the FLAC music sample as `source.flac`.
4. [`upload-fixtures.sh`](../upload-fixtures.sh) uploads all three files to the
   S3 ([Simple Storage Service](https://aws.amazon.com/s3/)) bucket
   `sc-cdn-cae3awai`, under the prefix `sc-inspector/benchmark-ffmpeg`, with
   the AWS ([Amazon Web Services](https://aws.amazon.com/)) profile `sc`.

The prepare scripts check their output against the pinned SHA-256
([Secure Hash Algorithm 256](https://csrc.nist.gov/pubs/fips/180-4/upd1/final))
hashes. If you change a file's content, update its hash in every place it is
pinned:

- [`BUILD_ARGS`](../BUILD_ARGS) and the `ARG` defaults in the
  [`Dockerfile`](../Dockerfile), which verify the downloads at build time.
- [`benchmark.py`](../benchmark.py), which verifies the video files again at
  run time and reports the audio hash in its output.
- The defaults in `prepare-video-fixture.sh` and `prepare-hevc-fixture.sh`.

The fixture files themselves are ignored by Git.
