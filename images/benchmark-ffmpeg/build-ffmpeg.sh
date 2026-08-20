#!/usr/bin/env bash
# Build a throughput-oriented FFmpeg for the benchmark image.
#
# amd64 + arm64: GPL CPU codecs (x264/x265/vorbis/opus/lame + native AAC/FLAC)
# and dynamically loaded NVENC/NVDEC (datacenter GPUs: T4/L4/… and arm64 T4G
# on g5g). Jetson/nvmpi is not supported. NVENC has no audio codecs.
set -euo pipefail

: "${TARGETARCH:?TARGETARCH is required}"
: "${FFMPEG_VERSION:?FFMPEG_VERSION is required}"

PREFIX="${PREFIX:-/usr/local}"
BUILD_DIR="${BUILD_DIR:-/tmp/ffmpeg-build}"
NPROC="${NPROC:-$(nproc)}"
NV_CODEC_HEADERS_TAG="${NV_CODEC_HEADERS_TAG:-n13.0.19.1}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -d "ffmpeg-${FFMPEG_VERSION}" ]; then
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
        | tar -xJ
fi
cd "ffmpeg-${FFMPEG_VERSION}"

NVIDIA_FLAGS=()
case "$TARGETARCH" in
    amd64|arm64)
        if [ ! -d nv-codec-headers ]; then
            git clone --depth 1 --branch "$NV_CODEC_HEADERS_TAG" \
                https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
                nv-codec-headers
        fi
        make -C nv-codec-headers install PREFIX="$PREFIX"
        NVIDIA_FLAGS=(
            --enable-nonfree
            --enable-cuvid
            --enable-nvenc
            --enable-ffnvcodec
        )
        ;;
    *)
        echo "unsupported TARGETARCH: $TARGETARCH" >&2
        exit 1
        ;;
esac

./configure \
    --prefix="$PREFIX" \
    --enable-gpl \
    --enable-version3 \
    --enable-shared \
    --disable-static \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --enable-indev=lavfi \
    --enable-pthreads \
    --enable-hardcoded-tables \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvorbis \
    --enable-libopus \
    --enable-libmp3lame \
    "${NVIDIA_FLAGS[@]}"

make -j"$NPROC"
make install
ldconfig

export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
ffmpeg -hide_banner -version | head -1
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265|libvorbis|libopus|libmp3lame|aac |h264_nvenc|hevc_nvenc' || true
ffmpeg -hide_banner -decoders 2>/dev/null | grep -E 'h264_cuvid|hevc_cuvid|h264 |hevc ' || true
