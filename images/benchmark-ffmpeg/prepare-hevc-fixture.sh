#!/usr/bin/env bash
# Transcode the pinned Tears of Steel H.264 excerpt to HEVC for decode scenarios.
#
# Encode and H.264 decode reuse source.mp4. H.265 decode (CPU hevc / GPU
# hevc_cuvid) needs an HEVC bitstream of the same clip; regenerating from the
# H.264 excerpt keeps geometry/fps/duration aligned without a second upstream
# download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${ROOT}/source.mp4"
OUTPUT="${ROOT}/source-hevc.mp4"
VIDEO_HEVC_FIXTURE_SHA256="${VIDEO_HEVC_FIXTURE_SHA256:-79a70a6aa81e745d650621768dee5cd3fc1da7d2c15d39871125737064f6cde7}"
FFMPEG="${FFMPEG:-ffmpeg}"

if ! command -v "${FFMPEG}" >/dev/null; then
    echo "ffmpeg is required" >&2
    exit 1
fi
if [ ! -f "${INPUT}" ]; then
    echo "missing ${INPUT}; run ./prepare-video-fixture.sh first" >&2
    exit 1
fi

"${FFMPEG}" -nostdin -hide_banner -loglevel error -y \
    -i "${INPUT}" -an \
    -c:v libx265 -crf 18 -preset medium -pix_fmt yuv420p \
    -tag:v hvc1 -movflags +faststart \
    "${OUTPUT}"

echo "${VIDEO_HEVC_FIXTURE_SHA256}  ${OUTPUT}" | sha256sum --check --strict
echo "wrote ${OUTPUT}"
echo "Upload with: ${ROOT}/upload-fixtures.sh"
