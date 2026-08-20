#!/usr/bin/env bash
# Download Tears of Steel (1080p) and cut the pinned 30-second H.264 excerpt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${ROOT}/source.mp4"
WORK="${TMPDIR:-/tmp}/tos-fixture-$$"
ARCHIVE="${WORK}/archive"
FULL="${WORK}/tos-full.mov"

VIDEO_FIXTURE_URL="${VIDEO_FIXTURE_URL:-http://ftp.halifax.rwth-aachen.de/blender/demo/movies/ToS/tears_of_steel_1080p.mov}"
VIDEO_FIXTURE_SHA256="${VIDEO_FIXTURE_SHA256:-a49fe8b82c96bafcc344d374facb716f481e3fa9c6753d56f6d1e0ed509a14e7}"
VIDEO_FIXTURE_OFFSET_SEC="${VIDEO_FIXTURE_OFFSET_SEC:-60}"
VIDEO_FIXTURE_DURATION_SEC="${VIDEO_FIXTURE_DURATION_SEC:-30}"

FFMPEG="${FFMPEG:-ffmpeg}"

if ! command -v curl >/dev/null || ! command -v "${FFMPEG}" >/dev/null; then
    echo "curl and ffmpeg are required" >&2
    exit 1
fi

mkdir -p "${WORK}"
trap 'rm -rf "${WORK}"' EXIT

curl --fail --location --retry 3 \
    --output "${ARCHIVE}" \
    "${VIDEO_FIXTURE_URL}"

if file -b "${ARCHIVE}" | grep -qi zip; then
    unzip -q "${ARCHIVE}" -d "${WORK}"
    FULL="${WORK}/tears_of_steel_1080p.mov"
else
    FULL="${ARCHIVE}"
fi

"${FFMPEG}" -nostdin -hide_banner -loglevel error -y \
    -ss "${VIDEO_FIXTURE_OFFSET_SEC}" -t "${VIDEO_FIXTURE_DURATION_SEC}" \
    -i "${FULL}" -an -c:v copy "${OUTPUT}"

echo "${VIDEO_FIXTURE_SHA256}  ${OUTPUT}" | sha256sum --check --strict
echo "wrote ${OUTPUT}"
echo "Next: ./prepare-hevc-fixture.sh then ./upload-fixtures.sh (requires source.flac, source.mp4, source-hevc.mp4)"
