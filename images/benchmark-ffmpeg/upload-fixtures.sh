#!/usr/bin/env bash
# Upload benchmark fixtures to sc-cdn (S3 bucket sc-cdn-cae3awai).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="${SC_CDN_BUCKET:-sc-cdn-cae3awai}"
PREFIX="${SC_CDN_PREFIX:-sc-inspector/benchmark-ffmpeg}"
PROFILE="${AWS_PROFILE:-sc}"

for name in source.flac source.mp4 source-hevc.mp4; do
    path="${ROOT}/${name}"
    if [ ! -f "${path}" ]; then
        echo "missing ${path}" >&2
        exit 1
    fi
done

aws s3 cp "${ROOT}/source.flac" "s3://${BUCKET}/${PREFIX}/source.flac" \
    --profile "${PROFILE}" --content-type audio/flac
aws s3 cp "${ROOT}/source.mp4" "s3://${BUCKET}/${PREFIX}/source.mp4" \
    --profile "${PROFILE}" --content-type video/mp4
aws s3 cp "${ROOT}/source-hevc.mp4" "s3://${BUCKET}/${PREFIX}/source-hevc.mp4" \
    --profile "${PROFILE}" --content-type video/mp4

echo "https://cdn.sparecores.net/${PREFIX}/source.flac"
echo "https://cdn.sparecores.net/${PREFIX}/source.mp4"
echo "https://cdn.sparecores.net/${PREFIX}/source-hevc.mp4"
