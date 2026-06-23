#!/usr/bin/env bash
# Pre-build hook for hammerdb-postgres: RAM-aware compile parallelism for arm64.
#
# Runs from the repo root. Emits to $GITHUB_OUTPUT:
#   buildkit_max_parallelism
# and appends runtime build-args to .cache/extra-build-args.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")/../../.github/scripts" && pwd)"
IMG_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE="$(cd "$(dirname "$0")/../.." && pwd)/.cache"
mkdir -p "$CACHE"

HAMMERDB_VERSION="$(tr -d '[:space:]' < "${IMG_DIR}/HAMMERDB_VERSION")"

{
  echo "HAMMERDB_VERSION=${HAMMERDB_VERSION}"
} >> "${CACHE}/extra-build-args"

if [ "${ARCH:-}" = "arm64" ]; then
  export VLLM_COMPILE_GIB_PER_SLOT="${HAMMERDB_COMPILE_GIB_PER_SLOT:-4}"
  export COMPILE_RAM_SOURCE=physical
  PAR="${CACHE}/parallelism.out"
  bash "${SCRIPTS}/compute-build-parallelism.sh" buildkit > "$PAR"
  NUM_JOBS="$(sed -n 's/^max_jobs=//p' "$PAR" | head -1)"
  BUILDKIT="$(sed -n 's/^buildkit_max_parallelism=//p' "$PAR" | head -1)"
  echo "NUM_JOBS=${NUM_JOBS}" >> "${CACHE}/extra-build-args"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "buildkit_max_parallelism=${BUILDKIT}" >> "$GITHUB_OUTPUT"
  fi
fi
