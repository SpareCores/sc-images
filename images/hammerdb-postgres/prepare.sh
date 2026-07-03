#!/usr/bin/env bash
# Pre-build hook for hammerdb-postgres: emit version build-arg.
#
# Runs from the repo root. Appends runtime build-args to .cache/extra-build-args.
set -euo pipefail

IMG_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE="$(cd "$(dirname "$0")/../.." && pwd)/.cache"
mkdir -p "$CACHE"

HAMMERDB_VERSION="$(tr -d '[:space:]' < "${IMG_DIR}/HAMMERDB_VERSION")"

{
  echo "HAMMERDB_VERSION=${HAMMERDB_VERSION}"
} >> "${CACHE}/extra-build-args"
