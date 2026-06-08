#!/usr/bin/env bash
# Install the pinned resource-tracker binary for the current runner arch.
set -euo pipefail

ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
VERSION="$(tr -d '[:space:]' < "${ROOT}/images/resource-tracker/RESOURCE_TRACKER_VERSION")"
ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')"
DEST="${RUNNER_TOOL_CACHE:-/tmp}/resource-tracker/${VERSION}-${ARCH}"
BIN="${DEST}/resource-tracker"

if [ -x "$BIN" ]; then
  echo "resource-tracker v${VERSION} (${ARCH}) cached at ${BIN}"
else
  mkdir -p "$DEST"
  URL="https://github.com/SpareCores/resource-tracker-rs/releases/download/v${VERSION}/resource-tracker-v${VERSION}-linux-${ARCH}.tar.gz"
  echo "Downloading resource-tracker v${VERSION} for ${ARCH}"
  curl -fsSL "$URL" | tar -xzf - -C "$DEST" resource-tracker
  chmod +x "$BIN"
fi

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$DEST" >> "$GITHUB_PATH"
fi
echo "resource-tracker=${BIN}"
