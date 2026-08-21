#!/usr/bin/env bash
# Normalize file ownership in a Docker build context so BuildKit's cache key for
# `COPY` is reproducible across runner types.
#
# BuildKit hashes local build contexts with a V1 tarsum over name, mode, uid,
# gid, size, typeflag, linkname, xattrs and content. mtime is excluded, uid/gid
# are not. GitHub-hosted runners check out as `runner`, the self-hosted builders
# as `ubuntu`, so byte-identical sources hash differently: every `COPY` from the
# context misses, and with it the whole graph below it. Chowning to a fixed
# uid:gid removes the only field that varies.
#
# Leaves the tree root-owned, so build-level.yml reclaims workspace ownership
# before actions/checkout runs `git clean -ffdx`.
set -euo pipefail

DIR="${1:?context directory}"
OWNER="${CONTEXT_CANONICAL_OWNER:-0:0}"

if [ ! -d "$DIR" ]; then
  echo "normalize-context-ownership: no such directory: $DIR" >&2
  exit 1
fi

if sudo -n chown -R "$OWNER" "$DIR" 2>/dev/null; then
  echo "normalize-context-ownership: ${DIR} -> ${OWNER}"
else
  echo "normalize-context-ownership: WARNING could not chown ${DIR} (no passwordless sudo);" \
    "layer cache will not be shared across runner types" >&2
fi
