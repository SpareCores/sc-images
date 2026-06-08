#!/usr/bin/env bash
# Build and push one image with docker buildx while resource-tracker monitors the host.
#
# Required env:
#   BUILD_CONTEXT, BUILD_DOCKERFILE, BUILD_PLATFORM, BUILD_TAGS
# Optional env:
#   BUILD_TARGET, BUILD_LABELS, BUILD_PULL, BUILD_NO_CACHE, BUILD_PROVENANCE_FALSE,
#   BUILD_ARGS_FILE, BUILD_SECRET, BUILD_CACHE_FROM, BUILD_CACHE_TO
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

: "${BUILD_CONTEXT:?BUILD_CONTEXT required}"
: "${BUILD_DOCKERFILE:?BUILD_DOCKERFILE required}"
: "${BUILD_PLATFORM:?BUILD_PLATFORM required}"
: "${BUILD_TAGS:?BUILD_TAGS required}"

args=(
  docker buildx build
  --file "$BUILD_DOCKERFILE"
  --platform "$BUILD_PLATFORM"
  --push
)

[ -n "${BUILD_TARGET:-}" ] && args+=(--target "$BUILD_TARGET")
[ "${BUILD_PULL:-}" = "true" ] && args+=(--pull)
[ "${BUILD_NO_CACHE:-}" = "true" ] && args+=(--no-cache)
[ "${BUILD_PROVENANCE_FALSE:-}" = "true" ] && args+=(--provenance=false --sbom=false)

while IFS= read -r tag; do
  [ -z "$tag" ] && continue
  args+=(-t "$tag")
done <<< "$BUILD_TAGS"

while IFS= read -r label; do
  [ -z "$label" ] && continue
  args+=(--label "$label")
done <<< "${BUILD_LABELS:-}"

if [ -n "${BUILD_ARGS_FILE:-}" ] && [ -s "$BUILD_ARGS_FILE" ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    args+=(--build-arg "$line")
  done < "$BUILD_ARGS_FILE"
fi

if [ -n "${BUILD_SECRET:-}" ]; then
  args+=(--secret "$BUILD_SECRET")
fi

[ -n "${BUILD_CACHE_FROM:-}" ] && args+=(--cache-from "$BUILD_CACHE_FROM")
[ -n "${BUILD_CACHE_TO:-}" ] && args+=(--cache-to "$BUILD_CACHE_TO")

args+=("$BUILD_CONTEXT")

echo "build-push-image: ${args[*]}"
bash "${SCRIPTS}/run-with-resource-tracker.sh" "${args[@]}"
