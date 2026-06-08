#!/usr/bin/env bash
# Resolve build configuration for one image folder + arch from per-folder metadata.
#
# Usage: read-image-config.sh <folder> <arch>   (run from repo root)
#
# Metadata files (all optional) in images/<folder>/:
#   DEPENDS_ON  - image folder names this image needs published first (one per line)
#   PLATFORMS   - arches to build (default: amd64 arm64)
#   CONTEXT     - build context, repo-relative (default: images/<folder>)
#   DOCKERFILE  - Dockerfile path, repo-relative (default: images/<folder>/Dockerfile)
#   TARGET      - build target stage (default: none)
#   BUILD_ARGS  - KEY=VALUE lines; tokens ${ARCH} ${VLLM_VERSION} ${RESOURCE_TRACKER_VERSION}
#   ZRAM        - enable compressed swap on the builder (true/1/yes, or PERCENT e.g. 125); default off
#   prepare.sh  - pre-build hook (presence reported as has_prepare=true)
#
# Emits key=value lines (for $GITHUB_OUTPUT) and writes resolved static
# build-args to .cache/build-args.
set -euo pipefail

folder="${1:?folder required}"
arch="${2:?arch required}"
images_dir="${IMAGES_DIR:-images}"
fdir="$images_dir/$folder"

read_scalar() {
  [ -f "$1" ] || return 0
  tr -d '\r' < "$1" | sed 's/#.*//' | tr -d '[:space:]'
}
read_list() {
  sed 's/#.*//' "$1" 2>/dev/null | tr -s ' \t' '\n' | sed '/^[[:space:]]*$/d'
}

context="$(read_scalar "$fdir/CONTEXT")"; [ -n "$context" ] || context="$fdir"
dockerfile="$(read_scalar "$fdir/DOCKERFILE")"; [ -n "$dockerfile" ] || dockerfile="$fdir/Dockerfile"
target="$(read_scalar "$fdir/TARGET")"

if [ -f "$fdir/PLATFORMS" ]; then
  platforms="$(read_list "$fdir/PLATFORMS" | tr '\n' ' ')"
else
  platforms="amd64 arm64"
fi
platforms="$(echo $platforms)"  # collapse whitespace

deps=()
[ -f "$fdir/DEPENDS_ON" ] && mapfile -t deps < <(read_list "$fdir/DEPENDS_ON")

has_prepare=false
[ -f "$fdir/prepare.sh" ] && has_prepare=true

zram=false
zram_percent="${ZRAM_PERCENT:-125}"
if [ -f "$fdir/ZRAM" ]; then
  zram_val="$(read_scalar "$fdir/ZRAM")"
  case "${zram_val,,}" in
    true|1|yes|on)
      zram=true
      ;;
    [0-9]*)
      if [[ "$zram_val" =~ ^[0-9]+$ ]]; then
        zram=true
        zram_percent="$zram_val"
      fi
      ;;
  esac
fi

# An image is a resource-tracker consumer if it is resource-tracker or (transitively) depends on it.
is_rt_consumer() {
  local f="$1" d
  [ "$f" = "resource-tracker" ] && return 0
  local dd=()
  [ -f "$images_dir/$f/DEPENDS_ON" ] && mapfile -t dd < <(read_list "$images_dir/$f/DEPENDS_ON")
  for d in "${dd[@]}"; do
    is_rt_consumer "$d" && return 0
  done
  return 1
}
rt_consumer=false
is_rt_consumer "$folder" && rt_consumer=true

# Pull fresh bases when this image depends on a just-built image (or resource-tracker).
pull=false
{ [ "${#deps[@]}" -gt 0 ] || [ "$rt_consumer" = true ]; } && pull=true

vllm_version="$(read_scalar vllm-common/VLLM_VERSION || true)"
rt_version="$(read_scalar "$images_dir/resource-tracker/RESOURCE_TRACKER_VERSION" || true)"

mkdir -p .cache
: > .cache/build-args
if [ -f "$fdir/BUILD_ARGS" ]; then
  while IFS= read -r line; do
    line="${line%$'\r'}"
    [ -z "${line//[[:space:]]/}" ] && continue
    case "$line" in \#*) continue ;; esac
    line="${line//\$\{ARCH\}/$arch}"
    line="${line//\$\{VLLM_VERSION\}/$vllm_version}"
    line="${line//\$\{RESOURCE_TRACKER_VERSION\}/$rt_version}"
    echo "$line" >> .cache/build-args
  done < "$fdir/BUILD_ARGS"
fi

{
  echo "context=$context"
  echo "dockerfile=$dockerfile"
  echo "target=$target"
  echo "platforms=$platforms"
  echo "pull=$pull"
  echo "rt_consumer=$rt_consumer"
  echo "has_prepare=$has_prepare"
  echo "zram=$zram"
  echo "zram_percent=$zram_percent"
}
