#!/usr/bin/env bash
# Emit BuildKit cache import/export specs (registry + GHA) for build-level.yml.
#
# Usage: assemble-build-cache.sh <folder> <arch> <rt_consumer> <has_prepare> <force_rebuild>
# Writes multiline cache_from and cache_to to GITHUB_OUTPUT when set.
set -euo pipefail

folder="${1:?folder}"
arch="${2:?arch}"
rt_consumer="${3:?rt_consumer}"
has_prepare="${4:?has_prepare}"
force_rebuild="${5:?force_rebuild}"

registry="${REGISTRY:-ghcr.io}"

if [ "$force_rebuild" = "true" ] && [ "$rt_consumer" = "true" ]; then
  exit 0
fi

scope="${folder}-${arch}"
if [ "$has_prepare" = "true" ] && [ -f vllm-common/VLLM_VERSION ]; then
  vllm_version="$(tr -d '[:space:]' < vllm-common/VLLM_VERSION)"
  [ -n "$vllm_version" ] && scope="${scope}-v${vllm_version}"
fi

registry_ref="${registry}/sparecores/${folder}:buildcache-${arch}"
cache_from=(
  "type=registry,ref=${registry_ref}"
  "type=gha,scope=${scope}"
)
cache_to=(
  "type=registry,ref=${registry_ref},mode=max,image-manifest=true"
  "type=gha,scope=${scope},mode=max"
)

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "cache_from<<EOF"
    printf '%s\n' "${cache_from[@]}"
    echo "EOF"
    echo "cache_to<<EOF"
    printf '%s\n' "${cache_to[@]}"
    echo "EOF"
    echo "gha_cache_scope=${scope}"
  } >> "$GITHUB_OUTPUT"
else
  printf 'cache_from:\n'
  printf '  %s\n' "${cache_from[@]}"
  printf 'cache_to:\n'
  printf '  %s\n' "${cache_to[@]}"
  echo "gha_cache_scope=${scope}"
fi
