#!/usr/bin/env bash
# Pre-build hook for vllm-cpu-base-avx2: clone pinned vLLM, compute RAM-aware
# build parallelism (oneDNN/C++ objects are RAM-heavy -> sized on physical RAM),
# patch the upstream CPU Dockerfile, and stage a BuildKit secret.
#
# Runs from the repo root. Emits to $GITHUB_OUTPUT:
#   buildkit_max_parallelism, secret_file
# and appends runtime build-args to .cache/extra-build-args.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")/../../.github/scripts" && pwd)"
ROOT="$(pwd)"
SRC="${ROOT}/vllm-common/.vllm-src"
CACHE="${ROOT}/.cache"
mkdir -p "$CACHE"

VLLM_VERSION="$(tr -d '[:space:]' < vllm-common/VLLM_VERSION)"

# CPU compile is RAM-heavy and has no nvcc; size slots on physical RAM only.
export VLLM_COMPILE_GIB_PER_SLOT="${VLLM_CPU_COMPILE_GIB_PER_SLOT:-6}"
export COMPILE_RAM_SOURCE=physical
export VLLM_NVCC_THREADS=1

PAR="${CACHE}/parallelism.out"
bash "${SCRIPTS}/compute-build-parallelism.sh" vllm-cpu > "$PAR"
get() { sed -n "s/^$1=//p" "$PAR" | head -1; }

MAX_JOBS="$(get max_jobs)"
NVCC_THREADS="$(get nvcc_threads)"
CARGO_JOBS="$(get cargo_build_jobs)"
DOCKER_MAX_JOBS="$(get docker_max_jobs)"
DOCKER_CARGO_JOBS="$(get docker_cargo_build_jobs)"
BUILDKIT="$(get buildkit_max_parallelism)"

printf 'MAX_JOBS=%s\nNVCC_THREADS=%s\nCARGO_BUILD_JOBS=%s\n' \
  "$MAX_JOBS" "$NVCC_THREADS" "$CARGO_JOBS" > "${CACHE}/vllm-parallelism.env"

if [ ! -d "${SRC}/.git" ]; then
  git clone --depth 1 --branch "v${VLLM_VERSION}" https://github.com/vllm-project/vllm.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "v${VLLM_VERSION}"
  git -C "$SRC" checkout -f "v${VLLM_VERSION}"
fi

( cd "$SRC" && export SCCACHE_PREFIX="${SCCACHE_PREFIX:-${IMAGE_FOLDER:-vllm-cpu-base-avx2}/${ARCH:-amd64}}" && \
  bash "${SCRIPTS}/tune-vllm-cpu-dockerfile.sh" \
  docker/Dockerfile.cpu \
  "$VLLM_VERSION" \
  "$DOCKER_MAX_JOBS" \
  "$DOCKER_CARGO_JOBS" )

# DEBUG: dump patched Dockerfile sections with sccache for CI visibility
echo "::group::DEBUG patched CPU Dockerfile (sccache lines)"
grep -n 'sccache\|SCCACHE\|RUSTC_WRAPPER\|USE_SCCACHE' "$SRC/docker/Dockerfile.cpu" || true
echo "::endgroup::"
echo "::group::DEBUG patched CPU Dockerfile (RUN lines with sccache install)"
grep -n -A2 'Installing sccache' "$SRC/docker/Dockerfile.cpu" || true
echo "::endgroup::"

{
  echo "max_jobs=${DOCKER_MAX_JOBS}"
} >> "${CACHE}/extra-build-args"

if [ "${SCCACHE_ENABLED:-}" = "true" ]; then
  : "${SCCACHE_BUCKET:?SCCACHE_BUCKET required when SCCACHE is enabled}"
  prefix="${SCCACHE_PREFIX:-${IMAGE_FOLDER:-vllm-cpu-base-avx2}/${ARCH:-amd64}}"
  {
    echo "USE_SCCACHE=1"
    echo "RUSTC_WRAPPER=/usr/local/bin/sc-rust-wrap"
    echo "SCCACHE_BUCKET_NAME=${SCCACHE_BUCKET}"
    echo "SCCACHE_REGION_NAME=${SCCACHE_REGION:-us-west-2}"
    echo "SCCACHE_S3_NO_CREDENTIALS=0"
    echo "SCCACHE_S3_KEY_PREFIX=${prefix}"
    if [ "${SCCACHE_VERBOSE:-}" = "1" ]; then
      echo "SCCACHE_VERBOSE=1"
    fi
  } >> "${CACHE}/extra-build-args"
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "buildkit_max_parallelism=${BUILDKIT}"
    echo "secret_file=.cache/vllm-parallelism.env"
  } >> "$GITHUB_OUTPUT"
fi
