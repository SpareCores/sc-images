#!/usr/bin/env bash
# Pre-build hook for vllm-gpu-base: clone pinned vLLM, compute RAM-aware build
# parallelism, patch the upstream CUDA Dockerfile, and stage a BuildKit secret.
#
# Runs from the repo root. Consumed by .github/workflows/build-level.yml when an
# image folder contains prepare.sh. Emits to $GITHUB_OUTPUT:
#   buildkit_max_parallelism, secret_file
# and appends runtime build-args to .cache/extra-build-args.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")/../../.github/scripts" && pwd)"
ROOT="$(pwd)"
SRC="${ROOT}/vllm-common/.vllm-src"
CACHE="${ROOT}/.cache"
mkdir -p "$CACHE"

VLLM_VERSION="$(tr -d '[:space:]' < vllm-common/VLLM_VERSION)"

PAR="${CACHE}/parallelism.out"
bash "${SCRIPTS}/compute-build-parallelism.sh" vllm-gpu > "$PAR"
get() { sed -n "s/^$1=//p" "$PAR" | head -1; }

MAX_JOBS="$(get max_jobs)"
NVCC_THREADS="$(get nvcc_threads)"
CARGO_JOBS="$(get cargo_build_jobs)"
DOCKER_MAX_JOBS="$(get docker_max_jobs)"
DOCKER_NVCC_THREADS="$(get docker_nvcc_threads)"
DOCKER_CARGO_JOBS="$(get docker_cargo_build_jobs)"
BUILDKIT="$(get buildkit_max_parallelism)"

# Runtime parallelism via BuildKit secret (excluded from layer cache).
printf 'MAX_JOBS=%s\nNVCC_THREADS=%s\nCARGO_BUILD_JOBS=%s\n' \
  "$MAX_JOBS" "$NVCC_THREADS" "$CARGO_JOBS" > "${CACHE}/vllm-parallelism.env"

if [ ! -d "${SRC}/.git" ]; then
  git clone --depth 1 --branch "v${VLLM_VERSION}" https://github.com/vllm-project/vllm.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "v${VLLM_VERSION}"
  git -C "$SRC" checkout -f "v${VLLM_VERSION}"
fi

( cd "$SRC" && export SCCACHE_PREFIX="${SCCACHE_PREFIX:-${IMAGE_FOLDER:-vllm-gpu-base}/${ARCH:-arm64}}" && \
  bash "${SCRIPTS}/tune-vllm-gpu-dockerfile.sh" \
  docker/Dockerfile \
  "$VLLM_VERSION" \
  "$DOCKER_MAX_JOBS" \
  "$DOCKER_NVCC_THREADS" \
  "$DOCKER_CARGO_JOBS" )

# DEBUG: dump patched Dockerfile sections with sccache for CI visibility
echo "::group::DEBUG patched GPU Dockerfile (sccache lines)"
grep -n 'sccache\|SCCACHE\|RUSTC_WRAPPER\|USE_SCCACHE' "$SRC/docker/Dockerfile" || true
echo "::endgroup::"
echo "::group::DEBUG patched GPU Dockerfile (RUN lines with sccache install)"
grep -n -A2 'Installing sccache' "$SRC/docker/Dockerfile" || true
echo "::endgroup::"

# Cache-stable reference build-args (runtime values come from the secret).
{
  echo "max_jobs=${DOCKER_MAX_JOBS}"
  echo "nvcc_threads=${DOCKER_NVCC_THREADS}"
  echo "vllm_ci_cache_bust=${VLLM_VERSION}"
  echo "torch_cuda_arch_list=9.0 10.0+PTX"
  echo "RUN_WHEEL_CHECK=false"
} >> "${CACHE}/extra-build-args"

if [ "${SCCACHE_ENABLED:-}" = "true" ]; then
  : "${SCCACHE_BUCKET:?SCCACHE_BUCKET required when SCCACHE is enabled}"
  prefix="${SCCACHE_PREFIX:-${IMAGE_FOLDER:-vllm-gpu-base}/${ARCH:-arm64}}"
  {
    echo "USE_SCCACHE=1"
    echo "RUSTC_WRAPPER=/usr/local/bin/sccache-wrapper"
    echo "SCCACHE_BUCKET_NAME=${SCCACHE_BUCKET}"
    echo "SCCACHE_REGION_NAME=${SCCACHE_REGION:-us-west-2}"
    echo "SCCACHE_S3_NO_CREDENTIALS=0"
    echo "SCCACHE_S3_KEY_PREFIX=${prefix}"
  } >> "${CACHE}/extra-build-args"
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "buildkit_max_parallelism=${BUILDKIT}"
    echo "secret_file=.cache/vllm-parallelism.env"
  } >> "$GITHUB_OUTPUT"
fi
