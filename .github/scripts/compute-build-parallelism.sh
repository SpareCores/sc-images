#!/usr/bin/env bash
# Compute build parallelism scaled linearly from a reference RAM size.
# Reference: settings that succeed on a 64 GiB self-hosted arm64 builder (c6g.8xlarge).
#
# Usage:
#   compute-build-parallelism.sh <profile> >> "$GITHUB_OUTPUT"
#   compute-build-parallelism.sh <profile> --env-file /path/to/parallelism.env
#
# Profiles: vllm-gpu | vllm-cpu | buildkit
#
# Runtime parallelism is written to --env-file for BuildKit secrets (excluded from layer cache).
# Reference/canonical values are emitted as docker_* outputs for stable Dockerfile build-args.
set -euo pipefail

PROFILE="${1:?profile required (vllm-gpu|vllm-cpu|buildkit)}"
shift

ENV_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env-file) ENV_FILE="${2:?}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# 64 GiB baseline (c6g.8xlarge arm64 CUDA build).
REFERENCE_RAM_GIB="${REFERENCE_RAM_GIB:-64}"
REFERENCE_MAX_JOBS="${REFERENCE_MAX_JOBS:-32}"
REFERENCE_NVCC_THREADS="${REFERENCE_NVCC_THREADS:-2}"
REFERENCE_CARGO_JOBS="${REFERENCE_CARGO_JOBS:-32}"
REFERENCE_BUILDKIT="${REFERENCE_BUILDKIT:-16}"

COMPILE_GIB_PER_SLOT="${VLLM_COMPILE_GIB_PER_SLOT:-3}"
MAX_JOBS_CAP="${VLLM_MAX_JOBS_CAP:-32}"
NVCC_THREADS_CAP="${VLLM_NVCC_THREADS:-2}"
CARGO_JOBS_CAP="${CARGO_BUILD_JOBS_CAP:-32}"
BUILDKIT_CAP="${BUILDKIT_MAX_PARALLELISM_CAP:-16}"
ZRAM_BUDGET_PERCENT="${ZRAM_COMPILE_BUDGET_PERCENT:-50}"
# physical = MemTotal only (C++ link/compile needs real RAM; zram is not compile budget).
# effective = MemTotal + zram budget (CUDA self-hosted builds).
COMPILE_RAM_SOURCE="${COMPILE_RAM_SOURCE:-effective}"

MEM_MIB=$(awk '/MemTotal:/ {print int($2/1024)}' /proc/meminfo)
SWAP_MIB=$(awk '/SwapTotal:/ {print int($2/1024)}' /proc/meminfo)
NCPU=$(nproc)

ZRAM_BUDGET=$(( SWAP_MIB * ZRAM_BUDGET_PERCENT / 100 ))
EFFECTIVE_MIB=$(( MEM_MIB + ZRAM_BUDGET ))
EFFECTIVE_GIB=$(( (EFFECTIVE_MIB + 512) / 1024 ))

SLOT_MIB=$(( COMPILE_GIB_PER_SLOT * 1024 ))
if [ "$COMPILE_RAM_SOURCE" = "physical" ]; then
  MEM_SLOTS=$(( MEM_MIB / SLOT_MIB ))
else
  MEM_SLOTS=$(( EFFECTIVE_MIB / SLOT_MIB ))
fi
[ "$MEM_SLOTS" -ge 1 ] || MEM_SLOTS=1

# Linear scale from 64 GiB reference; floor keeps small machines conservative.
RATIO_NUM=$EFFECTIVE_GIB
RATIO_DEN=$REFERENCE_RAM_GIB
SCALED_MAX=$(( REFERENCE_MAX_JOBS * RATIO_NUM / RATIO_DEN ))
[ "$SCALED_MAX" -ge 1 ] || SCALED_MAX=1

MAX_JOBS=$SCALED_MAX
if [ "$MAX_JOBS" -gt "$NCPU" ]; then MAX_JOBS=$NCPU; fi
if [ "$MAX_JOBS" -gt "$MEM_SLOTS" ]; then MAX_JOBS=$MEM_SLOTS; fi
if [ "$MAX_JOBS" -gt "$MAX_JOBS_CAP" ]; then MAX_JOBS=$MAX_JOBS_CAP; fi

NVCC_THREADS=$REFERENCE_NVCC_THREADS
if [ "$MAX_JOBS" -lt 4 ]; then NVCC_THREADS=1; fi
if [ "$NVCC_THREADS" -gt "$NVCC_THREADS_CAP" ]; then NVCC_THREADS=$NVCC_THREADS_CAP; fi
if [ "$MAX_JOBS" -lt "$NVCC_THREADS" ]; then MAX_JOBS=$NVCC_THREADS; fi

CARGO_JOBS=$(( REFERENCE_CARGO_JOBS * RATIO_NUM / RATIO_DEN ))
[ "$CARGO_JOBS" -ge 1 ] || CARGO_JOBS=1
if [ "$CARGO_JOBS" -gt "$MAX_JOBS" ]; then CARGO_JOBS=$MAX_JOBS; fi
if [ "$CARGO_JOBS" -gt "$CARGO_JOBS_CAP" ]; then CARGO_JOBS=$CARGO_JOBS_CAP; fi

BUILDKIT_PAR=$(( REFERENCE_BUILDKIT * RATIO_NUM / RATIO_DEN ))
[ "$BUILDKIT_PAR" -ge 2 ] || BUILDKIT_PAR=2
if [ "$BUILDKIT_PAR" -gt "$NCPU" ]; then BUILDKIT_PAR=$NCPU; fi
if [ "$BUILDKIT_PAR" -gt "$BUILDKIT_CAP" ]; then BUILDKIT_PAR=$BUILDKIT_CAP; fi

case "$PROFILE" in
  vllm-gpu)
    DOCKER_MAX_JOBS=$REFERENCE_MAX_JOBS
    DOCKER_NVCC_THREADS=$REFERENCE_NVCC_THREADS
    DOCKER_CARGO_JOBS=$REFERENCE_CARGO_JOBS
    ;;
  vllm-cpu)
    DOCKER_MAX_JOBS=$REFERENCE_MAX_JOBS
    DOCKER_NVCC_THREADS=1
    DOCKER_CARGO_JOBS=$REFERENCE_CARGO_JOBS
    BUILDKIT_PAR=$MAX_JOBS
    if [ "$BUILDKIT_PAR" -gt "$BUILDKIT_CAP" ]; then BUILDKIT_PAR=$BUILDKIT_CAP; fi
    ;;
  buildkit)
    MAX_JOBS=$BUILDKIT_PAR
    NVCC_THREADS=1
    CARGO_JOBS=1
    DOCKER_MAX_JOBS=1
    DOCKER_NVCC_THREADS=1
    DOCKER_CARGO_JOBS=1
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    exit 1
    ;;
esac

echo "Parallelism (${PROFILE}): mem=${MEM_MIB}MiB swap=${SWAP_MIB}MiB effective=${EFFECTIVE_MIB}MiB (${EFFECTIVE_GIB}GiB) ratio=${EFFECTIVE_GIB}/${REFERENCE_RAM_GIB} ncpu=${NCPU} mem_slots=${MEM_SLOTS} ram_source=${COMPILE_RAM_SOURCE} slot_gib=${COMPILE_GIB_PER_SLOT}" >&2
echo "  runtime: max_jobs=${MAX_JOBS} nvcc_threads=${NVCC_THREADS} ninja=$(( MAX_JOBS / NVCC_THREADS )) cargo=${CARGO_JOBS} buildkit=${BUILDKIT_PAR}" >&2
echo "  docker (cache-stable): max_jobs=${DOCKER_MAX_JOBS} nvcc_threads=${DOCKER_NVCC_THREADS:-n/a} cargo=${DOCKER_CARGO_JOBS}" >&2

if [ -n "$ENV_FILE" ]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  printf 'MAX_JOBS=%s\nNVCC_THREADS=%s\nCARGO_BUILD_JOBS=%s\n' \
    "$MAX_JOBS" "$NVCC_THREADS" "$CARGO_JOBS" > "$ENV_FILE"
fi

{
  echo "mem_mib=${MEM_MIB}"
  echo "effective_gib=${EFFECTIVE_GIB}"
  echo "ncpus=${NCPU}"
  echo "max_jobs=${MAX_JOBS}"
  echo "nvcc_threads=${NVCC_THREADS}"
  echo "cargo_build_jobs=${CARGO_JOBS}"
  echo "buildkit_max_parallelism=${BUILDKIT_PAR}"
  echo "docker_max_jobs=${DOCKER_MAX_JOBS}"
  echo "docker_nvcc_threads=${DOCKER_NVCC_THREADS:-1}"
  echo "docker_cargo_build_jobs=${DOCKER_CARGO_JOBS}"
}
