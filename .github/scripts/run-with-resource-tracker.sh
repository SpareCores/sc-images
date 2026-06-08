#!/usr/bin/env bash
# Monitor the host while a command runs.
#
# docker buildx build only talks to buildkitd; compiles run outside the CLI
# process tree, so shell-wrapper mode would miss almost all build CPU/RAM.
# Standalone mode (no wrapped command, no --pid) samples system-wide metrics.
#
# Expects TRACKER_* metadata and optional SENTINEL_API_TOKEN in the environment.
set -euo pipefail

if ! command -v resource-tracker >/dev/null 2>&1; then
  echo "run-with-resource-tracker: resource-tracker not found in PATH" >&2
  exit 1
fi

tags=()
[ -n "${GITHUB_RUN_ATTEMPT:-}" ] && tags+=(--tag "github_run_attempt=${GITHUB_RUN_ATTEMPT}")
[ -n "${GITHUB_SHA:-}" ] && tags+=(--tag "github_sha=${GITHUB_SHA}")
[ -n "${GITHUB_REF_NAME:-}" ] && tags+=(--tag "github_ref=${GITHUB_REF_NAME}")
[ -n "${MATRIX_ARCH:-}" ] && tags+=(--tag "arch=${MATRIX_ARCH}")

quiet=()
case "${TRACKER_QUIET:-}" in
  true|1|yes) quiet+=(--quiet) ;;
esac

resource-tracker "${quiet[@]}" "${tags[@]}" &
rt_pid=$!

stop_tracker() {
  kill -TERM "$rt_pid" 2>/dev/null || true
  wait "$rt_pid" 2>/dev/null || true
}

trap stop_tracker EXIT

set +e
"$@"
exit_code=$?
set -e

stop_tracker
trap - EXIT
exit "$exit_code"
