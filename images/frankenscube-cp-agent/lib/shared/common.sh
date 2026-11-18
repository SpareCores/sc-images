#!/usr/bin/env bash
# Shared helpers of k3s-prestart.sh and interlock.sh (sourced, never executed).
# Everything that touches the clock or a path goes through a function or a variable so the bats
# suites can control it: FC_NOW_FILE (epoch seconds), FC_RUN_DIR, FC_STATE_DIR.

FC_RUN_DIR="${FC_RUN_DIR:-/run/frankenscube}"
FC_STATE_DIR="${FC_STATE_DIR:-/var/lib/frankenscube}"

fc_log() {
  printf '%s %s: %s\n' "$(fc_now_iso)" "${FC_LOG_TAG:-k3s-prestart}" "$*" >&2
}

# Current time in epoch seconds. FC_NOW_FILE (tests) wins over the real clock.
fc_now_epoch() {
  if [ -n "${FC_NOW_FILE:-}" ] && [ -r "$FC_NOW_FILE" ]; then
    cat "$FC_NOW_FILE"
  else
    date -u +%s
  fi
}

fc_epoch_to_iso() {
  jq -rn --argjson t "$1" '$t | todate'
}

fc_now_iso() {
  fc_epoch_to_iso "$(fc_now_epoch)"
}

# RFC 3339 UTC ("...Z", optional fraction) to epoch seconds. Prints nothing and fails when the
# value is not parseable.
fc_iso_to_epoch() {
  jq -rn --arg t "$1" '$t | sub("\\.[0-9]+"; "") | fromdateiso8601' 2>/dev/null
}
