#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# cp-agent runtime: environment, logging, background jobs, host access. Sourced by bin/cp-agent.

FC_LOG_TAG=cp-agent
FC_HS_DIR="${FC_HS_DIR:-/var/lib/headscale}"
FC_K3S_DIR="${FC_K3S_DIR:-/var/lib/rancher/k3s}"
FC_K3S_BIN="${FC_K3S_BIN:-/opt/bin/k3s}"
FC_LEGO_DIR="${FC_LEGO_DIR:-/var/lib/frankenscube/lego}"

# Timing (PLAN 2.5): renew every 10 s, lease 30 s, self-fence at 20 s, 10 s skew margin on takeover.
LEASE_RENEW_SECONDS=10
LEASE_TTL_SECONDS=30
LEASE_FENCE_SECONDS=20
LEASE_SKEW_SECONDS=10
GENERATION_DELAY_SECONDS=240

agent_init() {
  FC_NODE_ENV="${FC_NODE_ENV:-/etc/frankenscube/node.env}"
  if [ -r "$FC_NODE_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$FC_NODE_ENV"
    set +a
  fi
  # docker --env-file keeps the quotes of PEERS="a b"
  PEERS="${PEERS:-}"
  PEERS="${PEERS//\"/}"
  : "${NODE_NAME:?NODE_NAME missing}"
  : "${CLUSTER:?CLUSTER missing}"
  : "${CONTROL_BUCKET:?CONTROL_BUCKET missing}"
  : "${PREFIX:?PREFIX missing}"
  AWS_REGION="${AWS_REGION:-us-east-1}"
  NODE_IP="${NODE_IP:-}"
  GENERATION="${GENERATION:-1}"
  HS_NAME="${HS_NAME:-}"
  DISABLE_AGENT="${DISABLE_AGENT:-false}"
  export PEERS
  umask 077
  AGENT_RUN="$FC_RUN_DIR/agent"
  HS_DB="$FC_HS_DIR/db.sqlite"
  mkdir -p "$AGENT_RUN/bg" "$FC_RUN_DIR/marker-requests" "$FC_RUN_DIR/secrets" "$FC_STATE_DIR" "$FC_HS_DIR"
  chmod 0700 "$FC_RUN_DIR/secrets" 2>/dev/null || true
  if [ ! -s "$FC_STATE_DIR/instance-uuid" ]; then
    cat /proc/sys/kernel/random/uuid >"$FC_STATE_DIR/instance-uuid"
  fi
  INSTANCE_UUID=$(head -n1 "$FC_STATE_DIR/instance-uuid")
}

# Host actions (systemctl, tailscale, k3s) run in the host namespaces.
host_exec() {
  nsenter -t 1 -m -u -i -n -- "$@"
}

# Step wrapper: a failing step is logged and never stops the loop (fail-static).
step() {
  local name=$1 rc=0
  shift
  "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    fc_log "step $name failed (rc=$rc)"
  fi
  return 0
}

# --- background jobs (slow work must never block the 10 s lease tick) --------------------------

bg_dir() { printf '%s/bg' "$AGENT_RUN"; }

# bg_start <name> <function> [args...]: runs the function in a subshell; writes <name>.rc at the end.
bg_start() {
  local name=$1 d
  shift
  d=$(bg_dir)
  rm -f "$d/$name.rc"
  (
    rc=0
    "$@" >>"$d/$name.log" 2>&1 || rc=$?
    echo "$rc" >"$d/$name.rc.tmp"
    mv -f "$d/$name.rc.tmp" "$d/$name.rc"
  ) &
  echo $! >"$d/$name.pid"
  disown "$!" 2>/dev/null || true
}

# prints running | done | none
bg_state() {
  local name=$1 d pid
  d=$(bg_dir)
  if [ -r "$d/$name.rc" ]; then
    echo "done"
    return 0
  fi
  if [ -r "$d/$name.pid" ]; then
    pid=$(cat "$d/$name.pid")
    if kill -0 "$pid" 2>/dev/null; then
      echo running
      return 0
    fi
  fi
  echo none
}

bg_rc() { cat "$(bg_dir)/$1.rc"; }

bg_clear() {
  local d
  d=$(bg_dir)
  rm -f "$d/$1.rc" "$d/$1.pid"
}

bg_kill() {
  local d pid
  d=$(bg_dir)
  if [ -r "$d/$1.pid" ]; then
    pid=$(cat "$d/$1.pid")
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  fi
  bg_clear "$1"
}
