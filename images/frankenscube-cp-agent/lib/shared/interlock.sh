#!/usr/bin/env bash
# Peer-health interlock (docs/frankenscube/CONTRACTS.md section 7, PLAN 2.3), shared by
# k3s-prestart.sh and the cp-agent's `reset` directive. Sourced, never executed.
#
#   interlock_ok <self-node> <self-uuid>
#
# Returns 0 only when it is safe to run `--cluster-init`, `--cluster-reset` or a restore:
#   (a) no peer answers https://<peer>:6443/readyz with any HTTP status;
#   (b) no heartbeat with `etcd: member` from another instance UUID is younger than 180 s;
#   (c) peer probes have failed continuously for at least 180 s.
# A fresh `etcd: none` heartbeat does not block. The own UUID is ignored. With an empty PEERS
# (restore drill) there is nobody to wait for: (b) and (c) pass.
#
# Inputs: PEERS (space separated private IPs), $FC_RUN_DIR/heartbeats.json (written by the agent:
# {"<node>": <heartbeat>, ..., "fetched_at": "<RFC3339>"}), $FC_STATE_DIR/probe-fail-since.
# Outputs (globals, set on every call): INTERLOCK_REASON (empty on success),
# INTERLOCK_HEALTHY_PEER (a peer whose /readyz answered 200, or empty).
#
# Heartbeat age: the agent adds `last_modified` (the S3 LastModified of the heartbeat object, S3's
# clock) to every heartbeat it caches; that is what is compared with the local clock. The heartbeat's
# own `ts` (the writer's clock) is only the fallback.
#
# Safety notes:
# - The heartbeat cache must be fresh (fetched_at under 120 s old). A stale cache means the agent
#   cannot reach S3, and "no member heartbeat" would be a guess: the interlock refuses.
# - probe-fail-since holds "<since> <last>" (epoch seconds). A gap of more than 90 s between two
#   failing probes (the probing process was not running, for example across a reboot) restarts
#   the window, so "failed for 180 s" always means probed and failed for 180 s.

INTERLOCK_BLOCK_SECONDS=180
INTERLOCK_FAIL_WINDOW=180
INTERLOCK_PROBE_GAP=90
INTERLOCK_CACHE_MAX_AGE=120

# shellcheck disable=SC2034  # read by the callers
INTERLOCK_REASON=""
# shellcheck disable=SC2034
INTERLOCK_HEALTHY_PEER=""

_interlock_fail_file() {
  printf '%s/probe-fail-since' "$FC_STATE_DIR"
}

# Probes every peer once. Sets INTERLOCK_ANSWERED (first peer that answered, any status) and
# INTERLOCK_HEALTHY_PEER (first peer answering 200).
_interlock_probe_peers() {
  local peer code
  INTERLOCK_ANSWERED=""
  INTERLOCK_HEALTHY_PEER=""
  for peer in ${PEERS:-}; do
    code=$(curl -sk --max-time 3 -o /dev/null -w '%{http_code}' "https://${peer}:6443/readyz" 2>/dev/null) || code="000"
    if [ -z "$code" ] || [ "$code" = "000" ]; then
      continue
    fi
    if [ -z "$INTERLOCK_ANSWERED" ]; then
      INTERLOCK_ANSWERED="$peer"
    fi
    if [ "$code" = "200" ] && [ -z "$INTERLOCK_HEALTHY_PEER" ]; then
      INTERLOCK_HEALTHY_PEER="$peer"
    fi
  done
}

# Updates probe-fail-since after a probe round. Args: now, "answered" or "failed".
_interlock_track_failures() {
  local now=$1 outcome=$2 f since last
  f=$(_interlock_fail_file)
  if [ "$outcome" = "answered" ]; then
    rm -f "$f"
    return 0
  fi
  since=""
  last=""
  if [ -r "$f" ]; then
    read -r since last <"$f" || true
  fi
  case "$since" in '' | *[!0-9]*) since="" ;; esac
  case "$last" in '' | *[!0-9]*) last="" ;; esac
  if [ -z "$since" ] || [ -z "$last" ] || [ $((now - last)) -gt "$INTERLOCK_PROBE_GAP" ]; then
    since=$now
  fi
  mkdir -p "$FC_STATE_DIR"
  printf '%s %s\n' "$since" "$now" >"$f.tmp"
  mv -f "$f.tmp" "$f"
}

# Prints the node names of fresh foreign `member` heartbeats (one per line); returns 2 when the
# heartbeat cache is missing, unparseable or stale.
_interlock_foreign_members() {
  local self_uuid=$1 now=$2 hb="$FC_RUN_DIR/heartbeats.json" fetched fetched_epoch node ts ts_epoch
  [ -r "$hb" ] || return 2
  fetched=$(jq -r '.fetched_at // empty' "$hb" 2>/dev/null) || return 2
  [ -n "$fetched" ] || return 2
  fetched_epoch=$(fc_iso_to_epoch "$fetched") || return 2
  [ -n "$fetched_epoch" ] || return 2
  if [ $((now - fetched_epoch)) -gt "$INTERLOCK_CACHE_MAX_AGE" ]; then
    return 2
  fi
  while IFS=$'\t' read -r node ts; do
    [ -n "$node" ] || continue
    ts_epoch=$(fc_iso_to_epoch "$ts") || ts_epoch=""
    if [ -z "$ts_epoch" ]; then
      # an unparseable timestamp cannot prove the heartbeat old
      printf '%s\n' "$node"
      continue
    fi
    if [ $((now - ts_epoch)) -lt "$INTERLOCK_BLOCK_SECONDS" ]; then
      printf '%s\n' "$node"
    fi
  done < <(jq -r --arg u "$self_uuid" '
      to_entries[]
      | select((.value | type) == "object")
      | select(.value.etcd == "member" and .value.instance_uuid != $u)
      | [.key, (.value.last_modified // .value.ts // "")] | @tsv' "$hb" 2>/dev/null)
  return 0
}

interlock_ok() {
  local self_node=${1:?self node} self_uuid=${2:?self uuid}
  local now members rc since last
  INTERLOCK_REASON=""
  now=$(fc_now_epoch)

  if [ -z "${PEERS:-}" ]; then
    # drill: no peers exist, so nothing can answer and nothing can hold a member heartbeat
    _interlock_probe_peers
    return 0
  fi

  _interlock_probe_peers
  if [ -n "$INTERLOCK_ANSWERED" ]; then
    _interlock_track_failures "$now" answered
    INTERLOCK_REASON="peer-answers:${INTERLOCK_ANSWERED}"
    return 1
  fi
  _interlock_track_failures "$now" failed

  rc=0
  members=$(_interlock_foreign_members "$self_uuid" "$now") || rc=$?
  if [ "$rc" -ne 0 ]; then
    INTERLOCK_REASON="heartbeats-unavailable"
    return 1
  fi
  if [ -n "$members" ]; then
    INTERLOCK_REASON="member-heartbeat:$(printf '%s' "$members" | head -n1)"
    return 1
  fi

  read -r since last <"$(_interlock_fail_file)" || true
  if [ $((now - since)) -lt "$INTERLOCK_FAIL_WINDOW" ]; then
    # shellcheck disable=SC2034
    INTERLOCK_REASON="probe-window:$((now - since))s<${INTERLOCK_FAIL_WINDOW}s"
    return 1
  fi
  : "$self_node"
  return 0
}
